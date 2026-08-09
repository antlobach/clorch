(ns clorch.distributed
  "Multi-process distributed runtime.

  One JVM represents one rank. NCCL collectives use JavaCPP's public CUDA
  bindings directly; this avoids a known ProcessGroup intrusive-pointer defect
  in the current JavaCPP PyTorch preset."
  (:refer-clojure :exclude [send])
  (:require [clojure.string :as str]
            [clorch.platform :as platform])
  (:import [java.util.concurrent TimeoutException]
           [org.bytedeco.javacpp SizeTPointer]
           [org.bytedeco.javacpp.chrono Milliseconds]
           [org.bytedeco.pytorch Store StringVector TCPStore TCPStoreOptions Tensor]))

(platform/configure-platform-extension!)

(declare destroy-process-group!)

(defrecord ProcessGroupContext
           [^Store store
            backend-state
            backend
            rank
            world-size
            timeout-ms
            master-address
            master-port
            local-rank
            operation-counter]
  java.lang.AutoCloseable
  (close [this]
    (destroy-process-group! this)))

(defrecord StoreWork [task result])

(defonce ^:private default-context (atom nil))

(defn- environment [environment-name]
  (System/getenv environment-name))

(defn- parse-long! [label value]
  (try
    (Long/parseLong (str value))
    (catch NumberFormatException cause
      (throw (ex-info (str label " must be an integer")
                      {:label label :value value}
                      cause)))))

(defn- option-or-env [options option-key environment-key default-value]
  (if (contains? options option-key)
    (get options option-key)
    (or (environment environment-key) default-value)))

(defn- validate-rendezvous!
  [{:keys [backend rank world-size timeout-ms master-address master-port local-rank]}]
  (when-not (= :nccl backend)
    (throw (ex-info "The current JavaCPP release supports Clorch distributed execution through :nccl only"
                    {:backend backend
                     :reason :javacpp-process-group-construction-defect})))
  (when-not (pos? world-size)
    (throw (ex-info "World size must be positive" {:world-size world-size})))
  (when-not (<= 0 rank (dec world-size))
    (throw (ex-info "Rank must be within the process group"
                    {:rank rank :world-size world-size})))
  (when-not (<= 0 local-rank)
    (throw (ex-info "Local rank must be non-negative" {:local-rank local-rank})))
  (when-not (pos? timeout-ms)
    (throw (ex-info "Process-group timeout must be positive"
                    {:timeout-ms timeout-ms})))
  (when (str/blank? master-address)
    (throw (ex-info "Master address must not be blank" {})))
  (when-not (<= 1 master-port 65535)
    (throw (ex-info "Master port must be between 1 and 65535"
                    {:master-port master-port}))))

(defn- rendezvous-options [options]
  (let [resolved {:backend (keyword (option-or-env options :backend "CLORCH_DIST_BACKEND" :nccl))
                  :rank (parse-long! "rank" (option-or-env options :rank "RANK" 0))
                  :world-size (parse-long! "world-size" (option-or-env options :world-size "WORLD_SIZE" 1))
                  :local-rank (parse-long! "local-rank" (option-or-env options :local-rank "LOCAL_RANK" 0))
                  :timeout-ms (parse-long! "timeout-ms" (option-or-env options :timeout-ms "CLORCH_DIST_TIMEOUT_MS" 300000))
                  :master-address (str (option-or-env options :master-address "MASTER_ADDR" "127.0.0.1"))
                  :master-port (parse-long! "master-port" (option-or-env options :master-port "MASTER_PORT" 29500))}]
    (validate-rendezvous! resolved)
    resolved))

(defn- tcp-store
  [{:keys [rank world-size timeout-ms master-address master-port]}]
  (let [timeout (Milliseconds. timeout-ms)
        workers (SizeTPointer. (long-array [world-size]))
        options (doto (TCPStoreOptions.)
                  (.port (unchecked-short master-port))
                  (.isServer (zero? rank))
                  (.numWorkers workers)
                  (.waitWorkers true)
                  (.multiTenant false)
                  (.timeout timeout)
                  (.useLibUV true))]
    (TCPStore. master-address options)))

(defn init-process-group!
  "Initializes this JVM's default NCCL process group.

  Options default from RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, and
  MASTER_PORT. One JVM must be launched per GPU."
  ([] (init-process-group! {}))
  ([options]
   (locking default-context
     (when @default-context
       (throw (ex-info "A default process group is already initialized"
                       {:context @default-context})))
     (let [{:keys [rank world-size local-rank] :as resolved}
           (rendezvous-options options)]
       (when-not ((requiring-resolve 'clorch.cuda/available?))
         (throw (ex-info "NCCL requires an available CUDA runtime"
                         {:backend :nccl :rank rank})))
       (let [store (tcp-store resolved)]
         (try
           (let [backend-state
                 ((requiring-resolve 'clorch.distributed.nccl/create!)
                  store rank world-size local-rank)
                 context (map->ProcessGroupContext
                          (assoc resolved
                                 :store store
                                 :backend-state backend-state
                                 :operation-counter (atom 0)))]
             (reset! default-context context)
             context)
           (catch Throwable cause
             (try (.close store) (catch Throwable _))
             (throw cause))))))))

(defn initialized?
  "Returns true when this JVM owns a live default process group."
  []
  (boolean @default-context))

(defn current-process-group
  "Returns the default ProcessGroupContext, throwing when uninitialized."
  []
  (or @default-context
      (throw (ex-info "Distributed process group is not initialized" {}))))

(defn- context-or-current [context]
  (or context (current-process-group)))

(defn rank
  "Returns this worker's global rank."
  ([] (:rank (current-process-group)))
  ([context] (:rank (context-or-current context))))

(defn local-rank
  "Returns this worker's node-local rank."
  ([] (:local-rank (current-process-group)))
  ([context] (:local-rank (context-or-current context))))

(defn world-size
  "Returns the number of ranks in the process group."
  ([] (:world-size (current-process-group)))
  ([context] (:world-size (context-or-current context))))

(defn backend
  "Returns the process group's backend keyword."
  ([] (:backend (current-process-group)))
  ([context] (:backend (context-or-current context))))

(defn rank-zero?
  "Returns true for global rank zero."
  ([] (zero? (rank)))
  ([context] (zero? (rank context))))

(defn- submit-store-work [operation result async?]
  (if async?
    (->StoreWork (future (operation)) result)
    (do (operation) result)))

(defn await!
  "Waits for asynchronous distributed work and returns its result."
  ([work] (await! work (:timeout-ms (current-process-group))))
  ([work timeout-ms]
   (cond
     (instance? StoreWork work)
     (let [outcome (deref (:task work) timeout-ms ::timeout)]
       (when (= ::timeout outcome)
         (throw (TimeoutException.
                 (str "Distributed operation exceeded " timeout-ms " ms"))))
       (:result work))

     (and (map? work) (contains? work :event))
     ((requiring-resolve 'clorch.distributed.nccl/await!) work timeout-ms)

     :else
     (throw (ex-info "Unknown distributed work value"
                     {:value-type (some-> work class str)})))))

(defn- finish-work [work async? timeout-ms]
  (if async?
    work
    (await! work timeout-ms)))

(defn all-reduce!
  "Reduces CUDA tensors across every rank in place."
  ([tensors] (all-reduce! nil tensors {}))
  ([tensors options] (all-reduce! nil tensors options))
  ([context tensors {:keys [op async? timeout-ms]
                     :or {op :sum async? false}}]
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         work ((requiring-resolve 'clorch.distributed.nccl/all-reduce!)
               (:backend-state context) tensors op)]
     (finish-work work async? timeout-ms))))

(defn broadcast!
  "Broadcasts CUDA tensors from root rank in place."
  ([tensors] (broadcast! nil tensors {}))
  ([tensors options] (broadcast! nil tensors options))
  ([context tensors {:keys [root-rank root-tensor async? timeout-ms]
                     :or {root-rank 0 root-tensor 0 async? false}}]
   (when-not (zero? root-tensor)
     (throw (ex-info "Direct NCCL broadcast supports root tensor index zero only"
                     {:root-tensor root-tensor})))
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         work ((requiring-resolve 'clorch.distributed.nccl/broadcast!)
               (:backend-state context) tensors root-rank)]
     (finish-work work async? timeout-ms))))

(defn reduce!
  "Reduces CUDA tensors in place onto root rank."
  ([tensors] (reduce! nil tensors {}))
  ([tensors options] (reduce! nil tensors options))
  ([context tensors {:keys [op root-rank root-tensor async? timeout-ms]
                     :or {op :sum root-rank 0 root-tensor 0 async? false}}]
   (when-not (zero? root-tensor)
     (throw (ex-info "Direct NCCL reduce supports root tensor index zero only"
                     {:root-tensor root-tensor})))
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         work ((requiring-resolve 'clorch.distributed.nccl/reduce!)
               (:backend-state context) tensors op root-rank)]
     (finish-work work async? timeout-ms))))

(defn all-gather-into!
  "Gathers equal-sized CUDA inputs into a preallocated output tensor."
  ([output input] (all-gather-into! nil output input {}))
  ([output input options] (all-gather-into! nil output input options))
  ([context ^Tensor output ^Tensor input {:keys [async? timeout-ms]
                                          :or {async? false}}]
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         work ((requiring-resolve 'clorch.distributed.nccl/all-gather-into!)
               (:backend-state context) output input (:world-size context))]
     (finish-work work async? timeout-ms))))

(defn reduce-scatter-into!
  "Reduces an input CUDA tensor and scatters equal chunks into output."
  ([output input] (reduce-scatter-into! nil output input {}))
  ([output input options] (reduce-scatter-into! nil output input options))
  ([context ^Tensor output ^Tensor input {:keys [op async? timeout-ms]
                                          :or {op :sum async? false}}]
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         work ((requiring-resolve 'clorch.distributed.nccl/reduce-scatter-into!)
               (:backend-state context) output input (:world-size context) op)]
     (finish-work work async? timeout-ms))))

(defn all-to-all-single!
  "Performs all-to-all into preallocated CUDA output with optional split sizes."
  ([output input] (all-to-all-single! nil output input {}))
  ([output input options] (all-to-all-single! nil output input options))
  ([context ^Tensor output ^Tensor input
    {:keys [output-split-sizes input-split-sizes async? timeout-ms]
     :or {output-split-sizes [] input-split-sizes [] async? false}}]
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         work ((requiring-resolve 'clorch.distributed.nccl/all-to-all-single!)
               (:backend-state context) output input (:world-size context)
               output-split-sizes input-split-sizes)]
     (finish-work work async? timeout-ms))))

(defn send
  "Sends CUDA tensors to a peer rank. Returns input after completion or work."
  ([tensors destination] (send nil tensors destination {}))
  ([tensors destination options] (send nil tensors destination options))
  ([context tensors destination {:keys [tag async? timeout-ms]
                                 :or {tag 0 async? false}}]
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         work ((requiring-resolve 'clorch.distributed.nccl/send!)
               (:backend-state context) tensors destination tag)]
     (finish-work work async? timeout-ms))))

(defn receive!
  "Receives CUDA tensors from a peer rank in place."
  ([tensors source] (receive! nil tensors source {}))
  ([tensors source options] (receive! nil tensors source options))
  ([context tensors source {:keys [tag async? timeout-ms]
                            :or {tag 0 async? false}}]
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         work ((requiring-resolve 'clorch.distributed.nccl/receive!)
               (:backend-state context) tensors source tag)]
     (finish-work work async? timeout-ms))))

(defn- next-operation-id! [context]
  (swap! (:operation-counter context) inc))

(defn- store-barrier! [context operation-id timeout-ms]
  (let [^Store store (:store context)
        prefix (str "clorch/barrier/" operation-id "/")
        rank-key (str prefix (:rank context))
        store-keys (StringVector.)]
    (.set store rank-key "ready")
    (dotimes [peer (:world-size context)]
      (.push_back store-keys (str prefix peer)))
    (._wait store store-keys (Milliseconds. timeout-ms))))

(defn barrier!
  "Blocks until every rank enters the barrier."
  ([] (barrier! nil {}))
  ([options] (barrier! nil options))
  ([context {:keys [async? timeout-ms] :or {async? false}}]
   (let [context (context-or-current context)
         timeout-ms (or timeout-ms (:timeout-ms context))
         operation-id (next-operation-id! context)]
     (submit-store-work #(store-barrier! context operation-id timeout-ms)
                        nil async?))))

(defn rank-zero-call!
  "Runs f only on rank zero and propagates its success or failure to all ranks."
  ([f] (rank-zero-call! nil f))
  ([context f]
   (let [context (context-or-current context)
         ^Store store (:store context)
         operation-id (next-operation-id! context)
         result-key (str "clorch/rank-zero-result/" operation-id)
         result (when (rank-zero? context)
                  (try
                    (let [value (f)]
                      (.set store result-key "ok")
                      {:value value})
                    (catch Throwable cause
                      (.set store result-key
                            (str "error\n" (.getName (class cause)) "\n"
                                 (.getMessage cause)))
                      {:error cause})))
         status (.get_to_str store result-key)]
     (if (= status "ok")
       (:value result)
       (let [[_ class-name message] (str/split status #"\n" 3)]
         (if-let [cause (:error result)]
           (throw cause)
           (throw (ex-info "Rank-zero operation failed"
                           {:remote-class class-name
                            :remote-message message}))))))))

(defn destroy-process-group!
  "Synchronizes and releases the native communicator. Repeated calls are safe."
  ([]
   (when-let [context @default-context]
     (destroy-process-group! context)))
  ([context]
   (locking default-context
     (when (and context
                (or (identical? context @default-context)
                    (nil? @default-context)))
       (try
         ((requiring-resolve 'clorch.distributed.nccl/destroy!)
          (:backend-state context))
         (finally
           (try (.close ^Store (:store context)) (catch Throwable _))
           (when (identical? context @default-context)
             (reset! default-context nil))))))
   nil))

(defmacro with-process-group
  "Initializes a process group for body and always destroys it."
  [options & body]
  `(let [context# (init-process-group! ~options)]
     (try
       ~@body
       (finally
         (destroy-process-group! context#)))))

(defn launch!
  "Launches local worker JVMs. See clorch.distributed.launch/launch!."
  [config]
  ((requiring-resolve 'clorch.distributed.launch/launch!) config))

(defn job-status [job]
  ((requiring-resolve 'clorch.distributed.launch/status) job))

(defn await-job!
  ([job]
   ((requiring-resolve 'clorch.distributed.launch/await!) job))
  ([job timeout-ms]
   ((requiring-resolve 'clorch.distributed.launch/await!) job timeout-ms)))

(defn stop-job! [job]
  ((requiring-resolve 'clorch.distributed.launch/stop!) job))

(defn job-logs
  ([job worker-rank]
   ((requiring-resolve 'clorch.distributed.launch/logs) job worker-rank))
  ([job worker-rank max-bytes]
   ((requiring-resolve 'clorch.distributed.launch/logs) job worker-rank max-bytes)))

(defn checkpoint!
  "Runs a checkpoint writer on rank zero and propagates failure to every rank."
  ([write-checkpoint!]
   (rank-zero-call! write-checkpoint!))
  ([context write-checkpoint!]
   (rank-zero-call! context write-checkpoint!)))

(defn save-checkpoint!
  "Atomically saves distributed model, optimizer, sampler, and training state."
  ([path values]
   (save-checkpoint! nil path values))
  ([context path values]
   ((requiring-resolve 'clorch.distributed.checkpoint/save!)
    context path values)))

(defn load-checkpoint!
  "Restores a distributed checkpoint on every rank and returns training state."
  ([path targets]
   (load-checkpoint! nil path targets))
  ([context path targets]
   ((requiring-resolve 'clorch.distributed.checkpoint/load!)
    context path targets)))
