(ns clorch.data
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io])
  (:import [java.io BufferedReader BufferedWriter InputStreamReader OutputStreamWriter]
           [java.util SplittableRandom]
           [java.util.concurrent Callable Future LinkedBlockingQueue ThreadFactory ThreadPoolExecutor TimeUnit TimeoutException]
           [java.util.concurrent.atomic AtomicInteger]))

(def ^:private torch-api
  (delay
    {:stack (requiring-resolve 'clorch.torch/stack)
     :->tensor (requiring-resolve 'clorch.torch/->tensor)
     :select (requiring-resolve 'clorch.torch/select)
     :randperm (requiring-resolve 'clorch.torch/randperm)
     :tseq (requiring-resolve 'clorch.torch/tseq)
     :item-float (requiring-resolve 'clorch.torch/item-float)}))

(defn- torch-fn [k]
  (get @torch-api k))

(def ^:private tensor-class
  (delay (Class/forName "org.bytedeco.pytorch.Tensor")))

(defprotocol IDataset
  (get-size [this] "Returns the total number of items in the dataset.")
  (get-item [this idx] "Returns the item at the given index as a map {:data tensor :target tensor}."))

(defprotocol IProcessDataset
  (process-spec [this] "Returns EDN-serializable worker spec, e.g. {:factory 'my.ns/mk-ds :args [...]}"))

(defprotocol ISampler
  (sample-indices [this] "Returns this rank's indices for the current epoch.")
  (set-epoch! [this epoch] "Changes the deterministic shuffle epoch.")
  (sampler-state [this] "Returns checkpointable sampler state.")
  (load-sampler-state! [this state] "Restores checkpointable sampler state."))
(def ^:private javadataset-extension
  (delay
    (let [cls (Class/forName "org.bytedeco.pytorch.JavaDataset")]
      (extend cls
        IDataset
        {:get-size (fn [this] (.get (.size this)))
         :get-item (fn [this idx]
                     (let [example (.get this (clojure.core/long idx))]
                       {:data (.data example) :target (.target example)}))}))))

(defn- ensure-javadataset-extension! []
  @javadataset-extension)

(defn dataset
  "Creates a custom dataset from functions.
   Example: (dataset :size #(count my-paths) :get-item #(load-item %))

   Optional:
   - :process-spec {:factory 'your.ns/make-dataset :args [...]}
     Enables process workers in dataloader."
  [& {:keys [size process-spec-map] :as opts}]
  (let [item-fn (:get-item opts)
        process-spec-map (or process-spec-map (:process-spec opts))]
    (if process-spec-map
      (reify IDataset
        (get-size [_] (size))
        (get-item [_ idx] (item-fn idx))
        IProcessDataset
        (process-spec [_] process-spec-map))
      (reify IDataset
        (get-size [_] (size))
        (get-item [_ idx] (item-fn idx))))))

(defmacro defdataset [dataset-name args init-bindings & specs]
  (let [field-syms (take-nth 2 init-bindings)
        size-fn (first (filter #(= (first %) 'get-size) specs))
        item-fn (first (filter #(= (first %) 'get-item) specs))
        record-name (symbol (str dataset-name "Record"))]
    `(do
       (defrecord ~record-name [~@field-syms]
         IDataset
         (get-size [this#]
           (let [{:keys [~@field-syms]} this#] ~@(rest size-fn)))
         (get-item [this# ~(first (second item-fn))]
           (let [{:keys [~@field-syms]} this#] ~@(nnext item-fn))))
       (defn ~dataset-name ~args
         (let [~@init-bindings] (~(symbol (str "->" record-name)) ~@field-syms))))))

(defn default-collate [items]
  (let [first-item (first items)]
    (cond
      (map? first-item)
      (let [ks (clojure.core/keys first-item)]
        (into {} (for [k ks] [k (default-collate (map #(get % k) items))])))
      (instance? @tensor-class first-item)
      (let [v (org.bytedeco.pytorch.TensorVector.)]
        (doseq [item items] (.push_back v item))
        (let [stacked ((torch-fn :stack) v 0)] (.retainReference stacked) stacked))
      :else (vec items))))

(defn- build-batch [ds idxs collate-fn]
  (let [items (map #(get-item ds %) idxs)] (collate-fn items)))

(defn- build-chunks [idxs batch-size drop-last?]
  (if drop-last?
    (partition batch-size idxs)
    (partition-all batch-size idxs)))

(defn- submit-batch [executor ds idxs collate-fn]
  (.submit executor
           ^Callable
           (reify Callable
             (call [_] (build-batch ds idxs collate-fn)))))

(defn- make-thread-worker-executor [num-workers]
  (let [thread-id (AtomicInteger. 0)
        thread-factory (reify ThreadFactory
                         (newThread [_ runnable]
                           (doto (Thread. runnable)
                             (.setName (str "clorch-data-thread-worker-" (.incrementAndGet thread-id)))
                             (.setDaemon true))))
        executor (ThreadPoolExecutor. (clojure.core/int num-workers)
                                      (clojure.core/int num-workers)
                                      (long 1)
                                      TimeUnit/SECONDS
                                      (LinkedBlockingQueue.)
                                      thread-factory)]
    (.allowCoreThreadTimeOut executor true)
    executor))

(defn- parallel-batches
  [ds chunks collate-fn num-workers prefetch-factor]
  (let [executor (make-thread-worker-executor num-workers)
        max-inflight (clojure.core/max 1 (* (clojure.core/int num-workers) (clojure.core/int prefetch-factor)))]
    (letfn [(fill-queue [remaining q]
              (loop [remaining remaining
                     q q]
                (if (and (seq remaining) (< (count q) max-inflight))
                  (recur (rest remaining)
                         (conj q (submit-batch executor ds (first remaining) collate-fn)))
                  [remaining q])))
            (step [remaining q]
              (lazy-seq
               (let [[remaining q] (fill-queue remaining q)]
                 (if (empty? q)
                   (do (.shutdown executor) nil)
                   (let [^Future f (first q)
                         batch (.get f)]
                     (cons batch (step remaining (subvec q 1))))))))]
      (step chunks []))))

(defn- write-edn-tempfile [prefix value]
  (let [f (java.io.File/createTempFile prefix ".edn")]
    (spit f (pr-str value))
    (.getAbsolutePath f)))

(defn- process-start-worker [spec-file worker-id]
  (let [proc-builder (ProcessBuilder. ["clojure" "-M" "-m" "clorch.data-worker" "--server" spec-file])
        _ (.redirectError proc-builder java.lang.ProcessBuilder$Redirect/INHERIT)
        proc (.start proc-builder)
        rdr (BufferedReader. (InputStreamReader. (.getInputStream proc)))
        w (BufferedWriter. (OutputStreamWriter. (.getOutputStream proc)))
        responses (LinkedBlockingQueue.)]
    (doto
     (Thread.
      (fn []
        (try
          (loop []
            (when-let [line (.readLine rdr)]
              (.put responses (edn/read-string line))
              (recur)))
          (.put responses {:ok false
                           :worker-id worker-id
                           :error "Worker output stream closed unexpectedly"})
          (catch Throwable t
            (.put responses {:ok false
                             :worker-id worker-id
                             :error (.getMessage t)})))))
      (.setName (str "clorch-data-worker-" worker-id "-reader"))
      (.setDaemon true)
      (.start))
    {:id worker-id
     :process proc
     :reader rdr
     :writer w
     :responses responses
     :request-id 0}))

(defn- process-stop-worker! [{:keys [process writer] :as worker}]
  (try
    (when writer
      (.write ^BufferedWriter writer (str (pr-str {:id -1 :op :shutdown}) "\n"))
      (.flush ^BufferedWriter writer))
    (catch Throwable _))
  (try
    (when process
      (.waitFor ^Process process 200 TimeUnit/MILLISECONDS)
      (when (.isAlive ^Process process)
        (.destroy ^Process process))
      (when (.isAlive ^Process process)
        (.destroyForcibly ^Process process)))
    (catch Throwable _))
  worker)

(defn- process-send! [worker idxs]
  (let [request-id (inc (:request-id worker))]
    (.write ^BufferedWriter (:writer worker)
            (str (pr-str {:id request-id :idxs idxs}) "\n"))
    (.flush ^BufferedWriter (:writer worker))
    (assoc worker :request-id request-id)))

(defn- process-await! [{:keys [responses id] :as worker} timeout-ms]
  (let [resp (if timeout-ms
               (.poll ^LinkedBlockingQueue responses timeout-ms TimeUnit/MILLISECONDS)
               (.take ^LinkedBlockingQueue responses))]
    (when (nil? resp)
      (throw (TimeoutException.
              (str "DataLoader worker " id " timed out after " timeout-ms "ms"))))
    (when-not (:ok resp)
      (throw (ex-info "Process dataloader worker failed" {:worker id :resp resp})))
    [worker (:items resp)]))

(defn- process-parallel-batches
  [ds chunks collate-fn num-workers prefetch-factor timeout-ms worker-init-fn]
  (when-not (satisfies? IProcessDataset ds)
    (throw (IllegalArgumentException.
            "Process workers require dataset to implement IProcessDataset via :process-spec.")))
  (let [spec (cond-> (process-spec ds)
               worker-init-fn (assoc :worker-init-fn worker-init-fn))
        spec-file (write-edn-tempfile "clorch-ds-spec" spec)
        workers (mapv #(process-start-worker spec-file %) (range (max 1 num-workers)))
        prefetch-factor (max 1 (clojure.core/int prefetch-factor))
        capacity (max 1 (* (clojure.core/int num-workers) prefetch-factor))
        cleaned? (atom false)]
    (letfn [(cleanup! [ws]
              (when (compare-and-set! cleaned? false true)
                (doseq [w ws] (process-stop-worker! w))
                (io/delete-file spec-file true)))
            (fill-inflight [remaining ws inflight next-worker]
              (loop [remaining (vec remaining)
                     ws ws
                     inflight (vec inflight)
                     next-worker next-worker]
                (if (or (empty? remaining) (>= (count inflight) capacity))
                  [remaining ws inflight next-worker]
                  (let [worker-idx (mod next-worker (count ws))
                        worker (nth ws worker-idx)
                        idxs (first remaining)
                        worker' (process-send! worker idxs)
                        ws' (assoc ws worker-idx worker')]
                    (recur (subvec remaining 1)
                           ws'
                           (conj inflight {:worker-id (:id worker') :idxs idxs})
                           (inc next-worker))))))
            (drain-one [ws inflight]
              (let [slot (first inflight)
                    worker-id (:worker-id slot)
                    worker-idx (first (keep-indexed (fn [i w] (when (= (:id w) worker-id) i)) ws))
                    worker (nth ws worker-idx)
                    [worker' items] (process-await! worker timeout-ms)
                    ws' (assoc ws worker-idx worker')]
                [ws' (subvec inflight 1) (collate-fn items)]))
            (step [remaining ws inflight next-worker]
              (lazy-seq
               (if (and (empty? remaining) (empty? inflight))
                 (do
                   (cleanup! ws)
                   nil)
                 (try
                   (let [[remaining ws inflight next-worker] (fill-inflight remaining ws inflight next-worker)
                         [ws inflight batch] (drain-one ws inflight)]
                     (cons batch (step remaining ws inflight next-worker)))
                   (catch Throwable t
                     (cleanup! ws)
                     (throw t))))))]
      (step (vec chunks) workers [] 0))))

(defn cleanup-data! [input]
  (cond
    (instance? java.lang.AutoCloseable input) (.close ^java.lang.AutoCloseable input)
    (coll? input) (doseq [i input] (cleanup-data! i))))

(defn tensor-dataset [X y]
  (let [->tensor (torch-fn :->tensor)
        select (torch-fn :select)
        X-t (->tensor X)
        y-t (->tensor y)
        n (.size X-t (clojure.core/long 0))]
    (dataset :size (fn [] n)
             :get-item (fn [idx] {:data (select X-t 0 idx)
                                  :target (select y-t 0 idx)}))))

(defn- ceil-div [dividend divisor]
  (quot (+ dividend (dec divisor)) divisor))

(defn- shuffled-range [n seed]
  (let [values (long-array (range n))
        random (SplittableRandom. (long seed))]
    (loop [i (dec n)]
      (when (pos? i)
        (let [j (.nextInt random (inc i))
              tmp (aget values i)]
          (aset-long values i (aget values j))
          (aset-long values j tmp)
          (recur (dec i)))))
    (vec values)))

(defn- sampler-size [dataset-size replicas drop-last?]
  (if drop-last?
    (if (zero? (mod dataset-size replicas))
      (quot dataset-size replicas)
      (max 0 (ceil-div (- dataset-size replicas) replicas)))
    (ceil-div dataset-size replicas)))

(deftype DistributedSampler
         [dataset-size replicas rank shuffle? seed drop-last? epoch]
  ISampler
  (sample-indices [_]
    (let [base (if shuffle?
                 (shuffled-range dataset-size (+ seed @epoch))
                 (vec (range dataset-size)))
          samples-per-rank (sampler-size dataset-size replicas drop-last?)
          total-size (* samples-per-rank replicas)
          indices (cond
                    (zero? total-size) []
                    drop-last? (subvec base 0 total-size)
                    (= total-size dataset-size) base
                    :else (into base (take (- total-size dataset-size)
                                           (cycle base))))]
      (->> indices
           (drop rank)
           (take-nth replicas)
           (take samples-per-rank)
           vec)))
  (set-epoch! [this next-epoch]
    (when (neg? next-epoch)
      (throw (ex-info "Sampler epoch must be non-negative"
                      {:epoch next-epoch})))
    (reset! epoch (long next-epoch))
    this)
  (sampler-state [_]
    {:epoch @epoch
     :seed seed
     :rank rank
     :replicas replicas
     :dataset-size dataset-size
     :drop-last? drop-last?
     :shuffle? shuffle?})
  (load-sampler-state! [this state]
    (doseq [[field expected] [[:seed seed]
                              [:rank rank]
                              [:replicas replicas]
                              [:dataset-size dataset-size]
                              [:drop-last? drop-last?]
                              [:shuffle? shuffle?]]]
      (when-not (= expected (get state field))
        (throw (ex-info "Sampler checkpoint is incompatible"
                        {:field field
                         :expected expected
                         :actual (get state field)}))))
    (set-epoch! this (:epoch state)))
  clojure.lang.Seqable
  (seq [this]
    (seq (sample-indices this)))
  clojure.lang.ILookup
  (valAt [this lookup-key]
    (.valAt this lookup-key nil))
  (valAt [_ lookup-key not-found]
    (case lookup-key
      :dataset-size dataset-size
      :num-replicas replicas
      :rank rank
      :shuffle? shuffle?
      :seed seed
      :drop-last? drop-last?
      :epoch @epoch
      not-found)))

(defn distributed-sampler
  "Partitions a dataset deterministically across distributed ranks.

  Defaults :num-replicas and :rank from WORLD_SIZE and RANK. Call set-epoch!
  before each epoch so every rank applies the same new shuffle."
  [dataset-source & args]
  (let [options (if (map? (first args)) (first args) (apply hash-map args))
        dataset-size (if (number? dataset-source) (long dataset-source) (long (get-size dataset-source)))
        replicas (long (or (:num-replicas options)
                           (some-> (System/getenv "WORLD_SIZE") Long/parseLong)
                           1))
        rank (long (or (:rank options)
                       (some-> (System/getenv "RANK") Long/parseLong)
                       0))
        shuffle? (get options :shuffle? true)
        seed (long (get options :seed 0))
        drop-last? (get options :drop-last? false)]
    (when (neg? dataset-size)
      (throw (ex-info "Dataset size must be non-negative"
                      {:dataset-size dataset-size})))
    (when-not (pos? replicas)
      (throw (ex-info "Sampler replica count must be positive"
                      {:num-replicas replicas})))
    (when-not (<= 0 rank (dec replicas))
      (throw (ex-info "Sampler rank is outside replica range"
                      {:rank rank :num-replicas replicas})))
    (DistributedSampler. dataset-size replicas rank shuffle? seed drop-last? (atom 0))))

(defn- resolve-worker-backend [num-workers worker-backend]
  (if (= worker-backend :auto)
    (if (pos? num-workers) :process :thread)
    worker-backend))

(deftype ClorchDataloader [ds batch-size shuffle? sampler idxs drop-last? num-workers prefetch-factor collate-fn worker-backend timeout-ms worker-init-fn]
  clojure.lang.Seqable
  (seq [_]
    (let [epoch-indices (if sampler (sample-indices sampler) idxs)
          chunks (build-chunks epoch-indices batch-size drop-last?)]
      (cond
        (and (pos? num-workers) (= worker-backend :process))
        (seq (process-parallel-batches ds chunks collate-fn num-workers prefetch-factor timeout-ms worker-init-fn))

        (pos? num-workers)
        (parallel-batches ds chunks collate-fn num-workers prefetch-factor)

        :else
        (map #(build-batch ds % collate-fn) chunks))))
  java.lang.Iterable
  (iterator [this]
    (clojure.lang.RT/iter (seq this)))
  clojure.lang.ILookup
  (valAt [this k] (.valAt this k nil))
  (valAt [_ k not-found]
    (case k
      :dataset ds
      :batch-size batch-size
      :shuffle? shuffle?
      :sampler sampler
      :indices (if sampler (sample-indices sampler) idxs)
      :drop-last? drop-last?
      :num-workers num-workers
      :prefetch-factor prefetch-factor
      :collate-fn collate-fn
      :worker-backend worker-backend
      :timeout-ms timeout-ms
      :worker-init-fn worker-init-fn
      not-found)))

(defn dataloader [ds & args]
  (let [opts (if (map? (first args)) (first args) (apply hash-map args))
        {:keys [batch-size shuffle? sampler drop-last? num-workers prefetch-factor collate-fn worker-backend timeout-ms worker-init-fn]
         :or {batch-size 32
              shuffle? true
              drop-last? false
              num-workers 0
              prefetch-factor 2
              collate-fn default-collate
              worker-backend :auto
              timeout-ms nil
              worker-init-fn nil}} opts
        _ (when (and sampler (contains? opts :shuffle?) shuffle?)
            (throw (ex-info "A dataloader with a sampler cannot also shuffle"
                            {:shuffle? shuffle?})))
        _ (when (= "org.bytedeco.pytorch.JavaDataset" (.getName (class ds)))
            (ensure-javadataset-extension!))
        worker-backend (resolve-worker-backend num-workers worker-backend)
        n (get-size ds)
        idxs (when-not sampler
               (if shuffle?
                 (let [randperm (torch-fn :randperm)
                       item-float (torch-fn :item-float)
                       tseq (torch-fn :tseq)
                       perm (randperm n)]
                   (mapv #(long (item-float %)) (tseq perm)))
                 (vec (clojure.core/range n))))]
    (->ClorchDataloader ds batch-size (and shuffle? (nil? sampler)) sampler idxs
                        drop-last? num-workers prefetch-factor collate-fn
                        worker-backend timeout-ms worker-init-fn)))
