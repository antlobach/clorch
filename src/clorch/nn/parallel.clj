(ns clorch.nn.parallel
  "Distributed neural-network wrappers backed by direct NCCL collectives."
  (:require [clorch.distributed :as dist]
            [clorch.nn :as nn])
  (:import [org.bytedeco.pytorch Module Scalar SizeTVector Tensor
            TensorBase TensorTensorHook TensorVector]
           [org.bytedeco.pytorch.global torch]))

(def ^:dynamic *synchronize-gradients*
  "False inside no-sync so DDP accumulates local gradients without collectives."
  true)

(defn- parameter-vector [model]
  (let [parameters (nn/parameters model :trainable-only true)]
    (when (zero? (.size parameters))
      (throw (ex-info "DistributedDataParallel requires trainable parameters" {})))
    parameters))

(defn- tensor-vector->vec [^TensorVector values]
  (mapv #(Tensor. (.get values (long %))) (range (.size values))))

(defn- native-buffers [model]
  (let [output (TensorVector.)]
    (letfn [(walk [node]
              (cond
                (instance? Module node)
                (let [buffers (.buffers ^Module node)]
                  (dotimes [index (.size buffers)]
                    (.push_back output (.get buffers (long index)))))

                (record? node) (run! walk (vals node))
                (map? node) (run! walk (vals node))
                (sequential? node) (run! walk node)))]
      (walk model)
      output)))

(defn- validate-parameters! [context parameters]
  (doseq [[index ^Tensor parameter] (map-indexed vector parameters)]
    (when-not (.is_cuda parameter)
      (throw (ex-info "DDP parameters must be on the rank-local CUDA device"
                      {:parameter-index index
                       :local-rank (:local-rank context)})))
    (when-not (= (:local-rank context) (long (.index (.device parameter))))
      (throw (ex-info "DDP parameter is on the wrong CUDA device"
                      {:parameter-index index
                       :parameter-device (long (.index (.device parameter)))
                       :local-rank (:local-rank context)}))))
  parameters)

(defn- parameter-signature [parameters]
  (mapv (fn [^Tensor parameter]
          {:shape (let [sizes (.sizes parameter)]
                    (mapv #(.get sizes (long %)) (range (.size sizes))))
           :dtype (.toString (.scalar_type parameter))
           :device (long (.index (.device parameter)))})
        parameters))

(defn- verify-parameters! [context parameters]
  (let [operation-id (swap! (:operation-counter context) inc)
        prefix (str "clorch/ddp/signature/" operation-id "/")
        rank-key (str prefix (:rank context))
        signature (pr-str (parameter-signature parameters))
        store (:store context)
        store-keys (org.bytedeco.pytorch.StringVector.)]
    (.set store rank-key signature)
    (dotimes [peer (:world-size context)]
      (.push_back store-keys (str prefix peer)))
    (._wait store store-keys
            (org.bytedeco.javacpp.chrono.Milliseconds. (:timeout-ms context)))
    (let [signatures
          (mapv #(.get_to_str store (str prefix %))
                (range (:world-size context)))]
      (when-not (apply = signatures)
        (throw (ex-info "DDP parameter shapes, dtypes, and devices differ across ranks"
                        {:rank (:rank context)}))))))

(defn- bucket-indices [^TensorVector parameters bucket-cap-bytes]
  (let [limits (SizeTVector.
                (long-array [(long (torch/kDefaultFirstBucketBytes))
                             (long bucket-cap-bytes)]))
        native-buckets (.get0
                        (torch/compute_bucket_assignment_by_size
                         parameters limits))]
    (mapv (fn [bucket-index]
            (let [indices (.get native-buckets (long bucket-index))]
              (mapv #(.get indices (long %)) (range (.size indices)))))
          (range (.size native-buckets)))))

(defn- make-bucket [parameters indices]
  (let [native-parameters (TensorVector.)]
    (doseq [index indices]
      (.push_back native-parameters (parameters index)))
    (let [buffer (torch/flatten_dense_tensors native-parameters)
          views (torch/unflatten_dense_tensors buffer native-parameters)]
      {:indices indices
       :parameters native-parameters
       :buffer buffer
       :views views
       :remaining (atom (count indices))
       :gradients (atom (vec (repeat (count indices) nil)))
       :lock (Object.)})))

(defn- await-pending! [context pending]
  (let [works (locking pending
                (let [works @pending]
                  (reset! pending [])
                  works))]
    (doseq [work works]
      (dist/await! work (:timeout-ms context)))))

(defn- reset-prior-gradient! [^Tensor parameter ^Tensor gradient]
  (let [prior (.grad parameter)]
    (when (.defined prior)
      (.add_ gradient prior)
      (.reset (.mutable_grad parameter)))))

(defn- complete-bucket!
  [context bucket pending]
  (let [buffer (:buffer bucket)
        work (dist/all-reduce! context buffer {:op :sum :async? true})]
    (.div_ ^Tensor buffer (Scalar. (double (:world-size context))))
    (doseq [slot (range (count (:indices bucket)))]
      (.copy_ ^Tensor (@(:gradients bucket) slot)
              (.get ^TensorVector (:views bucket) (long slot))))
    (swap! pending conj work)
    (reset! (:remaining bucket) (count (:indices bucket)))
    (reset! (:gradients bucket) (vec (repeat (count (:indices bucket)) nil)))))

(defn- gradient-hook
  [context bucket slot ^Tensor parameter synchronize? pending classloader]
  (proxy [TensorTensorHook] []
    (call [gradient-base]
      ;; Native autograd workers do not inherit the application classloader.
      (.setContextClassLoader (Thread/currentThread) classloader)
      (when @synchronize?
        (let [gradient (Tensor. ^TensorBase gradient-base)
              bucket-lock (:lock bucket)]
          (reset-prior-gradient! parameter gradient)
          (locking bucket-lock
            (.copy_ ^Tensor (.get ^TensorVector (:views bucket) (long slot))
                    gradient)
            (swap! (:gradients bucket) assoc slot gradient)
            (when (zero? (swap! (:remaining bucket) dec))
              (complete-bucket! context bucket pending)))))
      gradient-base)))

(defn- install-gradient-hooks!
  [context parameters buckets synchronize? pending]
  (let [classloader (.getContextClassLoader (Thread/currentThread))]
    (vec
     (mapcat
      (fn [bucket]
        (mapv
         (fn [slot parameter-index]
           (let [parameter (parameters parameter-index)
                 hook (gradient-hook context bucket slot parameter
                                     synchronize? pending classloader)
                 hook-id (.register_hook ^Tensor parameter hook)]
             {:parameter parameter :hook hook :hook-id hook-id}))
         (range (count (:indices bucket)))
         (:indices bucket)))
      buckets))))

(deftype DistributedDataParallel
         [model context parameters buckets hooks buffers broadcast-buffers?
          synchronize? pending closed?]
  nn/IModule
  (-forward [_ input]
    (when @closed?
      (throw (ex-info "DistributedDataParallel has been closed" {})))
    (await-pending! context pending)
    (doseq [bucket buckets]
      (when-not (= (count (:indices bucket)) @(:remaining bucket))
        (throw (ex-info "Previous backward pass did not produce every parameter gradient"
                        {:missing-gradient-count @(:remaining bucket)
                         :bucket-indices (:indices bucket)}))))
    (when (and broadcast-buffers? (pos? (.size ^TensorVector buffers)))
      (dist/broadcast! context buffers {:root-rank 0}))
    (reset! synchronize? *synchronize-gradients*)
    (nn/forward model input))
  (-train [this training?]
    (nn/train model training?)
    this)
  (-to [_ _]
    (throw (ex-info "Move a model to its rank-local device before wrapping it in DistributedDataParallel"
                    {})))
  clojure.lang.IFn
  (invoke [this input]
    (nn/forward this input))
  clojure.lang.ILookup
  (valAt [this lookup-key]
    (.valAt this lookup-key nil))
  (valAt [_ lookup-key not-found]
    (case lookup-key
      :model model
      :context context
      :parameters parameters
      :buckets buckets
      :buffers buffers
      :broadcast-buffers? broadcast-buffers?
      :closed? @closed?
      not-found))
  java.lang.AutoCloseable
  (close [_]
    (when (compare-and-set! closed? false true)
      (await-pending! context pending)
      (doseq [{:keys [^Tensor parameter hook-id]} hooks]
        (.remove_hook parameter hook-id)))))

(defn distributed-data-parallel
  "Wraps a rank-local model in synchronous NCCL data parallelism.

  Parameters are verified and broadcast from rank zero. Reusable gradient
  buckets launch asynchronous NCCL all-reduce operations as backward produces
  each bucket. Construct and move the model to its rank-local CUDA device first.

  Options:
  - :process-group       ProcessGroupContext; defaults to current group
  - :bucket-cap-mb       gradient bucket capacity, default 25 MiB
  - :broadcast-buffers?  synchronize module buffers before forward, default true"
  [model & args]
  (let [options (if (map? (first args)) (first args) (apply hash-map args))
        context (or (:process-group options) (dist/current-process-group))
        bucket-cap-mb (double (get options :bucket-cap-mb 25.0))
        broadcast-buffers? (get options :broadcast-buffers? true)]
    (when-not (pos? bucket-cap-mb)
      (throw (ex-info "DDP bucket capacity must be positive"
                      {:bucket-cap-mb bucket-cap-mb})))
    (when (get options :find-unused-parameters? false)
      (throw (ex-info "Direct NCCL DDP requires every trainable parameter in each synchronized backward pass"
                      {:find-unused-parameters? true})))
    (when (get options :gradient-as-bucket-view? false)
      (throw (ex-info "Direct NCCL DDP does not expose gradients as bucket views"
                      {:gradient-as-bucket-view? true})))
    (let [native-parameters (parameter-vector model)
          parameters (validate-parameters!
                      context (tensor-vector->vec native-parameters))
          _ (verify-parameters! context parameters)
          _ (dist/broadcast! context native-parameters {:root-rank 0})
          buckets (mapv #(make-bucket parameters %)
                        (bucket-indices native-parameters
                                        (long (* bucket-cap-mb 1024.0 1024.0))))
          synchronize? (atom true)
          pending (atom [])
          hooks (install-gradient-hooks! context parameters buckets
                                         synchronize? pending)]
      (DistributedDataParallel.
       model context parameters buckets hooks (native-buffers model)
       broadcast-buffers? synchronize? pending (atom false)))))

(defmacro no-sync
  "Accumulates local gradients in body without NCCL synchronization."
  [& body]
  `(binding [*synchronize-gradients* false]
     ~@body))

(defn synchronize!
  "Waits for all gradient buckets launched by the most recent backward pass."
  [ddp]
  (await-pending! (:context ddp) (.-pending ^DistributedDataParallel ddp))
  ddp)

(defn optimizer-step!
  "Waits for DDP reductions, then steps an optimizer directly or through AMP."
  ([ddp optimizer]
   (optimizer-step! ddp optimizer {}))
  ([ddp optimizer {:keys [scaler]}]
   (synchronize! ddp)
   (if scaler
     ((requiring-resolve 'clorch.amp/step!) scaler optimizer)
     (do
       (.step optimizer)
       true))))

(defn reducer-statistics
  "Returns bucket element counts for capacity and profiling diagnostics."
  [ddp]
  (mapv #(long (.numel ^Tensor (:buffer %))) (:buckets ddp)))
