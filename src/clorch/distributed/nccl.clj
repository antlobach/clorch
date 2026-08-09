(ns clorch.distributed.nccl
  "Direct NCCL backend for Clorch distributed collectives.

  This namespace is loaded only for :nccl process groups so CPU-only JVMs never
  initialize CUDA classes."
  (:require [clorch.cuda :as cuda])
  (:import [org.bytedeco.cuda.cudart CUevent_st CUstream_st]
           [org.bytedeco.cuda.global cudart nccl]
           [org.bytedeco.cuda.nccl ncclComm ncclUniqueId]
           [org.bytedeco.javacpp BytePointer IntPointer Pointer]
           [org.bytedeco.pytorch ByteVector Store TensorBase TensorVector]))

(defrecord NcclWork [^CUevent_st event ^ncclComm communicator result released?])

(defn- nccl-message [status]
  (if-let [message (nccl/ncclGetErrorString status)]
    (.getString message)
    (str "NCCL status " status)))

(defn- check-nccl! [operation status]
  (when-not (= nccl/ncclSuccess status)
    (throw (ex-info (str operation " failed: " (nccl-message status))
                    {:operation operation :status status})))
  status)

(defn- check-cuda! [operation status]
  (when-not (= cudart/cudaSuccess status)
    (throw (ex-info (str operation " failed with CUDA status " status)
                    {:operation operation :status status})))
  status)

(defn- scalar-type->nccl [^TensorBase tensor]
  (case (.toString (.scalar_type tensor))
    "Byte" nccl/ncclUint8
    "Char" nccl/ncclInt8
    "Int" nccl/ncclInt32
    "Long" nccl/ncclInt64
    "Half" nccl/ncclFloat16
    "Float" nccl/ncclFloat32
    "Double" nccl/ncclFloat64
    "BFloat16" nccl/ncclBfloat16
    "Bool" nccl/ncclUint8
    (throw (ex-info "NCCL does not support this tensor dtype"
                    {:dtype (.toString (.scalar_type tensor))}))))

(defn- validate-tensor! [^TensorBase tensor]
  (when-not (.is_cuda tensor)
    (throw (ex-info "NCCL collectives require CUDA tensors" {})))
  (when-not (.is_contiguous tensor)
    (throw (ex-info "NCCL collectives require contiguous tensors" {})))
  tensor)

(defn- tensors [value]
  (cond
    (instance? TensorBase value) [value]
    (instance? TensorVector value)
    (mapv #(.get ^TensorVector value (long %)) (range (.size ^TensorVector value)))
    (sequential? value) (mapv validate-tensor! value)
    :else (throw (ex-info "Collectives require a tensor or tensor collection"
                          {:value-type (some-> value class str)}))))

(defn- reduction [operation]
  (case operation
    :sum nccl/ncclSum
    :avg nccl/ncclAvg
    :product nccl/ncclProd
    :min nccl/ncclMin
    :max nccl/ncclMax
    (throw (ex-info "Unsupported NCCL reduction operation"
                    {:operation operation}))))

(defn- unique-id-bytes [^ncclUniqueId unique-id]
  (byte-array
   (map #(.internal unique-id (int %))
        (range nccl/NCCL_UNIQUE_ID_BYTES))))

(defn- install-unique-id! [^ncclUniqueId unique-id ^ByteVector byte-values]
  (when-not (= nccl/NCCL_UNIQUE_ID_BYTES (.size byte-values))
    (throw (ex-info "Invalid NCCL rendezvous identifier"
                    {:expected nccl/NCCL_UNIQUE_ID_BYTES
                     :actual (.size byte-values)})))
  (dotimes [index nccl/NCCL_UNIQUE_ID_BYTES]
    (.internal unique-id index (.get byte-values (long index))))
  unique-id)

(defn create!
  "Creates a rank-local NCCL communicator and non-blocking communication stream."
  [^Store store rank world-size local-rank]
  (cuda/set-device! local-rank)
  (let [unique-id (ncclUniqueId.)
        rendezvous-key "clorch/nccl/unique-id"]
    (if (zero? rank)
      (do
        (check-nccl! "ncclGetUniqueId" (nccl/ncclGetUniqueId unique-id))
        (.set store rendezvous-key (ByteVector. (unique-id-bytes unique-id))))
      (install-unique-id! unique-id (.get store rendezvous-key)))
    (let [communicator (ncclComm.)
          stream (CUstream_st.)]
      (try
        (check-nccl! "ncclCommInitRank"
                     (nccl/ncclCommInitRank communicator (int world-size)
                                            unique-id (int rank)))
        (check-cuda! "cudaStreamCreateWithFlags"
                     (cudart/cudaStreamCreateWithFlags
                      stream cudart/cudaStreamNonBlocking))
        {:communicator communicator :stream stream}
        (catch Throwable cause
          (when-not (.isNull communicator)
            (nccl/ncclCommAbort communicator))
          (throw cause))))))

(defn destroy! [{:keys [^ncclComm communicator ^CUstream_st stream]}]
  (when (and stream (not (.isNull stream)))
    (cudart/cudaStreamSynchronize stream))
  (when (and communicator (not (.isNull communicator)))
    (let [finalize-status (nccl/ncclCommFinalize communicator)]
      (when-not (= nccl/ncclSuccess finalize-status)
        (nccl/ncclCommAbort communicator))
      (nccl/ncclCommDestroy communicator)))
  (when (and stream (not (.isNull stream)))
    (cudart/cudaStreamDestroy stream))
  nil)

(defn abort! [{:keys [^ncclComm communicator]}]
  (when (and communicator (not (.isNull communicator)))
    (nccl/ncclCommAbort communicator))
  nil)

(defn- enqueue! [{:keys [^ncclComm communicator ^CUstream_st stream]}
                 result operation]
  (let [ready (CUevent_st.)
        done (CUevent_st.)]
    (try
      (check-cuda! "cudaEventCreateWithFlags"
                   (cudart/cudaEventCreateWithFlags
                    ready cudart/cudaEventDisableTiming))
      (check-cuda! "cudaEventCreateWithFlags"
                   (cudart/cudaEventCreateWithFlags
                    done cudart/cudaEventDisableTiming))
      (check-cuda! "cudaEventRecord" (cudart/cudaEventRecord ready))
      (check-cuda! "cudaStreamWaitEvent"
                   (cudart/cudaStreamWaitEvent stream ready))
      (operation communicator stream)
      (check-cuda! "cudaEventRecord"
                   (cudart/cudaEventRecord done stream))
      (check-cuda! "cudaStreamWaitEvent" (cudart/cudaStreamWaitEvent
                                          (CUstream_st. (cast Pointer nil)) done))
      (cudart/cudaEventDestroy ready)
      (->NcclWork done communicator result (atom false))
      (catch Throwable cause
        (when-not (.isNull ready) (cudart/cudaEventDestroy ready))
        (when-not (.isNull done) (cudart/cudaEventDestroy done))
        (throw cause)))))

(defn await!
  "Waits for NCCL work with a bounded timeout and returns its result."
  [^NcclWork work timeout-ms]
  (when-not work
    (throw (ex-info "Cannot await nil NCCL work" {})))
  (when-not @(:released? work)
    (let [deadline (+ (System/nanoTime) (* (long timeout-ms) 1000000))]
      (loop []
        (let [status (cudart/cudaEventQuery (:event work))]
          (cond
            (= status cudart/cudaSuccess)
            (let [async-status (IntPointer. 1)]
              (check-nccl! "ncclCommGetAsyncError"
                           (nccl/ncclCommGetAsyncError
                            (:communicator work) async-status))
              (check-nccl! "NCCL asynchronous operation" (.get async-status 0))
              (cudart/cudaEventDestroy (:event work))
              (reset! (:released? work) true))

            (= status cudart/cudaErrorNotReady)
            (if (< (System/nanoTime) deadline)
              (do (Thread/sleep 1) (recur))
              (throw (java.util.concurrent.TimeoutException.
                      (str "NCCL operation exceeded " timeout-ms " ms"))))

            :else
            (check-cuda! "cudaEventQuery" status))))))
  (:result work))

(defn all-reduce! [backend tensors-value operation]
  (let [values (mapv validate-tensor! (tensors tensors-value))]
    (enqueue!
     backend tensors-value
     (fn [communicator stream]
       (check-nccl! "ncclGroupStart" (nccl/ncclGroupStart))
       (try
         (doseq [^TensorBase tensor values]
           (check-nccl! "ncclAllReduce"
                        (nccl/ncclAllReduce
                         (.data_ptr tensor) (.data_ptr tensor) (.numel tensor)
                         (scalar-type->nccl tensor) (reduction operation)
                         communicator stream)))
         (finally
           (check-nccl! "ncclGroupEnd" (nccl/ncclGroupEnd))))))))

(defn broadcast! [backend tensors-value root-rank]
  (let [values (mapv validate-tensor! (tensors tensors-value))]
    (enqueue!
     backend tensors-value
     (fn [communicator stream]
       (check-nccl! "ncclGroupStart" (nccl/ncclGroupStart))
       (try
         (doseq [^TensorBase tensor values]
           (check-nccl! "ncclBroadcast"
                        (nccl/ncclBroadcast
                         (.data_ptr tensor) (.data_ptr tensor) (.numel tensor)
                         (scalar-type->nccl tensor) (int root-rank)
                         communicator stream)))
         (finally
           (check-nccl! "ncclGroupEnd" (nccl/ncclGroupEnd))))))))

(defn reduce! [backend tensors-value operation root-rank]
  (let [values (mapv validate-tensor! (tensors tensors-value))]
    (enqueue!
     backend tensors-value
     (fn [communicator stream]
       (check-nccl! "ncclGroupStart" (nccl/ncclGroupStart))
       (try
         (doseq [^TensorBase tensor values]
           (check-nccl! "ncclReduce"
                        (nccl/ncclReduce
                         (.data_ptr tensor) (.data_ptr tensor) (.numel tensor)
                         (scalar-type->nccl tensor) (reduction operation)
                         (int root-rank) communicator stream)))
         (finally
           (check-nccl! "ncclGroupEnd" (nccl/ncclGroupEnd))))))))

(defn all-gather-into! [backend ^TensorBase output ^TensorBase input world-size]
  (validate-tensor! output)
  (validate-tensor! input)
  (when-not (= (.numel output) (* world-size (.numel input)))
    (throw (ex-info "All-gather output size must equal input size times world size"
                    {:output-elements (.numel output)
                     :input-elements (.numel input)
                     :world-size world-size})))
  (when-not (= (.toString (.scalar_type output))
               (.toString (.scalar_type input)))
    (throw (ex-info "All-gather input and output dtypes must match" {})))
  (enqueue!
   backend output
   (fn [communicator stream]
     (check-nccl! "ncclAllGather"
                  (nccl/ncclAllGather
                   (.data_ptr input) (.data_ptr output) (.numel input)
                   (scalar-type->nccl input) communicator stream)))))

(defn reduce-scatter-into!
  [backend ^TensorBase output ^TensorBase input world-size operation]
  (validate-tensor! output)
  (validate-tensor! input)
  (when-not (= (.numel input) (* world-size (.numel output)))
    (throw (ex-info "Reduce-scatter input size must equal output size times world size"
                    {:output-elements (.numel output)
                     :input-elements (.numel input)
                     :world-size world-size})))
  (when-not (= (.toString (.scalar_type output))
               (.toString (.scalar_type input)))
    (throw (ex-info "Reduce-scatter input and output dtypes must match" {})))
  (enqueue!
   backend output
   (fn [communicator stream]
     (check-nccl! "ncclReduceScatter"
                  (nccl/ncclReduceScatter
                   (.data_ptr input) (.data_ptr output) (.numel output)
                   (scalar-type->nccl input) (reduction operation)
                   communicator stream)))))

(defn- byte-offset-pointer [^TensorBase tensor element-offset]
  (doto (BytePointer. (.data_ptr tensor))
    (.position (* (long element-offset) (.element_size tensor)))))

(defn all-to-all-single!
  [backend ^TensorBase output ^TensorBase input world-size output-splits input-splits]
  (validate-tensor! output)
  (validate-tensor! input)
  (when-not (= (.toString (.scalar_type output))
               (.toString (.scalar_type input)))
    (throw (ex-info "All-to-all input and output dtypes must match" {})))
  (let [default-input-count (quot (.numel input) world-size)
        default-output-count (quot (.numel output) world-size)
        input-counts (if (seq input-splits) input-splits
                         (repeat world-size default-input-count))
        output-counts (if (seq output-splits) output-splits
                          (repeat world-size default-output-count))
        input-counts (mapv long input-counts)
        output-counts (mapv long output-counts)]
    (when-not (and (= world-size (count input-counts))
                   (= world-size (count output-counts))
                   (= (.numel input) (reduce + input-counts))
                   (= (.numel output) (reduce + output-counts)))
      (throw (ex-info "All-to-all split sizes must partition input and output"
                      {:world-size world-size
                       :input-splits input-counts
                       :output-splits output-counts})))
    (enqueue!
     backend output
     (fn [communicator stream]
       (check-nccl! "ncclGroupStart" (nccl/ncclGroupStart))
       (try
         (loop [peer 0 input-offset 0 output-offset 0]
           (when (< peer world-size)
             (check-nccl! "ncclSend"
                          (nccl/ncclSend
                           (byte-offset-pointer input input-offset)
                           (input-counts peer) (scalar-type->nccl input) peer
                           communicator stream))
             (check-nccl! "ncclRecv"
                          (nccl/ncclRecv
                           (byte-offset-pointer output output-offset)
                           (output-counts peer) (scalar-type->nccl output) peer
                           communicator stream))
             (recur (inc peer)
                    (+ input-offset (input-counts peer))
                    (+ output-offset (output-counts peer)))))
         (finally
           (check-nccl! "ncclGroupEnd" (nccl/ncclGroupEnd))))))))

(defn send! [backend tensors-value destination tag]
  (when-not (zero? tag)
    (throw (ex-info "NCCL point-to-point operations do not support tags"
                    {:tag tag})))
  (let [values (mapv validate-tensor! (tensors tensors-value))]
    (enqueue!
     backend tensors-value
     (fn [communicator stream]
       (check-nccl! "ncclGroupStart" (nccl/ncclGroupStart))
       (try
         (doseq [^TensorBase tensor values]
           (check-nccl! "ncclSend"
                        (nccl/ncclSend
                         (.data_ptr tensor) (.numel tensor)
                         (scalar-type->nccl tensor) (int destination)
                         communicator stream)))
         (finally
           (check-nccl! "ncclGroupEnd" (nccl/ncclGroupEnd))))))))

(defn receive! [backend tensors-value source tag]
  (when-not (zero? tag)
    (throw (ex-info "NCCL point-to-point operations do not support tags"
                    {:tag tag})))
  (let [values (mapv validate-tensor! (tensors tensors-value))]
    (enqueue!
     backend tensors-value
     (fn [communicator stream]
       (check-nccl! "ncclGroupStart" (nccl/ncclGroupStart))
       (try
         (doseq [^TensorBase tensor values]
           (check-nccl! "ncclRecv"
                        (nccl/ncclRecv
                         (.data_ptr tensor) (.numel tensor)
                         (scalar-type->nccl tensor) (int source)
                         communicator stream)))
         (finally
           (check-nccl! "ncclGroupEnd" (nccl/ncclGroupEnd))))))))
