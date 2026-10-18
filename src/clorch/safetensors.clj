(ns clorch.safetensors
  "SafeTensors state-dictionary I/O without Python or pickle execution."
  (:refer-clojure :exclude [load-file])
  (:require [clojure.data.json :as json]
            [clorch.nn :as nn]
            [clorch.torch :as t])
  (:import [java.io RandomAccessFile]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.channels FileChannel]
           [java.nio.charset StandardCharsets]
           [java.nio.file AtomicMoveNotSupportedException Files Path StandardCopyOption StandardOpenOption]
           [org.bytedeco.javacpp BytePointer Pointer]
           [org.bytedeco.pytorch ScalarTypeOptional Tensor TensorOptions]
           [org.bytedeco.pytorch.global torch torch$ScalarType]))

(set! *warn-on-reflection* true)

(def ^:private dtype-info
  {"BOOL" {:bytes 1 :native "Bool"}
   "U8" {:bytes 1 :native "Byte"}
   "I8" {:bytes 1 :native "Char"}
   "I16" {:bytes 2 :native "Short"}
   "I32" {:bytes 4 :native "Int"}
   "I64" {:bytes 8 :native "Long"}
   "F16" {:bytes 2 :native "Half"}
   "BF16" {:bytes 2 :native "BFloat16"}
   "F32" {:bytes 4 :native "Float"}
   "F64" {:bytes 8 :native "Double"}
   "C64" {:bytes 8 :native "ComplexFloat"}
   "C128" {:bytes 16 :native "ComplexDouble"}})

(def ^:private native->dtype
  (into {} (map (fn [[safe {:keys [native]}]] [native safe]) dtype-info)))

(defn- scalar-type
  ^torch$ScalarType [dtype]
  (case dtype
    "BOOL" (torch/kBool)
    "U8" (torch/kByte)
    "I8" (torch/kChar)
    "I16" (torch/kShort)
    "I32" (torch/kInt)
    "I64" (torch/kLong)
    "F16" (torch/kHalf)
    "BF16" (torch/kBFloat16)
    "F32" (torch/kFloat)
    "F64" (torch/kDouble)
    "C64" (torch/kComplexFloat)
    "C128" (torch/kComplexDouble)
    (throw (ex-info "Unsupported SafeTensors dtype" {:dtype dtype}))))

(defn- path
  ^Path [value]
  (if (instance? Path value)
    value
    (Path/of (str value) (make-array String 0))))

(defn- read-fully!
  [^FileChannel channel ^ByteBuffer buffer position]
  (loop [offset (long position)]
    (when (.hasRemaining buffer)
      (let [read-count (.read channel buffer offset)]
        (when (neg? read-count)
          (throw (ex-info "Unexpected end of SafeTensors file"
                          {:position offset})))
        (recur (+ offset read-count)))))
  (.flip buffer))

(defn- write-fully!
  [^FileChannel channel ^ByteBuffer buffer]
  (while (.hasRemaining buffer)
    (.write channel buffer)))

(defn- read-header
  [^FileChannel channel]
  (let [file-size (.size channel)]
    (when (< file-size 8)
      (throw (ex-info "SafeTensors file is shorter than its header length" {})))
    (let [length-buffer (doto (ByteBuffer/allocate 8)
                          (.order ByteOrder/LITTLE_ENDIAN))]
      (read-fully! channel length-buffer 0)
      (let [header-length (.getLong length-buffer)]
        (when (or (neg? header-length)
                  (> header-length (- file-size 8))
                  (> header-length (* 100 1024 1024)))
          (throw (ex-info "Invalid SafeTensors header length"
                          {:header-length header-length
                           :file-size file-size})))
        (let [header-buffer (ByteBuffer/allocate
                             (Math/toIntExact header-length))]
          (read-fully! channel header-buffer 8)
          {:data-start (+ 8 header-length)
           :data-length (- file-size 8 header-length)
           :header (json/read-str
                    (String. (.array header-buffer)
                             StandardCharsets/UTF_8))})))))

(defn- element-count
  [shape]
  (reduce * 1 (map long shape)))

(defn- validate-entry!
  [tensor-name data-length {:strs [dtype shape data_offsets]}]
  (let [{bytes-per-element :bytes} (get dtype-info dtype)
        [start end] data_offsets
        expected-bytes (when bytes-per-element
                         (* bytes-per-element (element-count shape)))]
    (when-not (and (string? tensor-name)
                   bytes-per-element
                   (vector? shape)
                   (every? #(and (integer? %) (not (neg? (long %)))) shape)
                   (= 2 (count data_offsets))
                   (integer? start)
                   (integer? end)
                   (<= 0 start end data-length)
                   (= expected-bytes (- end start)))
      (throw (ex-info "Invalid SafeTensors tensor entry"
                      {:name tensor-name
                       :dtype dtype
                       :shape shape
                       :data-offsets data_offsets
                       :data-length data-length})))
    {:name tensor-name
     :dtype dtype
     :shape (mapv long shape)
     :start (long start)
     :end (long end)}))

(defn- validate-entries!
  [header data-length]
  (let [validated-entries (->> (dissoc header "__metadata__")
                               (map (fn [[tensor-name entry]]
                                      (validate-entry! tensor-name data-length entry)))
                               (sort-by :start)
                               vec)]
    (loop [previous-end (long 0)
           [entry & remaining] validated-entries]
      (when entry
        (when-not (= previous-end (:start entry))
          (throw (ex-info "SafeTensors data offsets overlap or contain gaps"
                          {:previous-end previous-end
                           :entry entry})))
        (recur (long (:end entry)) remaining)))
    (when-not (= data-length (or (:end (peek validated-entries)) 0))
      (throw (ex-info "SafeTensors data does not cover the file"
                      {:data-length data-length
                       :last-offset (or (:end (peek validated-entries)) 0)})))
    validated-entries))

(defn- tensor-options
  ^TensorOptions [dtype]
  (with-open [options (TensorOptions.)
              scalar (ScalarTypeOptional. (scalar-type dtype))]
    (.dtype options scalar)))

(defn- read-tensor
  [^FileChannel channel data-start {:keys [dtype shape start end]}]
  (let [byte-count (- end start)
        options (tensor-options dtype)
        buffer (ByteBuffer/allocate (Math/toIntExact byte-count))]
    (try
      (read-fully! channel buffer (+ data-start start))
      (with-open [pointer (BytePointer. (.array buffer))
                  view (torch/from_blob pointer (long-array shape) options)]
        (.clone ^Tensor view))
      (finally
        (.close options)))))

(defn load-file
  "Loads a SafeTensors file as a map of string tensor names to CPU tensors.

  The caller owns the returned tensors and must release them with their model or
  enclosing native-memory scope. Pickle and executable metadata are never read."
  [file]
  (when-not (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder))
    (throw (UnsupportedOperationException.
            "SafeTensors requires little-endian native tensor storage")))
  (with-open [random-access (RandomAccessFile. (str (path file)) "r")
              channel (.getChannel random-access)]
    (let [{:keys [data-start data-length header]} (read-header channel)
          validated-entries (validate-entries! header data-length)]
      (into {}
            (map (fn [{tensor-name :name :as entry}]
                   [tensor-name (read-tensor channel data-start entry)]))
            validated-entries))))

(defn metadata
  "Returns the string metadata map from a SafeTensors file."
  [file]
  (with-open [random-access (RandomAccessFile. (str (path file)) "r")
              channel (.getChannel random-access)]
    (or (get-in (read-header channel) [:header "__metadata__"]) {})))

(defn- tensor-entry
  [offset [tensor-name value]]
  (when-not (and (string? tensor-name) (not= tensor-name "__metadata__"))
    (throw (ex-info "SafeTensors tensor names must be strings other than __metadata__"
                    {:name tensor-name})))
  (let [tensor (t/->tensor value)
        native-name (.toString (.scalar_type tensor))
        dtype (get native->dtype native-name)
        byte-count (Math/multiplyExact (.numel tensor) (.element_size tensor))]
    (when-not dtype
      (throw (ex-info "Tensor dtype is unsupported by SafeTensors"
                      {:name tensor-name :dtype native-name})))
    [{:name tensor-name
      :tensor tensor
      :dtype dtype
      :shape (t/size tensor)
      :start offset
      :end (Math/addExact (long offset) byte-count)}
     (Math/addExact (long offset) byte-count)]))

(defn- entries
  [tensors]
  (first
   (reduce (fn [[result offset] pair]
             (let [[entry next-offset] (tensor-entry offset pair)]
               [(conj result entry) next-offset]))
           [[] 0]
           (sort-by key tensors))))

(defn- padded-header
  [tensor-entries file-metadata]
  (let [tensor-header
        (into (sorted-map)
              (map (fn [{tensor-name :name :keys [dtype shape start end]}]
                     [tensor-name {"dtype" dtype
                                   "shape" shape
                                   "data_offsets" [start end]}]))
              tensor-entries)
        header (cond-> tensor-header
                 (seq file-metadata) (assoc "__metadata__" file-metadata))
        header-bytes (.getBytes (json/write-str header) StandardCharsets/UTF_8)
        padding (mod (- 8 (mod (alength header-bytes) 8)) 8)
        result (byte-array (+ (alength header-bytes) padding))]
    (System/arraycopy header-bytes 0 result 0 (alength header-bytes))
    (java.util.Arrays/fill result (alength header-bytes) (alength result) (byte 32))
    result))

(defn- write-tensor!
  [^FileChannel channel tensor]
  (t/with-torch
    (let [^Tensor contiguous (.contiguous (.cpu (.detach (t/->tensor tensor))))
          byte-count (* (.numel contiguous) (.element_size contiguous))]
      (when (pos? byte-count)
        (let [^Pointer pointer (.data_ptr contiguous)]
          (.capacity pointer byte-count)
          (write-fully! channel (.asByteBuffer pointer))))
      nil)))

(defn save-file!
  "Atomically writes a map of string tensor names to a SafeTensors file.

  Optional `:metadata` must be a string-to-string map. Tensors are detached,
  made contiguous, and copied to CPU before their raw bytes are written."
  [tensors file & {:as options}]
  (let [file-metadata (get options :metadata {})]
    (when-not (map? tensors)
      (throw (IllegalArgumentException. "tensors must be a map")))
    (when-not (and (map? file-metadata)
                   (every? (fn [[metadata-key metadata-value]]
                             (and (string? metadata-key) (string? metadata-value)))
                           file-metadata))
      (throw (IllegalArgumentException.
              "metadata must contain only string keys and values")))
    (when-not (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder))
      (throw (UnsupportedOperationException.
              "SafeTensors requires little-endian native tensor storage")))
    (let [destination (path file)
          parent (or (.getParent destination)
                     (Path/of "." (make-array String 0)))
          tensor-entries (entries tensors)
          ^bytes header (padded-header tensor-entries file-metadata)]
      (Files/createDirectories parent
                               (make-array java.nio.file.attribute.FileAttribute 0))
      (let [temporary
            (Files/createTempFile
             parent
             (str "." (.getFileName destination) "-")
             ".tmp"
             (make-array java.nio.file.attribute.FileAttribute 0))]
        (try
          (with-open [channel (FileChannel/open
                               temporary
                               (into-array java.nio.file.OpenOption
                                           [StandardOpenOption/WRITE
                                            StandardOpenOption/TRUNCATE_EXISTING]))]
            (let [length-buffer (doto (ByteBuffer/allocate 8)
                                  (.order ByteOrder/LITTLE_ENDIAN)
                                  (.putLong (long (alength header)))
                                  (.flip))]
              (write-fully! channel length-buffer)
              (write-fully! channel (ByteBuffer/wrap header))
              (doseq [{:keys [tensor]} tensor-entries]
                (write-tensor! channel tensor))
              (.force channel true)))
          (try
            (Files/move temporary destination
                        (into-array java.nio.file.CopyOption
                                    [StandardCopyOption/ATOMIC_MOVE
                                     StandardCopyOption/REPLACE_EXISTING]))
            (catch AtomicMoveNotSupportedException _
              (Files/move temporary destination
                          (into-array java.nio.file.CopyOption
                                      [StandardCopyOption/REPLACE_EXISTING]))))
          (finally
            (Files/deleteIfExists temporary))))
      destination)))

(defn load-state-dict!
  "Loads a SafeTensors file into a model using flat dotted parameter names."
  [model file]
  (t/with-torch
    (let [state (load-file file)]
      (nn/load-flat-state-dict! model state)
      nil))
  model)
