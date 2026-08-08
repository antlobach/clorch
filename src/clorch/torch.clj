(ns clorch.torch
  "Core Tensor operations and factory methods for Clorch.
   
   Built on LibTorch C++. All operations support automatic differentiation when 
   `:requires-grad true` is set on the input tensors.
   
   ## Memory Management
   Most tensor operations create temporary native C++ objects. Use the `with-torch`
   macro in your hot loops to ensure these are instantly cleaned up.
   
   ## Dtypes
   Supported keywords: `:float32`, `:float64`, `:int32`, `:int64`, `:int8`, `:uint8`, `:bool`."
  (:refer-clojure :exclude [add sub mul div abs float double byte short int long boolean
                            rand rand-int min max
                            concat split flatten load cat empty all any sort repeat chunk])
  (:require [clorch.platform :as platform]
            [tech.v3.datatype.native-buffer :as native]
            [tech.v3.tensor :as dtt]
            [tech.v3.tensor.dimensions :as dims]
            [clojure.string :as str])
  (:import [org.bytedeco.pytorch NoGradGuard Tensor Device DeviceOptional JitModule IValue IValueVector MemoryFormatOptional TensorIndex TensorIndexVector Slice SymInt SymIntOptional LongOptional DoubleOptional StringViewOptional Scalar ScalarOptional ScalarTypeOptional DoubleArrayRef]
           [org.bytedeco.pytorch.global torch torch$ScalarType]
           [org.bytedeco.javacpp PointerScope Pointer DoublePointer]))

(platform/configure-platform-extension!)

;; --- Memory Management ---

(def ^:private ^ThreadLocal session-scope (ThreadLocal.))

(defn release!
  "Manually deletes the native C++ memory associated with a tensor or collection of tensors.
   Once released, the tensor object becomes invalid and cannot be used for further operations.
   Non-tensor values in collections are ignored.
   
   Parameter types:
   - node: Tensor or a collection/map containing Tensors.
   
   Example:
   (let [a (randn [100 100])]
     (do-something a)
     (release! a))"
  [node]
  (cond
    (instance? Pointer node) (.deallocate ^Pointer node)
    (map? node) (run! release! (vals node))
    (coll? node) (run! release! node))
  nil)

(defn retain!
  "Rescues a tensor from its current native memory scope, preventing it from being 
   automatically deleted when the scope (like with-torch) exits. 
   The tensor will instead be managed by the JVM Garbage Collector.
   Non-tensor values in collections are ignored.
   
   Parameter types:
   - node: Tensor or a collection/map containing Tensors.
   
   Return type:
   - Returns the input node.
   
   Example:
   (with-torch
     (let [a (randn [2 2])]
       (retain! a)
       a)) ;; a stays alive after block exits"
  [node]
  (cond
    (instance? Pointer node) (.retainReference ^Pointer node)
    (map? node) (run! retain! (vals node))
    (coll? node) (run! retain! node))
  node)

(defn gc!
  "Triggers aggressive native memory cleanup by prodding the JVM Garbage Collector.
   Useful in interactive sessions if you are creating many tensors without a scope."
  []
  (System/gc)
  (System/runFinalization)
  nil)

(defn start-session!
  "Starts a stateful interactive session scope for the CURRENT THREAD. 
   All tensors created after this call in this thread will be tracked by this 
   session and can be cleared all at once using stop-session!.
   
   This is ideal for REPL development where you don't want to use with-torch for every command."
  []
  (when-let [old (.get session-scope)]
    (.close ^PointerScope old))
  (.set session-scope (PointerScope.))
  nil)

(defn stop-session!
  "Closes the current thread's interactive session scope and deletes all tensors created within it."
  []
  (when-let [old (.get session-scope)]
    (.close ^PointerScope old)
    (.set session-scope nil))
  nil)

(defn rescue-pointers! [node]
  (retain! node))

(defmacro with-torch
  "Executes body in a native memory scope.
   
   Tensors created inside this block that are NOT returned as the final result 
   will be instantly deleted in C++ memory when the block exits. Returned 
   tensors are 'rescued' and stay alive as long as your Clojure references.
   
   To keep a tensor alive that is NOT the return value, use (retain! x).
   
   Example:
   (with-torch 
     (let [a (tensor [1 2])
           b (tensor [3 4])]
       (add a b)))"
  [& body]
  `(with-open [scope# (PointerScope.)]
     (let [result# (do ~@body)]
       (clorch.torch/retain! result#)
       result#)))

;; --- Configuration Maps ---

(def dtype-map
  {:float32 (torch/kFloat)
   :float64 (torch/kDouble)
   :int32 (torch/kInt)
   :int64 (torch/kLong)
   :int8 (torch/kChar)
   :uint8 (torch/kByte)
   :bool (torch/kBool)
   :float16 (torch/kHalf)
   :bfloat16 (torch/kBFloat16)
   :complex64 (torch/kComplexFloat)
   :complex128 (torch/kComplexDouble)
   :qint8 (torch/kQInt8)
   :quint8 (torch/kQUInt8)
   :qint32 (torch/kQInt32)})

(def tech-dtype-info
  {:Float {:kw :float32 :bytes 4}
   :Double {:kw :float64 :bytes 8}
   :Int {:kw :int32 :bytes 4}
   :Long {:kw :int64 :bytes 8}
   :Char {:kw :int8 :bytes 1}
   :Byte {:kw :uint8 :bytes 1}
   :Bool {:kw :uint8 :bytes 1}
   ;; Fallbacks for other types if they are convertible or viewable
   :Half {:kw :float32 :bytes 2}
   :BFloat16 {:kw :float32 :bytes 2}
   :ComplexFloat {:kw :float32 :bytes 8}
   :ComplexDouble {:kw :float64 :bytes 16}
   :QInt8 {:kw :int8 :bytes 1}
   :QUInt8 {:kw :uint8 :bytes 1}
   :QInt32 {:kw :int32 :bytes 4}})

;; --- Core API ---

(defprotocol ITensorContainer
  "Protocol for objects that wrap or contain a native LibTorch Tensor."
  (get-tensor [this] "Returns the underlying Tensor object."))

(extend-type org.bytedeco.pytorch.Tensor
  ITensorContainer
  (get-tensor [this] this))

(defn tensor
  "Creates a new Tensor from Clojure data (scalars, vectors, or sequences).

   Parameters:
   - data: Clojure sequence, number, or native Tensor.
   - opts (optional): A map of options:
     - :dtype (Keyword): The desired data type (e.g., :float32, :int64, :bool).
     - :device (String/Keyword): The device to place the tensor on (e.g., \"cpu\", \"cuda:0\").
     - :requires-grad (Boolean): If true, autograd will record operations on this tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (tensor [[1 2] [3 4]] {:dtype :float32 :requires-grad true})
   ```"
  [data & [{:keys [dtype] :as opts}]]
  (when (nil? data)
    (throw (IllegalArgumentException. "data cannot be nil")))
  (when (and (or (vector? data) (seq? data)) (empty? data))
    (throw (IllegalArgumentException. "data cannot be empty")))
  (let [t (cond
            (instance? Tensor data) data
            (and (record? data) (:tensor data)) (:tensor data)
            (number? data) (torch/tensor (clojure.core/float data))
            (or (vector? data) (seq? data))
            (let [shape (loop [curr data s []] (if (coll? curr) (recur (first curr) (conj s (count curr))) s))
                  flat-data (clojure.core/flatten data)]
              (cond
                (= dtype :bool)
                (let [bs (byte-array (map #(if % (clojure.core/byte 1) (clojure.core/byte 0)) flat-data))
                      byte-t (torch/tensor bs)
                      bool-t (.to byte-t (torch/kBool) false false (MemoryFormatOptional.))]
                  (torch/reshape bool-t (long-array shape)))

                (= dtype :int64) (torch/reshape (torch/tensor (long-array flat-data)) (long-array shape))
                (= dtype :int32) (torch/reshape (torch/tensor (int-array flat-data)) (long-array shape))
                (= dtype :float64) (let [values (double-array flat-data)
                                         ptr (DoublePointer. values)]
                                     (torch/reshape (torch/tensor (DoubleArrayRef. ptr (alength values))) (long-array shape)))
                :else (torch/reshape (torch/tensor (float-array flat-data)) (long-array shape))))
            :else (throw (IllegalArgumentException. (str "Unsupported data type: " (type data)))))]
    (if opts
      (let [{:keys [dtype device requires-grad]} opts
            t-dtype (if dtype
                      (let [stype (get dtype-map dtype (torch/kFloat))]
                        (if (not= (.scalar_type ^Tensor t) stype)
                          (.to ^Tensor t stype false false (MemoryFormatOptional.))
                          t))
                      t)
            t-dev (if device
                    (let [dev (Device. (clojure.core/str (name device)))]
                      (.to ^Tensor t-dtype dev (.scalar_type t-dtype) false false (MemoryFormatOptional.)))
                    t-dtype)
            t-grad (if (some? requires-grad) (.set_requires_grad ^Tensor t-dev (clojure.core/boolean requires-grad)) t-dev)]
        t-grad)
      t)))

(defn ->tensor [x]
  (cond
    (instance? org.bytedeco.pytorch.Tensor x) x
    (instance? org.bytedeco.pytorch.Module x) x
    (satisfies? ITensorContainer x) (get-tensor x)
    :else (tensor x)))

;; --- Factory Methods ---

(def inf Float/POSITIVE_INFINITY)
(def -inf Float/NEGATIVE_INFINITY)

(defn randn
  "Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1.

   Parameters:
   - dims: Sequence of Longs representing the shape of the output tensor.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (randn [2 3] {:device :cuda})
   ```"
  [dims & [opts]]
  (let [t (torch/randn (long-array dims))]
    (if opts (tensor t opts) t)))

(defn rand
  "Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).

   Parameters:
   - dims: Sequence of Longs representing the shape of the output tensor.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (rand [2 3] {:device :cuda})
   ```"
  [dims & [opts]]
  (let [t (torch/rand (long-array dims))]
    (if opts (tensor t opts) t)))

(defn ones
  "Returns a tensor filled with the scalar value 1.

   Parameters:
   - dims: Sequence of Longs representing the shape of the output tensor.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (ones [5] {:device :cuda})
   ```"
  [dims & [opts]]
  (let [t (torch/ones (long-array dims))]
    (if opts (tensor t opts) t)))

(defn zeros
  "Returns a tensor filled with the scalar value 0.

   Parameters:
   - dims: Sequence of Longs representing the shape of the output tensor.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (zeros [2 2] {:device :cuda})
   ```"
  [dims & [opts]]
  (let [t (torch/zeros (long-array dims))]
    (if opts (tensor t opts) t)))

(defn eye
  "Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

   Parameters:
   - n: Number of rows/cols in the 2-D tensor.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (eye 3)
   ```"
  [n & [opts]]
  (let [t (torch/eye (clojure.core/long n))]
    (if opts (tensor t opts) t)))

(defn tril
  "Returns the lower triangular part of a matrix (2-D tensor) or batch of matrices.

   Parameters:
   - t: The input Tensor.
   - diagonal (optional): The diagonal to consider. 0 is the main diagonal, 
     positive values are above, negative are below.

   Returns: A new Tensor.

   Example:
   ```clojure
   (tril (ones [3 3]))
   ```"
  ([t] (torch/tril (->tensor t)))
  ([t diagonal] (torch/tril (->tensor t) (clojure.core/long diagonal))))

(defn triu
  "Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices.

   Parameters:
   - t: The input Tensor.
   - diagonal (optional): The diagonal to consider. 0 is the main diagonal, 
     positive values are above, negative are below.

   Returns: A new Tensor.

   Example:
   ```clojure
   (triu (ones [3 3]))
   ```"
  ([t] (torch/triu (->tensor t)))
  ([t diagonal] (torch/triu (->tensor t) (clojure.core/long diagonal))))

(defn rand-int
  "Returns a tensor of integers randomly sampled from the interval [low, high).

   Parameters:
   - low: Lower bound (inclusive).
   - high: Upper bound (exclusive).
   - dims: Sequence of Longs representing the shape of the output tensor.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (rand-int 0 10 [5])
   ```"
  [low high dims & [opts]]
  (let [t (torch/randint (clojure.core/long low) (clojure.core/long high) (long-array dims))]
    (if opts (tensor t opts) t)))

(defn bernoulli
  "Draws binary random numbers (0 or 1) from a Bernoulli distribution.

   Parameters:
   - input: A Tensor of probabilities.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (bernoulli (full [5] 0.5))
   ```"
  [input & [opts]]
  (let [t (torch/bernoulli (->tensor input) (org.bytedeco.pytorch.GeneratorOptional.))]
    (if opts (tensor t opts) t)))

(defn manual-seed
  "Sets the seed for generating random numbers for all devices.

   Parameters:
   - seed: The seed value (Long).

   Returns: nil.

   Example:
   ```clojure
   (manual-seed 42)
   ```"
  [seed]
  (torch/manual_seed (clojure.core/long seed)))

(defn randperm
  "Returns a tensor containing a random permutation of integers from 0 to n-1.

   Parameters:
   - n: The number of elements.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (randperm 10)
   ```"
  [n & [opts]]
  (let [t (torch/randperm (clojure.core/long n))]
    (if opts (tensor t opts) t)))

(defn empty
  "Returns a tensor filled with uninitialized data.

   Parameters:
   - dims: Sequence of Longs representing the shape of the output tensor.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (empty [2 3])
   ```"
  [dims & [opts]]
  (let [t (torch/empty (long-array dims))]
    (if opts (tensor t opts) t)))

(defn empty-strided
  "Returns a tensor filled with uninitialized data with the given shape and strides.

   Parameters:
   - size: Sequence of Longs representing the shape.
   - stride: Sequence of Longs representing the strides.
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (empty-strided [2 3] [3 1])
   ```"
  [size stride & [opts]]
  (let [t (torch/empty_strided (long-array size) (long-array stride))]
    (if opts (tensor t opts) t)))

(defn full
  "Returns a tensor of size `dims` filled with `value`.

   Parameters:
   - dims: Sequence of Longs representing the shape.
   - value: The constant value to fill the tensor with (Number).
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (full [2 3] 3.14)
   ```"
  [dims value & [opts]]
  (let [t (torch/full (long-array dims) (Scalar. (clojure.core/double value)))]
    (if opts (tensor t opts) t)))

(defn linspace
  "Returns a one-dimensional tensor of `steps` points equally spaced between `start` and `end`.

   Parameters:
   - start: The starting value (Number).
   - end: The ending value (Number).
   - steps: Number of samples to generate (Long).
   - opts (optional): A map of options (see `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linspace 0 10 5)
   ```"
  [start end steps & [opts]]
  (let [t (torch/linspace (Scalar. (clojure.core/double start))
                          (Scalar. (clojure.core/double end))
                          (clojure.core/long steps))]
    (if opts (tensor t opts) t)))

(defn logspace
  "Returns a one-dimensional tensor of `steps` points logarithmically spaced between `base^start` and `base^end`.

   Parameters:
   - start: Starting exponent (Number).
   - end: Ending exponent (Number).
   - steps: Number of samples to generate (Long).
   - opts (optional): A map of options:
     - :base (Number): The base for the logarithms (default: 10.0).
     - (plus other options from `tensor`).

   Returns: A new Tensor.

   Example:
   ```clojure
   (logspace 0 10 5 :base 2)
   ```"
  [start end steps & {:keys [base] :or {base 10.0} :as opts}]
  (let [t (torch/logspace (Scalar. (clojure.core/double start))
                          (Scalar. (clojure.core/double end))
                          (clojure.core/long steps)
                          (clojure.core/double base)
                          (org.bytedeco.pytorch.TensorOptions.))]
    (if (not-empty (dissoc opts :base))
      (tensor t (dissoc opts :base))
      t)))

(defn arange
  "Returns a 1-D tensor with values from the interval [start, end) with common difference `step`.

   Parameters:
   - end: Upper bound (exclusive).
   - start (optional): Lower bound (inclusive, default: 0).
   - step (optional): Difference between adjacent values (default: 1).

   Returns: A new Tensor.

   Example:
   ```clojure
   (arange 0 10 2)
   ```"
  ([end] (torch/arange (org.bytedeco.pytorch.Scalar. (clojure.core/double end))))
  ([start end] (torch/arange (org.bytedeco.pytorch.Scalar. (clojure.core/double start))
                             (org.bytedeco.pytorch.Scalar. (clojure.core/double end))))
  ([start end step] (torch/arange (org.bytedeco.pytorch.Scalar. (clojure.core/double start))
                                  (org.bytedeco.pytorch.Scalar. (clojure.core/double end))
                                  (org.bytedeco.pytorch.Scalar. (clojure.core/double step)))))

(defn masked-fill
  "Fills elements of `t` tensor with `value` where `mask` is True.

   Parameters:
   - t: The input Tensor.
   - mask: A Boolean Tensor with the same shape as `t` (or broadcastable).
   - value: The fill value (Number).

   Returns: A new Tensor.

   Example:
   ```clojure
   (masked-fill (zeros [3 3]) (eye 3) 1.0)
   ```"
  [t mask value]
  (.masked_fill (->tensor t) (->tensor mask) (org.bytedeco.pytorch.Scalar. (clojure.core/double value))))

;; --- Operations ---

(defn size
  "Returns the shape of the tensor.
   Example: (size t) => [2 3]"
  ([t]
   (let [t-obj (->tensor t)
         size-ptr (.sizes t-obj)]
     (mapv #(.get size-ptr (clojure.core/long %)) (range (.dim t-obj)))))
  ([t dim] (.size (->tensor t) (clojure.core/long dim))))

(defn dtype
  "Returns the dtype of the tensor as a keyword."
  [t]
  (let [stype (.toString (.scalar_type (->tensor t)))]
    (:kw (get tech-dtype-info (keyword stype) {:kw :unknown}))))

(defn item-float
  "Returns the value of a 1-element tensor as a float.

   Parameters:
   - t: A Tensor with exactly one element.

   Returns: A Float value.

   Example:
   ```clojure
   (item-float (tensor [1.5]))
   ```"
  [t] (.item_float (->tensor t)))

(defn eq
  "Element-wise equality comparison.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (eq (tensor [1 2]) (tensor [1 3]))
   ```"
  [a b]
  (if (instance? Tensor b)
    (.eq (->tensor a) b)
    (.eq (->tensor a) (Scalar. (clojure.core/double b)))))

(defn gt
  "Element-wise greater-than comparison.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (gt (tensor [1 2]) 1)
   ```"
  [a b]
  (if (instance? Tensor b)
    (.gt (->tensor a) b)
    (.gt (->tensor a) (Scalar. (clojure.core/double b)))))

(defn lt
  "Element-wise less-than comparison.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (lt (tensor [1 2]) 2)
   ```"
  [a b]
  (if (instance? Tensor b)
    (.lt (->tensor a) b)
    (.lt (->tensor a) (Scalar. (clojure.core/double b)))))

(defn ge
  "Element-wise greater-than-or-equal-to comparison.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (ge (tensor [1 2]) 1)
   ```"
  [a b]
  (if (instance? Tensor b)
    (.ge (->tensor a) b)
    (.ge (->tensor a) (Scalar. (clojure.core/double b)))))

(defn le
  "Element-wise less-than-or-equal-to comparison.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (le (tensor [1 2]) 2)
   ```"
  [a b]
  (if (instance? Tensor b)
    (.le (->tensor a) b)
    (.le (->tensor a) (Scalar. (clojure.core/double b)))))

(defn ne
  "Element-wise not-equal comparison.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (ne (tensor [1 2]) 1)
   ```"
  [a b]
  (if (instance? Tensor b)
    (.ne (->tensor a) b)
    (.ne (->tensor a) (Scalar. (clojure.core/double b)))))

(defn logical-and
  "Computes the element-wise logical AND of two tensors.

   Parameters:
   - a: Input Tensor.
   - b: Input Tensor.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (logical-and (tensor [true false]) (tensor [true true]))
   ```"
  [a b] (torch/logical_and (->tensor a) (->tensor b)))

(defn logical-or
  "Computes the element-wise logical OR of two tensors.

   Parameters:
   - a: Input Tensor.
   - b: Input Tensor.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (logical-or (tensor [true false]) (tensor [false false]))
   ```"
  [a b] (torch/logical_or (->tensor a) (->tensor b)))

(defn logical-xor
  "Computes the element-wise logical XOR of two tensors.

   Parameters:
   - a: Input Tensor.
   - b: Input Tensor.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (logical-xor (tensor [true false]) (tensor [true true]))
   ```"
  [a b] (torch/logical_xor (->tensor a) (->tensor b)))

(defn logical-not
  "Computes the element-wise logical NOT of a tensor.

   Parameters:
   - a: Input Tensor.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (logical-not (tensor [true false]))
   ```"
  [a] (torch/logical_not (->tensor a)))

(defn gather
  "Gathers values along an axis specified by `dim`.

   Parameters:
   - t: The source Tensor.
   - dim: The axis along which to index (Long).
   - index: The indices of elements to gather (Long Tensor).

   Returns: A new Tensor.

   Example:
   ```clojure
   (gather (tensor [[1 2] [3 4]]) 1 (tensor [[0 0] [1 0]]))
   ```"
  [t dim index]
  (torch/gather (->tensor t) (clojure.core/long dim) (->tensor index)))

(defn argmax
  "Returns the indices of the maximum values of all elements in the input tensor.
   
   If `dim` is provided, returns the indices along that dimension.

   Parameters:
   - t: The input Tensor.
   - dim (optional): The dimension to reduce.

   Returns: A new Tensor (Long).

   Example:
   ```clojure
   (argmax (tensor [1 3 2]))
   ```"
  ([t] (.argmax (->tensor t)))
  ([t dim] (.argmax (->tensor t) (LongOptional. (clojure.core/long dim)) false)))

(defn argmin
  "Returns the indices of the minimum values of all elements in the input tensor.
   
   If `dim` is provided, returns the indices along that dimension.

   Parameters:
   - t: The input Tensor.
   - dim (optional): The dimension to reduce.

   Returns: A new Tensor (Long).

   Example:
   ```clojure
   (argmin (tensor [1 3 2]))
   ```"
  ([t] (.argmin (->tensor t)))
  ([t dim] (.argmin (->tensor t) (LongOptional. (clojure.core/long dim)) false)))

(defn topk
  "Returns the `k` largest elements of the given input tensor along a given dimension.

   Parameters:
   - t: The input Tensor.
   - k: The number of top elements to return (Long).
   - opts:
     - :dim (Long): The dimension to sort along (default: -1).
     - :largest (Boolean): Whether to return the largest or smallest elements (default: true).
     - :sorted (Boolean): Whether to return the elements in sorted order (default: true).

   Returns: A vector [values indices] (Tensors).

   Example:
   ```clojure
   (topk (tensor [1 5 2 4]) 2)
   ```"
  [t k & {:keys [dim largest sorted] :or {dim -1 largest true sorted true}}]
  (let [res (torch/topk (->tensor t) (clojure.core/long k) (clojure.core/long dim) (clojure.core/boolean largest) (clojure.core/boolean sorted))]
    [(.get0 res) (.get1 res)]))

(defn sort
  "Sorts the elements of the input tensor along a given dimension in ascending order by value.

   Parameters:
   - t: The input Tensor.
   - opts:
     - :dim (Long): The dimension to sort along (default: -1).
     - :descending (Boolean): Whether to sort in descending order (default: false).

   Returns: A vector [values indices] (Tensors).

   Example:
   ```clojure
   (sort (tensor [1 5 2 4]))
   ```"
  [t & {:keys [dim descending] :or {dim -1 descending false}}]
  (let [res (torch/sort (->tensor t) (clojure.core/long dim) (clojure.core/boolean descending))]
    [(.get0 res) (.get1 res)]))

(defn where
  "Return a tensor of elements selected from either x or y, depending on condition.
   
   If `condition` is True, select from `x`, otherwise select from `y`.

   Parameters:
   - condition: A Boolean Tensor.
   - x: First input Tensor.
   - y: Second input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (where (gt (tensor [1 2 3]) 2) (ones [3]) (zeros [3]))
   ```"
  [condition x y]
  (torch/where (->tensor condition) (->tensor x) (->tensor y)))

(defn nonzero
  "Returns a tensor containing the indices of all non-zero elements of `t`.

   Parameters:
   - t: The input Tensor.

   Returns: A new Tensor (Long) of shape (N, dim).

   Example:
   ```clojure
   (nonzero (tensor [1 0 2]))
   ```"
  [t]
  (torch/nonzero (->tensor t)))

(defn all
  "Tests if all elements in the input tensor evaluate to True.

   Parameters:
   - t: The input Tensor.
   - dim (optional): The dimension to reduce.
   - opts:
     - :keepdim (Boolean): Whether the output tensor has `dim` retained or not.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (all (tensor [true true]))
   ```"
  ([t] (torch/all (->tensor t)))
  ([t dim & {:keys [keepdim] :or {keepdim false}}]
   (torch/all (->tensor t) (clojure.core/long dim) (clojure.core/boolean keepdim))))

(defn any
  "Tests if any element in the input tensor evaluates to True.

   Parameters:
   - t: The input Tensor.
   - dim (optional): The dimension to reduce.
   - opts:
     - :keepdim (Boolean): Whether the output tensor has `dim` retained or not.

   Returns: A Boolean Tensor.

   Example:
   ```clojure
   (any (tensor [true false]))
   ```"
  ([t] (torch/any (->tensor t)))
  ([t dim & {:keys [keepdim] :or {keepdim false}}]
   (torch/any (->tensor t) (clojure.core/long dim) (clojure.core/boolean keepdim))))

(defn clamp
  "Clamp all elements in input into the range [min, max]."
  [t min max]
  (torch/clamp (->tensor t)
               (org.bytedeco.pytorch.ScalarOptional. (org.bytedeco.pytorch.Scalar. (clojure.core/double min)))
               (org.bytedeco.pytorch.ScalarOptional. (org.bytedeco.pytorch.Scalar. (clojure.core/double max)))))

(defn repeat
  "Repeats this tensor along the specified dimensions."
  [t dims]
  (.repeat (->tensor t) (long-array dims)))

(defn softmax
  "Applies the softmax function to the input tensor along a specified dimension.
   Resulting values are in range [0, 1] and sum to 1.

   Parameters:
   - t: The input Tensor.
   - dim: Dimension along which softmax will be computed (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (softmax (tensor [1.0 2.0 3.0]) 0)
   ```"
  [t dim]
  (torch/softmax (->tensor t) (clojure.core/long dim) (ScalarTypeOptional.)))

(defn cumsum
  "Returns the cumulative sum of elements of `t` in the dimension `dim`.

   Parameters:
   - t: The input Tensor.
   - dim: The dimension to do the operation over (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (cumsum (tensor [1 2 3]) 0)
   ```"
  [t dim]
  (torch/cumsum (->tensor t) (clojure.core/long dim) (ScalarTypeOptional.)))

(defn to
  "Convert tensor to dtype or device."
  [t dtype-or-device]
  (if (instance? org.bytedeco.pytorch.Module t)
    t
    (let [t-obj (if (instance? org.bytedeco.pytorch.Tensor t) t (->tensor t))]
      (cond
        (keyword? dtype-or-device)
        (if-let [stype (get dtype-map dtype-or-device)]
          (.to ^Tensor t-obj ^torch$ScalarType stype false false (MemoryFormatOptional.))
          (let [dev (Device. (clojure.core/str (name dtype-or-device)))]
            (.to ^Tensor t-obj dev (.scalar_type t-obj) false false (MemoryFormatOptional.))))

        (instance? Device dtype-or-device)
        (.to ^Tensor t-obj ^Device dtype-or-device (.scalar_type t-obj) false false (MemoryFormatOptional.))

        (instance? torch$ScalarType dtype-or-device)
        (.to ^Tensor t-obj ^torch$ScalarType dtype-or-device false false (MemoryFormatOptional.))

        :else
        (throw (IllegalArgumentException. (str "Unsupported to target: " (type dtype-or-device))))))))

(defn to-float [t] (to t :float32))
(defn to-long [t] (to t :int64))

(defn type-as
  "Returns the input tensor cast to the type of the other tensor.

   Parameters:
   - t: The input Tensor.
   - other: The Tensor whose type should be matched.

   Returns: A new Tensor with the same data as `t` but the dtype of `other`.

   Example:
   ```clojure
   (type-as (ones [2] {:dtype :int32}) (zeros [2] {:dtype :float32}))
   ```"
  [t other]
  (.type_as (->tensor t) (->tensor other)))

(defn add
  "Element-wise addition of two tensors or a tensor and a scalar.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (add (ones [2]) (ones [2]))
   ```"
  [a b] (torch/add (->tensor a) (->tensor b)))

(defn sub
  "Element-wise subtraction of two tensors or a tensor and a scalar.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (sub (ones [2]) 0.5)
   ```"
  [a b] (torch/sub (->tensor a) (->tensor b)))

(defn mul
  "Element-wise multiplication of two tensors or a tensor and a scalar.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (mul (ones [2]) 2.0)
   ```"
  [a b] (torch/mul (->tensor a) (->tensor b)))

(defn div
  "Element-wise division of two tensors or a tensor and a scalar.

   Parameters:
   - a: Input Tensor or Number.
   - b: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (div (ones [2]) 2.0)
   ```"
  [a b] (torch/div (->tensor a) (->tensor b)))

(defn lerp
  "Does a linear interpolation of two tensors based on a weight.
   `out = input + weight * (end - input)`

   Parameters:
   - input: Starting Tensor.
   - end: Ending Tensor.
   - weight: Interpolation weight (Number or Tensor).

   Returns: A new Tensor.

   Example:
   ```clojure
   (lerp (zeros [2]) (ones [2]) 0.5)
   ```"
  [input end weight]
  (if (number? weight)
    (torch/lerp (->tensor input) (->tensor end) (org.bytedeco.pytorch.Scalar. (clojure.core/double weight)))
    (torch/lerp (->tensor input) (->tensor end) (->tensor weight))))

(defn addcmul
  "Performs the element-wise multiplication of `tensor1` by `tensor2`, multiplies the result by `value` and adds it to `input`.
   `out = input + value * tensor1 * tensor2`

   Parameters:
   - input: Tensor to be added to.
   - tensor1: First multiplier Tensor.
   - tensor2: Second multiplier Tensor.
   - value (optional): Scalar multiplier (Number, default: 1.0).

   Returns: A new Tensor.

   Example:
   ```clojure
   (addcmul (ones [2]) (ones [2]) (ones [2]) :value 2.0)
   ```"
  [input tensor1 tensor2 & {:keys [value] :or {value 1.0}}]
  (torch/addcmul (->tensor input) (->tensor tensor1) (->tensor tensor2) (org.bytedeco.pytorch.Scalar. (clojure.core/double value))))

(defn addcdiv
  "Performs the element-wise division of `tensor1` by `tensor2`, multiplies the result by `value` and adds it to `input`.
   `out = input + value * tensor1 / tensor2`

   Parameters:
   - input: Tensor to be added to.
   - tensor1: Dividend Tensor.
   - tensor2: Divisor Tensor.
   - value (optional): Scalar multiplier (Number, default: 1.0).

   Returns: A new Tensor.

   Example:
   ```clojure
   (addcdiv (ones [2]) (ones [2]) (full [2] 2) :value 2.0)
   ```"
  [input tensor1 tensor2 & {:keys [value] :or {value 1.0}}]
  (torch/addcdiv (->tensor input) (->tensor tensor1) (->tensor tensor2) (org.bytedeco.pytorch.Scalar. (clojure.core/double value))))

(defn pow
  "Computes the power of each element of `a` with exponent `b`.

   Parameters:
   - a: Input Tensor or Number.
   - b: Exponent (Number or Tensor).

   Returns: A new Tensor.

   Example:
   ```clojure
   (pow (tensor [2 3]) 2)
   ```"
  [a b]
  (if (number? b)
    (torch/pow (->tensor a) (org.bytedeco.pytorch.Scalar. (clojure.core/double b)))
    (torch/pow (->tensor a) (->tensor b))))

(defn sqrt
  "Returns a new tensor with the square-root of the elements of `a`.

   Parameters:
   - a: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (sqrt (tensor [4 9]))
   ```"
  [a] (torch/sqrt (->tensor a)))

(defn rsqrt
  "Returns a new tensor with the reciprocal of the square-root of each of the elements of `a`.

   Parameters:
   - a: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (rsqrt (tensor [4 9]))
   ```"
  [a] (torch/rsqrt (->tensor a)))

(defn cos
  "Computes the cosine of each element of `a`.

   Parameters:
   - a: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (cos (tensor [0 3.1415]))
   ```"
  [a] (torch/cos (->tensor a)))

(defn sin
  "Computes the sine of each element of `a`.

   Parameters:
   - a: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (sin (tensor [0 3.1415]))
   ```"
  [a] (torch/sin (->tensor a)))

(defn abs
  "Computes the absolute value of each element of `t`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (abs (tensor [-1 2 -3]))
   ```"
  [t] (torch/abs (->tensor t)))

(defn sign
  "Returns a new tensor with the signs of the elements of `t`: -1, 0, or 1.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (sign (tensor [-5 0 5]))
   ```"
  [t] (torch/sign (->tensor t)))

(defn neg
  "Returns a new tensor with the negative of the elements of `t`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (neg (tensor [1 -2 3]))
   ```"
  [t] (torch/neg (->tensor t)))

(defn exp
  "Computes the exponential of each element of `t`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (exp (tensor [0 1]))
   ```"
  [t] (torch/exp (->tensor t)))

(defn exp2
  "Computes 2 to the power of each element of `t`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (exp2 (tensor [0 1 2]))
   ```"
  [t] (torch/exp2 (->tensor t)))

(defn expm1
  "Computes the exponential of each element minus 1: `e^x - 1`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (expm1 (tensor [0 1]))
   ```"
  [t] (torch/expm1 (->tensor t)))

(defn log
  "Computes the natural logarithm of each element of `t`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (log (tensor [1 2.718]))
   ```"
  [t] (torch/log (->tensor t)))

(defn log1p
  "Computes the natural logarithm of `1 + t`: `log(1 + x)`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (log1p (tensor [0 1]))
   ```"
  [t] (torch/log1p (->tensor t)))

(defn log2
  "Computes the base-2 logarithm of each element of `t`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (log2 (tensor [1 2 4]))
   ```"
  [t] (torch/log2 (->tensor t)))

(defn log10
  "Computes the base-10 logarithm of each element of `t`.

   Parameters:
   - t: Input Tensor or Number.

   Returns: A new Tensor.

   Example:
   ```clojure
   (log10 (tensor [1 10 100]))
   ```"
  [t] (torch/log10 (->tensor t)))

(defn expand
  "Returns a new view of the tensor with singleton dimensions expanded to a larger size.

   Parameters:
   - t: The input Tensor.
   - dims: Sequence of Longs representing the target shape.

   Returns: A new Tensor (view).

   Example:
   ```clojure
   (expand (tensor [1 2 3]) [2 3])
   ```"
  [t dims] (.expand (->tensor t) (long-array dims) true))

(defn repeat-interleave
  "Repeats elements of a tensor along a specified dimension.

   Parameters:
   - t: The input Tensor.
   - repeats: Number of repetitions for each element (Long).
   - dim: The dimension along which to repeat (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (repeat-interleave (tensor [1 2]) 3 0)
   ```"
  [t repeats dim] (.repeat_interleave (->tensor t) (clojure.core/long repeats) (LongOptional. (clojure.core/long dim)) (LongOptional.)))

(defn index-select
  "Returns a new tensor which indexes the input tensor along dimension `dim` using the entries in `index`.

   Parameters:
   - t: The input Tensor.
   - dim: The dimension in which we index (Long).
   - index: A 1-D Tensor containing the indices to extract.

   Returns: A new Tensor.

   Example:
   ```clojure
   (index-select (tensor [[1 2] [3 4]]) 0 (tensor [1]))
   ```"
  [t dim index] (torch/index_select (->tensor t) (clojure.core/long dim) (->tensor index)))

(defn unsqueeze
  "Returns a new tensor with a dimension of size one inserted at the specified position.

   Parameters:
   - t: The input Tensor.
   - dim: Index at which to insert the singleton dimension (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (unsqueeze (tensor [1 2]) 0)
   ```"
  [t dim] (torch/unsqueeze (->tensor t) (clojure.core/long dim)))

(defn outer
  "Computes the outer product of two vectors `a` and `b`.

   Parameters:
   - a: First vector.
   - b: Second vector.

   Returns: A new Matrix (2-D Tensor).

   Example:
   ```clojure
   (outer (tensor [1 2]) (tensor [3 4 5]))
   ```"
  [a b] (torch/outer (->tensor a) (->tensor b)))

(defn matmul
  "Matrix product of two tensors.
   
   Supports broadcasting and batching.

   Parameters:
   - a: First Tensor.
   - b: Second Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (matmul (ones [2 3]) (ones [3 4]))
   ```"
  [a b] (torch/matmul (->tensor a) (->tensor b)))

(defn bmm
  "Performs a batch matrix-matrix product of matrices stored in `a` and `b`.

   Parameters:
   - a: First batch of matrices (B x N x M).
   - b: Second batch of matrices (B x M x P).

   Returns: A new Tensor (B x N x P).

   Example:
   ```clojure
   (bmm (ones [10 2 3]) (ones [10 3 4]))
   ```"
  [a b] (torch/bmm (->tensor a) (->tensor b)))

(defn mm
  "Performs a matrix multiplication of the matrices `a` and `b`.
   Shorthand for `matmul`.

   Parameters:
   - a: First matrix.
   - b: Second matrix.

   Returns: A new Tensor.

   Example:
   ```clojure
   (mm (ones [2 3]) (ones [3 4]))
   ```"
  [a b] (matmul a b))

(defn ->tensor-vector
  "Converts a Clojure sequence of tensors into a native TensorVector."
  [tensors]
  (if (instance? org.bytedeco.pytorch.TensorVector tensors)
    tensors
    (let [tv (org.bytedeco.pytorch.TensorVector.)]
      (doseq [t tensors]
        (.push_back tv (->tensor t)))
      tv)))

(defn einsum
  "Evaluates Einstein summation notation over tensors.

   Example:
   (einsum \"ij,j->i\" [A x])"
  [equation tensors]
  (torch/einsum (clojure.core/str equation) (->tensor-vector tensors)))

(defn ->vector
  "Converts a native collection (TensorVector, etc.) into a Clojure vector."
  [coll]
  (cond
    (instance? org.bytedeco.pytorch.TensorVector coll)
    (mapv #(.get ^org.bytedeco.pytorch.TensorVector coll (clojure.core/long %))
          (range (.size ^org.bytedeco.pytorch.TensorVector coll)))
    :else (vec coll)))

(defn stack
  "Concatenates a sequence of tensors along a new dimension.

   Parameters:
   - tensors: Sequence of Tensors to concatenate.
   - dim: The dimension along which to stack (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (stack [(ones [2]) (ones [2])] 0)
   ```"
  [tensors dim]
  (torch/stack (->tensor-vector tensors) (clojure.core/long dim)))

(defn cat
  "Concatenates the given sequence of tensors along a specified dimension.

   Parameters:
   - tensors: Sequence of Tensors to concatenate.
   - dim: The dimension along which the tensors are concatenated (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (cat [(ones [2 1]) (ones [2 1])] 1)
   ```"
  [tensors dim]
  (torch/cat (->tensor-vector tensors) (clojure.core/long dim)))

(defn chunk
  "Splits a tensor into a specific number of chunks.
   Each chunk is a view of the input tensor.

   Parameters:
   - t: The input Tensor.
   - chunks: Number of chunks to return (Long).
   - dim: Dimension along which to split the tensor (Long).

   Returns: A Clojure vector of Tensors.

   Example:
   ```clojure
   (chunk (ones [4]) 2 0)
   ```"
  [t chunks dim]
  (->vector (torch/chunk (->tensor t) (clojure.core/long chunks) (clojure.core/long dim))))

(defn split
  "Splits the tensor into chunks."
  [t split-size-or-sections dim]
  (if (number? split-size-or-sections)
    (->vector (torch/split (->tensor t) (clojure.core/long split-size-or-sections) (clojure.core/long dim)))
    (let [sections (org.bytedeco.pytorch.LongVector. (long-array split-size-or-sections))
          lar (org.bytedeco.pytorch.LongArrayRef. sections)]
      (->vector (torch/split_with_sizes (->tensor t) lar (clojure.core/long dim))))))

(defn split-with-sizes
  "Splits the tensor into chunks of given sizes."
  [t split-sizes dim]
  (let [sections (org.bytedeco.pytorch.LongVector. (long-array split-sizes))
        lar (org.bytedeco.pytorch.LongArrayRef. sections)]
    (->vector (torch/split_with_sizes (->tensor t) lar (clojure.core/long dim)))))

(defn slice
  "Slices the input tensor along the selected dimension at the given index."
  [t dim start end step]
  (let [opt-start (if start (org.bytedeco.pytorch.LongOptional. (clojure.core/long start)) (org.bytedeco.pytorch.LongOptional.))
        opt-end (if end (org.bytedeco.pytorch.LongOptional. (clojure.core/long end)) (org.bytedeco.pytorch.LongOptional.))]
    (torch/slice (->tensor t) (clojure.core/long dim) opt-start opt-end (clojure.core/long step))))

(defn unbind
  "Removes a tensor dimension."
  [t dim]
  (->vector (torch/unbind (->tensor t) (clojure.core/long dim))))

(defn copy-

  "Copies src tensor into dst tensor in-place."
  [dst src]
  (let [dst-t (->tensor dst)
        src-t (->tensor src)]
    (with-open [_ (NoGradGuard.)]
      (.set_ dst-t src-t))
    dst-t))

(defn multinomial
  "Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input."
  ([t num-samples] (multinomial t num-samples true))
  ([t num-samples replacement]
   (torch/multinomial (->tensor t) (clojure.core/long num-samples) (clojure.core/boolean replacement) (org.bytedeco.pytorch.GeneratorOptional.))))

(defn sum
  "Returns the sum of all elements in the input tensor.
   
   If `dim` is provided, the sum is performed along that dimension.

   Parameters:
   - a: The input Tensor.
   - dim (optional): The dimension or sequence of dimensions to reduce.
   - opts:
     - :keepdim (Boolean): Whether the output tensor has `dim` retained or not.

   Returns: A new Tensor.

   Example:
   ```clojure
   (sum (tensor [1 2 3]))
   (sum (tensor [[1 2] [3 4]]) 0)
   ```"
  ([a] (torch/sum (->tensor a)))
  ([a dim & {:keys [keepdim] :or {keepdim false}}]
   (torch/sum (->tensor a) (long-array (if (number? dim) [dim] dim)) (clojure.core/boolean keepdim) (org.bytedeco.pytorch.ScalarTypeOptional.))))

(defn mean
  "Returns the mean value of all elements in the input tensor.
   
   If `dim` is provided, the mean is performed along that dimension.

   Parameters:
   - a: The input Tensor.
   - dim (optional): The dimension or sequence of dimensions to reduce.
   - opts:
     - :keepdim (Boolean): Whether the output tensor has `dim` retained or not.

   Returns: A new Tensor.

   Example:
   ```clojure
   (mean (tensor [1.0 2.0 3.0]))
   (mean (tensor [[1.0 2.0] [3.0 4.0]]) 1)
   ```"
  ([a] (torch/mean (->tensor a)))
  ([a dim & {:keys [keepdim] :or {keepdim false}}]
   (torch/mean (->tensor a) (long-array (if (number? dim) [dim] dim)) (clojure.core/boolean keepdim) (org.bytedeco.pytorch.ScalarTypeOptional.))))

(defn var
  "Returns the variance of all elements in the input tensor.
   
   If `dim` is provided, the variance is performed along that dimension.

   Parameters:
   - a: The input Tensor.
   - dim (optional): The dimension or sequence of dimensions to reduce.
   - opts:
     - :unbiased (Boolean): Whether to use the unbiased estimation (default: true).
     - :keepdim (Boolean): Whether the output tensor has `dim` retained or not.

   Returns: A new Tensor.

   Example:
   ```clojure
   (var (tensor [1.0 2.0 3.0]))
   ```"
  ([a] (torch/var (->tensor a)))
  ([a dim & {:keys [unbiased keepdim] :or {unbiased true keepdim false}}]
   (torch/var (->tensor a) (long-array (if (number? dim) [dim] dim)) (clojure.core/boolean unbiased) (clojure.core/boolean keepdim))))

(defn max
  "Returns the maximum value of all elements in the input tensor.
   
   If `dim` is provided, returns a tensor with the maximum values along that dimension.

   Parameters:
   - a: The input Tensor.
   - dim (optional): The dimension to reduce.
   - opts:
     - :keepdim (Boolean): Whether the output tensor has `dim` retained or not.

   Returns: A new Tensor.

   Example:
   ```clojure
   (max (tensor [1 3 2]))
   ```"
  ([a] (.max (->tensor a)))
  ([a dim & {:keys [keepdim] :or {keepdim false}}]
   (let [res (.max (->tensor a) (clojure.core/long dim) (clojure.core/boolean keepdim))]
     (.get res (clojure.core/long 0)))))

(defn min
  "Returns the minimum value of all elements in the input tensor.
   
   If `dim` is provided, returns a tensor with the minimum values along that dimension.

   Parameters:
   - a: The input Tensor.
   - dim (optional): The dimension to reduce.
   - opts:
     - :keepdim (Boolean): Whether the output tensor has `dim` retained or not.

   Returns: A new Tensor.

   Example:
   ```clojure
   (min (tensor [1 3 2]))
   ```"
  ([a] (.min (->tensor a)))
  ([a dim & {:keys [keepdim] :or {keepdim false}}]
   (let [res (.min (->tensor a) (clojure.core/long dim) (clojure.core/boolean keepdim))]
     (.get res (clojure.core/long 0)))))

(defn view
  "Returns a new tensor with the same data as the self tensor but of a different shape.
   Strictly zero-copy; throws an error if the tensor is not contiguous.

   Parameters:
   - t: The input Tensor.
   - dims: Sequence of Longs representing the target shape.

   Returns: A new Tensor (view).

   Example:
   ```clojure
   (view (tensor [1 2 3 4]) [2 2])
   ```"
  [t dims]
  (.view (->tensor t) (long-array dims)))

(defn select
  "Returns a new tensor which is a sliced version of the input tensor 
   along the selected dimension at the given index.
   
   Example: (select t 0 1)"
  [t dim index]
  (let [t-obj (->tensor t)
        dim (clojure.core/long dim)
        index (clojure.core/long index)
        dim-size (.size t-obj dim)]
    (when (or (neg? dim) (>= dim (.dim t-obj)))
      (throw (IllegalArgumentException. (str "Dimension index " dim " out of range for tensor with " (.dim t-obj) " dimensions"))))
    (when (or (and (neg? index) (< index (- dim-size))) (>= index dim-size))
      (throw (IllegalArgumentException. (str "Index " index " out of range for dimension " dim " with size " dim-size))))
    (.select t-obj dim index)))

(defn clone
  "Returns a copy of the input tensor.

   Parameters:
   - t: The input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (clone (ones [2 2]))
   ```"
  [t] (.clone (->tensor t)))

(defn permute
  "Returns a view of the original tensor with its dimensions permuted.

   Parameters:
   - t: The input Tensor.
   - dims: The desired ordering of dimensions (sequence of Longs).

   Returns: A new Tensor (view).

   Example:
   ```clojure
   (permute (ones [2 3 4]) [2 0 1])
   ```"
  [t dims] (.permute (->tensor t) (long-array dims)))

(defn reshape
  "Returns a tensor with the same data as input but of a different shape.
   May return a view if possible, otherwise returns a copy.

   Parameters:
   - t: The input Tensor.
   - dims: Sequence of Longs representing the target shape.

   Returns: A new Tensor.

   Example:
   ```clojure
   (reshape (tensor [1 2 3 4]) [2 2])
   ```"
  [t dims]
  (let [t-obj (->tensor t)
        actual (.numel t-obj)
        ;; Replace -1 with inferred size
        dims (if (some #(= -1 %) dims)
               (let [known-prod (reduce * (remove #(= -1 %) dims))]
                 (mapv #(if (= -1 %) (quot actual known-prod) %) dims))
               dims)
        requested (reduce * dims)]
    (when (not= requested actual)
      (throw (IllegalArgumentException. (str "Cannot reshape tensor with " actual " elements to shape " dims " (" requested " requested elements)"))))
    (torch/reshape t-obj (long-array dims))))

(defn unflatten
  "Returns a tensor with the same data as input but with a dimension expanded into multiple dimensions.

   Parameters:
   - t: The input Tensor.
   - dim: The dimension to unflatten (Long).
   - dims: The target shape for the unflattened dimension (sequence of Longs).

   Returns: A new Tensor.

   Example:
   ```clojure
   (unflatten (ones [12]) 0 [3 4])
   ```"
  [t dim dims]
  (.unflatten (->tensor t) (clojure.core/long dim) (long-array dims)))

(defn transpose
  "Returns a tensor that is a transposed version of input. 
   The given dimensions `dim0` and `dim1` are swapped.

   Parameters:
   - t: The input Tensor.
   - dim0: First dimension to swap (Long).
   - dim1: Second dimension to swap (Long).

   Returns: A new Tensor (view).

   Example:
   ```clojure
   (transpose (ones [2 3]) 0 1)
   ```"
  [t dim0 dim1] (.transpose (->tensor t) (clojure.core/long dim0) (clojure.core/long dim1)))

(defn T
  "Returns a tensor that is a transposed version of input (swaps last two dimensions).
   Expects input to be at least 2-dimensional.

   Parameters:
   - t: The input Tensor.

   Returns: A new Tensor (view).

   Example:
   ```clojure
   (T (ones [2 3]))
   ```"
  [t] (transpose t -2 -1))

(defn squeeze
  "Returns a tensor with all specified dimensions of input of size 1 removed.

   Parameters:
   - t: The input Tensor.
   - dim (optional): The dimension to squeeze (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (squeeze (zeros [1 2 1 3]))
   (squeeze (zeros [1 2 1 3]) 0)
   ```"
  ([t] (torch/squeeze (->tensor t)))
  ([t dim] (torch/squeeze (->tensor t) (clojure.core/long dim))))

(defn flatten
  "Flattens input by rescaling it into a one-dimensional tensor. 
   If `start-dim` or `end-dim` are passed, only those dimensions are flattened.

   Parameters:
   - t: The input Tensor.
   - start-dim (optional): The first dimension to flatten (Long, default: 0).
   - end-dim (optional): The last dimension to flatten (Long, default: -1).

   Returns: A new Tensor.

   Example:
   ```clojure
   (flatten (ones [2 3 4]))
   (flatten (ones [2 3 4]) 1)
   ```"
  ([t] (torch/flatten (->tensor t)))
  ([t start-dim] (torch/flatten (->tensor t) (clojure.core/long start-dim) -1))
  ([t start-dim end-dim] (torch/flatten (->tensor t) (clojure.core/long start-dim) (clojure.core/long end-dim))))

;; --- Slicing & Indexing ---

(defn- negative-step-slice?
  "Check if any slice spec has a negative step."
  [indexers]
  (some #(and (vector? %) (let [step (get % 2)] (and step (neg? step)))) indexers))

(defn- convert-negative-step
  "Convert negative step slicing to positive step using flip.
   Returns [transformed-tensor new-indexers]."
  [t-obj indexers]
  (loop [t-obj t-obj
         idx indexers
         dim 0
         new-indexers []]
    (if (empty? idx)
      [t-obj new-indexers]
      (let [spec (first idx)]
        (cond
          (or (= spec :_) (= spec :all))
          (recur t-obj (rest idx) (inc dim) (conj new-indexers spec))

          (vector? spec)
          (let [step (get spec 2)]
            (if (and step (neg? step))
              (let [abs-step (clojure.core/long (- step))
                    start (get spec 0)
                    end (get spec 1)
                    dim-size (.size t-obj dim)
                    original-end end
                    start (cond (nil? start) (dec dim-size)
                                (neg? start) (+ dim-size start)
                                :else start)
                    _end (cond (nil? end) 0
                               (neg? end) (+ dim-size end)
                               :else end)
                    new-start (clojure.core/max 0 (- dim-size 1 start))
                    new-end (if (nil? original-end)
                              dim-size
                              (clojure.core/min dim-size (- dim-size original-end 1)))
                    new-spec [new-start new-end abs-step]]
                (recur (.flip t-obj (long-array [dim]))
                       (rest idx)
                       (inc dim)
                       (conj new-indexers new-spec)))
              (recur t-obj (rest idx) (inc dim) (conj new-indexers spec))))

          :else
          (recur t-obj (rest idx) (inc dim) (conj new-indexers spec)))))))

(defn ix
  "Ergonomic indexer. Mirrors Python's t[0, :, 1:5, ...].
   
   Indexers:
   - 0, -1: Select index.
   - :_ or :all: Select entire dim.
   - ... : Ellipsis.
   - [start end]: Slices dim.
   - [start end step]: Slices with step (including negative for reverse).
   
   Example:
   #_
   (ix t 0 :_ [1 5])
   
   Negative step examples:
   #_
   (ix t [nil nil -1])  ; Reverse entire tensor
   (ix t [5 nil -1])    ; From index 5 to end, reversed
   (ix t [5 0 -1])     ; From 5 down to 1 (exclusive of 0)"
  [t & indexers]
  (let [t-obj (->tensor t)
        t-dim (.dim t-obj)
        idx-vec (TensorIndexVector.)
        ellipsis-count (count (filter #(= % '...) indexers))]
    (when (> ellipsis-count 1)
      (throw (IllegalArgumentException. "Only one ellipsis (...) allowed in indexing")))
    (let [effective-dim (- t-dim (count (remove #(or (= % :_) (= % :all) (= % '...) (instance? Tensor %) (vector? %)) indexers)))]
      (when (neg? effective-dim)
        (throw (IllegalArgumentException. (str "Too many indexers: " (count indexers) " for tensor with " t-dim " dimensions")))))
    (let [[t-final indexers-final]
          (if (negative-step-slice? indexers)
            (let [[t-converted new-indexers] (convert-negative-step t-obj indexers)]
              [t-converted new-indexers])
            [t-obj indexers])]
      (doseq [spec indexers-final]
        (.push_back idx-vec
                    (cond
                      (nil? spec)
                      (TensorIndex. (Slice.))

                      (integer? spec) (TensorIndex. (clojure.core/long spec))
                      (or (= spec :_) (= spec :all)) (TensorIndex. (Slice.))
                      (= spec '...) (TensorIndex. (torch/Ellipsis))
                      (instance? Tensor spec) (TensorIndex. ^Tensor spec)
                      (vector? spec)
                      (let [start (get spec 0)
                            end (get spec 1)
                            step (get spec 2)]
                        (when (and step (zero? step))
                          (throw (IllegalArgumentException. "slice step cannot be zero")))
                        (let [->sio (fn [v] (if (nil? v) (SymIntOptional.) (SymIntOptional. (SymInt. (clojure.core/long v)))))]
                          (TensorIndex. (Slice. (->sio start) (->sio end) (->sio step)))))
                      :else (throw (IllegalArgumentException. (str "Invalid indexer: " spec))))))
      (.index t-final idx-vec))))

(defn precompute-rope-freqs [dim seq-len & {:keys [theta] :or {theta 10000.0}}]
  (let [freqs (div (arange 0 dim 2) (clojure.core/double dim))
        inv-freq (pow (clojure.core/float theta) (mul freqs -1.0))
        t (arange seq-len)
        freqs-matrix (outer t inv-freq)
        emb (cat [freqs-matrix freqs-matrix] -1)]
    [(cos emb) (sin emb)]))

(defn apply-rope [x cos-emb sin-emb]
  (let [[_B _T _H D] (size x)
        cos-emb (reshape cos-emb [1 _T 1 D])
        sin-emb (reshape sin-emb [1 _T 1 D])
        half-d (quot D 2)
        x1 (ix x :_ :_ :_ [0 half-d])
        x2 (ix x :_ :_ :_ [half-d D])
        ;; standard rope rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        neg-x2 (mul x2 -1.0)
        rotated-x (cat [neg-x2 x1] -1)]
    (add (mul x cos-emb) (mul rotated-x sin-emb))))

;; --- Extended Math ---

(defn polygamma
  "Computes the n-th derivative of the digamma function on `input`.

   Parameters:
   - n: The order of the polygamma function (Long).
   - input: The input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (polygamma 1 (tensor [1.0 2.0]))
   ```"
  [n input]
  (torch/polygamma (clojure.core/long n) (->tensor input)))

(defn bitwise-and [a b]
  (cond
    (and (number? a) (number? b)) (throw (IllegalArgumentException. "At least one argument must be a tensor"))
    (number? a) (torch/bitwise_and (org.bytedeco.pytorch.Scalar. (clojure.core/long a)) (->tensor b))
    (number? b) (torch/bitwise_and (->tensor a) (org.bytedeco.pytorch.Scalar. (clojure.core/long b)))
    :else (torch/bitwise_and (->tensor a) (->tensor b))))

(defn bitwise-or [a b]
  (cond
    (and (number? a) (number? b)) (throw (IllegalArgumentException. "At least one argument must be a tensor"))
    (number? a) (torch/bitwise_or (org.bytedeco.pytorch.Scalar. (clojure.core/long a)) (->tensor b))
    (number? b) (torch/bitwise_or (->tensor a) (org.bytedeco.pytorch.Scalar. (clojure.core/long b)))
    :else (torch/bitwise_or (->tensor a) (->tensor b))))

(defn bitwise-xor [a b]
  (cond
    (and (number? a) (number? b)) (throw (IllegalArgumentException. "At least one argument must be a tensor"))
    (number? a) (torch/bitwise_xor (org.bytedeco.pytorch.Scalar. (clojure.core/long a)) (->tensor b))
    (number? b) (torch/bitwise_xor (->tensor a) (org.bytedeco.pytorch.Scalar. (clojure.core/long b)))
    :else (torch/bitwise_xor (->tensor a) (->tensor b))))

(defn bitwise-left-shift [a b]
  (cond
    (and (number? a) (number? b)) (throw (IllegalArgumentException. "At least one argument must be a tensor"))
    (number? a) (torch/bitwise_left_shift (org.bytedeco.pytorch.Scalar. (clojure.core/long a)) (->tensor b))
    (number? b) (torch/bitwise_left_shift (->tensor a) (org.bytedeco.pytorch.Scalar. (clojure.core/long b)))
    :else (torch/bitwise_left_shift (->tensor a) (->tensor b))))

(defn bitwise-right-shift [a b]
  (cond
    (and (number? a) (number? b)) (throw (IllegalArgumentException. "At least one argument must be a tensor"))
    (number? a) (torch/bitwise_right_shift (org.bytedeco.pytorch.Scalar. (clojure.core/long a)) (->tensor b))
    (number? b) (torch/bitwise_right_shift (->tensor a) (org.bytedeco.pytorch.Scalar. (clojure.core/long b)))
    :else (torch/bitwise_right_shift (->tensor a) (->tensor b))))

(defn bitwise-not [a] (torch/bitwise_not (->tensor a)))

(defn floor [t] (torch/floor (->tensor t)))
(defn ceil [t] (torch/ceil (->tensor t)))
(defn round
  ([t] (torch/round (->tensor t)))
  ([t decimals] (torch/round (->tensor t) (clojure.core/long decimals))))
(defn trunc [t] (torch/trunc (->tensor t)))
(defn frac [t] (torch/frac (->tensor t)))

(defn tan [t] (torch/tan (->tensor t)))
(defn asin [t] (torch/asin (->tensor t)))
(defn acos [t] (torch/acos (->tensor t)))
(defn atan [t] (torch/atan (->tensor t)))
(defn atan2 [y x] (torch/atan2 (->tensor y) (->tensor x)))

(defn sinh [t] (torch/sinh (->tensor t)))
(defn cosh [t] (torch/cosh (->tensor t)))
(defn tanh [t] (torch/tanh (->tensor t)))
(defn asinh [t] (torch/asinh (->tensor t)))
(defn acosh [t] (torch/acosh (->tensor t)))
(defn atanh [t] (torch/atanh (->tensor t)))

(defn erf [t] (torch/erf (->tensor t)))
(defn erfc [t] (torch/erfc (->tensor t)))
(defn erfinv [t] (torch/erfinv (->tensor t)))
(defn digamma [t] (torch/digamma (->tensor t)))
(defn lgamma [t] (torch/lgamma (->tensor t)))

(defn rad2deg
  "Returns a new tensor with each element of `t` converted from radians to degrees.

   Parameters:
   - t: The input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (rad2deg (tensor [3.14159]))
   ```"
  [t] (torch/rad2deg (->tensor t)))

(defn deg2rad
  "Returns a new tensor with each element of `t` converted from degrees to radians.

   Parameters:
   - t: The input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (deg2rad (tensor [180.0]))
   ```"
  [t] (torch/deg2rad (->tensor t)))
(defn reciprocal [t] (torch/reciprocal (->tensor t)))
(defn square [t] (torch/square (->tensor t)))

(defn remainder [a b] (torch/remainder (->tensor a) (->tensor b)))
(defn fmod [a b] (torch/fmod (->tensor a) (->tensor b)))

(defn isclose
  ([a b] (torch/isclose (->tensor a) (->tensor b)))
  ([a b {:keys [rtol atol equal-nan]
         :or {rtol 1e-5 atol 1e-8 equal-nan false}}]
   (torch/isclose (->tensor a) (->tensor b)
                  (clojure.core/double rtol)
                  (clojure.core/double atol)
                  (clojure.core/boolean equal-nan))))

(defn allclose
  ([a b] (torch/allclose (->tensor a) (->tensor b)))
  ([a b {:keys [rtol atol equal-nan]
         :or {rtol 1e-5 atol 1e-8 equal-nan false}}]
   (torch/allclose (->tensor a) (->tensor b)
                   (clojure.core/double rtol)
                   (clojure.core/double atol)
                   (clojure.core/boolean equal-nan))))

;; --- Extended Tensor Manipulation ---

(defn flip [t dims]
  (torch/flip (->tensor t) (long-array (if (number? dims) [dims] dims))))

(defn roll
  ([t shifts]
   (torch/roll (->tensor t) (long-array (if (number? shifts) [shifts] shifts))))
  ([t shifts dims]
   (torch/roll (->tensor t)
               (long-array (if (number? shifts) [shifts] shifts))
               (long-array (if (number? dims) [dims] dims)))))

(defn rot90
  ([t] (torch/rot90 (->tensor t)))
  ([t k dims]
   (torch/rot90 (->tensor t)
                (clojure.core/long k)
                (long-array (if (number? dims) [dims] dims)))))

(defn diag
  ([t] (torch/diag (->tensor t)))
  ([t diagonal] (torch/diag (->tensor t) (clojure.core/long diagonal))))

(defn trace
  "Returns the sum of the elements of the diagonal of the input 2-D matrix.

   Parameters:
   - t: The input 2-D Tensor.

   Returns: A Scalar Tensor.

   Example:
   ```clojure
   (trace (eye 3))
   ```"
  [t] (torch/trace (->tensor t)))

(defn dot
  "Computes the dot product of two 1D tensors.

   Parameters:
   - a: First 1D Tensor.
   - b: Second 1D Tensor.

   Returns: A Scalar Tensor.

   Example:
   ```clojure
   (dot (tensor [1 2]) (tensor [3 4]))
   ```"
  [a b] (torch/dot (->tensor a) (->tensor b)))

(defn inner
  "Computes the inner product of two tensors.

   Parameters:
   - a: First Tensor.
   - b: Second Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (inner (tensor [1 2]) (tensor [3 4]))
   ```"
  [a b] (torch/inner (->tensor a) (->tensor b)))

(defn cross
  ([a b] (torch/cross (->tensor a) (->tensor b)))
  ([a b dim] (torch/cross (->tensor a) (->tensor b) (LongOptional. (clojure.core/long dim)))))

(defn meshgrid
  ([tensors]
   (->vector (torch/meshgrid (->tensor-vector tensors))))
  ([tensors indexing]
   (->vector (torch/meshgrid (->tensor-vector tensors) (clojure.core/str indexing)))))

(defn broadcast-to [t shape]
  (torch/broadcast_to (->tensor t) (long-array shape)))

(defn broadcast-tensors [tensors]
  (->vector (torch/broadcast_tensors (->tensor-vector tensors))))

(defn logsumexp
  "Returns the log of summed exponentials of each row of the input tensor in the given dimension `dim`.

   Parameters:
   - t: The input Tensor.
   - dim: The dimension or sequence of dimensions to reduce.
   - keepdim (optional): Whether the output tensor has `dim` retained or not (Boolean).

   Returns: A new Tensor.

   Example:
   ```clojure
   (logsumexp (tensor [1.0 2.0]) 0)
   ```"
  ([t dim]
   (torch/logsumexp (->tensor t)
                    (long-array (if (number? dim) [dim] dim))))
  ([t dim keepdim]
   (torch/logsumexp (->tensor t)
                    (long-array (if (number? dim) [dim] dim))
                    (clojure.core/boolean keepdim))))

(defn clip
  ([t min-val max-val]
   (torch/clip (->tensor t) (clojure.core/double min-val) (clojure.core/double max-val))))

(defn count-nonzero
  "Counts the number of non-zero values in the tensor `t`.

   Parameters:
   - t: The input Tensor.
   - dim (optional): The dimension or sequence of dimensions along which to count.

   Returns: A new Tensor (Long).

   Example:
   ```clojure
   (count-nonzero (tensor [1 0 2]))
   ```"
  ([t]
   (torch/count_nonzero (->tensor t)))
  ([t dim]
   (torch/count_nonzero (->tensor t)
                        (long-array (if (number? dim) [dim] dim)))))

(defn isnan [t] (torch/isnan (->tensor t)))
(defn isinf [t] (torch/isinf (->tensor t)))
(defn isfinite [t] (torch/isfinite (->tensor t)))
(defn isreal [t] (torch/isreal (->tensor t)))
(defn isposinf [t] (torch/isposinf (->tensor t)))
(defn isneginf [t] (torch/isneginf (->tensor t)))
(defn signbit [t] (torch/signbit (->tensor t)))

(defn amax
  ([t]
   (torch/amax (->tensor t)))
  ([t dim]
   (torch/amax (->tensor t)
               (long-array (if (number? dim) [dim] dim))
               false))
  ([t dim keepdim]
   (torch/amax (->tensor t)
               (long-array (if (number? dim) [dim] dim))
               (clojure.core/boolean keepdim))))

(defn amin
  ([t]
   (torch/amin (->tensor t)))
  ([t dim]
   (torch/amin (->tensor t)
               (long-array (if (number? dim) [dim] dim))
               false))
  ([t dim keepdim]
   (torch/amin (->tensor t)
               (long-array (if (number? dim) [dim] dim))
               (clojure.core/boolean keepdim))))

(defn maximum [a b] (torch/maximum (->tensor a) (->tensor b)))
(defn minimum [a b] (torch/minimum (->tensor a) (->tensor b)))

(defn fmax [a b] (torch/fmax (->tensor a) (->tensor b)))
(defn fmin [a b] (torch/fmin (->tensor a) (->tensor b)))

(defn argsort
  ([t]
   (torch/argsort (->tensor t)))
  ([t dim]
   (torch/argsort (->tensor t) (clojure.core/long dim)))
  ([t dim descending]
   (torch/argsort (->tensor t) (clojure.core/long dim) (clojure.core/boolean descending)))
  ([t dim descending stable]
   (torch/argsort (->tensor t)
                  (clojure.core/long dim)
                  (clojure.core/boolean descending)
                  (clojure.core/boolean stable))))

(defn argwhere [t] (torch/argwhere (->tensor t)))
(defn kron [a b] (torch/kron (->tensor a) (->tensor b)))

(defn tensordot
  ([a b dims]
   (let [a-t (->tensor a)
         b-t (->tensor b)]
     (if (number? dims)
       (let [n (clojure.core/long dims)
             a-rank (.dim a-t)
             dims-a (range (- a-rank n) a-rank)
             dims-b (range n)]
         (torch/tensordot a-t b-t (long-array dims-a) (long-array dims-b)))
       (let [[dims-a dims-b] dims]
         (torch/tensordot a-t b-t (long-array dims-a) (long-array dims-b)))))))

(defn vdot
  "Computes the dot product of two 1D tensors. 
   If the tensors are complex, the first tensor is conjugated.

   Parameters:
   - a: First 1D Tensor.
   - b: Second 1D Tensor.

   Returns: A Scalar Tensor.

   Example:
   ```clojure
   (vdot (tensor [1 2]) (tensor [3 4]))
   ```"
  [a b] (torch/vdot (->tensor a) (->tensor b)))

(defn tile [t reps]
  (torch/tile (->tensor t) (long-array reps)))

(defn hstack [tensors]
  (torch/hstack (->tensor-vector tensors)))

(defn vstack [tensors]
  (torch/vstack (->tensor-vector tensors)))

(defn dstack [tensors]
  (torch/dstack (->tensor-vector tensors)))

(defn column-stack [tensors]
  (torch/column_stack (->tensor-vector tensors)))

(defn row-stack [tensors]
  (torch/row_stack (->tensor-vector tensors)))

(defn movedim [t source destination]
  (torch/movedim (->tensor t)
                 (long-array (if (number? source) [source] source))
                 (long-array (if (number? destination) [destination] destination))))

(defn swapaxes [t dim0 dim1]
  (torch/swapaxes (->tensor t) (clojure.core/long dim0) (clojure.core/long dim1)))

(defn take-along-dim
  ([input indices]
   (torch/take_along_dim (->tensor input) (->tensor indices)))
  ([input indices dim]
   (torch/take_along_dim (->tensor input)
                         (->tensor indices)
                         (LongOptional. (clojure.core/long dim)))))

(defn diagonal
  ([t]
   (torch/diagonal (->tensor t)))
  ([t offset dim1 dim2]
   (torch/diagonal (->tensor t)
                   (clojure.core/long offset)
                   (clojure.core/long dim1)
                   (clojure.core/long dim2))))

(defn diagflat
  ([t]
   (torch/diagflat (->tensor t)))
  ([t offset]
   (torch/diagflat (->tensor t) (clojure.core/long offset))))

(defn matrix-exp [t] (torch/matrix_exp (->tensor t)))

(defn cholesky-inverse
  ([chol]
   (torch/cholesky_inverse (->tensor chol)))
  ([chol upper]
   (torch/cholesky_inverse (->tensor chol) (clojure.core/boolean upper))))

(defn cholesky-solve
  ([b chol]
   (torch/cholesky_solve (->tensor b) (->tensor chol)))
  ([b chol upper]
   (torch/cholesky_solve (->tensor b)
                         (->tensor chol)
                         (clojure.core/boolean upper))))

(defn- ->long-optional [n]
  (if (some? n)
    (LongOptional. (clojure.core/long n))
    (LongOptional.)))

(defn- ->string-view-optional [s]
  (if (some? s)
    (StringViewOptional. (clojure.core/str s))
    (StringViewOptional.)))

(defn fft
  ([t]
   (torch/fft_fft (->tensor t)))
  ([t n]
   (torch/fft_fft (->tensor t)
                  (->long-optional n)
                  -1
                  (->string-view-optional nil)))
  ([t n dim]
   (torch/fft_fft (->tensor t)
                  (->long-optional n)
                  (clojure.core/long dim)
                  (->string-view-optional nil)))
  ([t n dim norm]
   (torch/fft_fft (->tensor t)
                  (->long-optional n)
                  (clojure.core/long dim)
                  (->string-view-optional norm))))

(defn ifft
  ([t]
   (torch/fft_ifft (->tensor t)))
  ([t n]
   (torch/fft_ifft (->tensor t)
                   (->long-optional n)
                   -1
                   (->string-view-optional nil)))
  ([t n dim]
   (torch/fft_ifft (->tensor t)
                   (->long-optional n)
                   (clojure.core/long dim)
                   (->string-view-optional nil)))
  ([t n dim norm]
   (torch/fft_ifft (->tensor t)
                   (->long-optional n)
                   (clojure.core/long dim)
                   (->string-view-optional norm))))

(defn rfft
  ([t]
   (torch/fft_rfft (->tensor t)))
  ([t n]
   (torch/fft_rfft (->tensor t)
                   (->long-optional n)
                   -1
                   (->string-view-optional nil)))
  ([t n dim]
   (torch/fft_rfft (->tensor t)
                   (->long-optional n)
                   (clojure.core/long dim)
                   (->string-view-optional nil)))
  ([t n dim norm]
   (torch/fft_rfft (->tensor t)
                   (->long-optional n)
                   (clojure.core/long dim)
                   (->string-view-optional norm))))

(defn irfft
  ([t]
   (torch/fft_irfft (->tensor t)))
  ([t n]
   (torch/fft_irfft (->tensor t)
                    (->long-optional n)
                    -1
                    (->string-view-optional nil)))
  ([t n dim]
   (torch/fft_irfft (->tensor t)
                    (->long-optional n)
                    (clojure.core/long dim)
                    (->string-view-optional nil)))
  ([t n dim norm]
   (torch/fft_irfft (->tensor t)
                    (->long-optional n)
                    (clojure.core/long dim)
                    (->string-view-optional norm))))

(defn fft2
  ([t]
   (torch/fft_fft2 (->tensor t)))
  ([t s]
   (torch/fft_fft2 (->tensor t)
                   (long-array s)
                   (long-array [-2 -1])
                   (->string-view-optional nil)))
  ([t s dim]
   (torch/fft_fft2 (->tensor t)
                   (long-array s)
                   (long-array (if (number? dim) [dim] dim))
                   (->string-view-optional nil)))
  ([t s dim norm]
   (torch/fft_fft2 (->tensor t)
                   (long-array s)
                   (long-array (if (number? dim) [dim] dim))
                   (->string-view-optional norm))))

(defn ifft2
  ([t]
   (torch/fft_ifft2 (->tensor t)))
  ([t s]
   (torch/fft_ifft2 (->tensor t)
                    (long-array s)
                    (long-array [-2 -1])
                    (->string-view-optional nil)))
  ([t s dim]
   (torch/fft_ifft2 (->tensor t)
                    (long-array s)
                    (long-array (if (number? dim) [dim] dim))
                    (->string-view-optional nil)))
  ([t s dim norm]
   (torch/fft_ifft2 (->tensor t)
                    (long-array s)
                    (long-array (if (number? dim) [dim] dim))
                    (->string-view-optional norm))))

(defn rfft2
  ([t]
   (torch/fft_rfft2 (->tensor t)))
  ([t s]
   (torch/fft_rfft2 (->tensor t)
                    (long-array s)
                    (long-array [-2 -1])
                    (->string-view-optional nil)))
  ([t s dim]
   (torch/fft_rfft2 (->tensor t)
                    (long-array s)
                    (long-array (if (number? dim) [dim] dim))
                    (->string-view-optional nil)))
  ([t s dim norm]
   (torch/fft_rfft2 (->tensor t)
                    (long-array s)
                    (long-array (if (number? dim) [dim] dim))
                    (->string-view-optional norm))))

(defn irfft2
  ([t]
   (torch/fft_irfft2 (->tensor t)))
  ([t s]
   (torch/fft_irfft2 (->tensor t)
                     (long-array s)
                     (long-array [-2 -1])
                     (->string-view-optional nil)))
  ([t s dim]
   (torch/fft_irfft2 (->tensor t)
                     (long-array s)
                     (long-array (if (number? dim) [dim] dim))
                     (->string-view-optional nil)))
  ([t s dim norm]
   (torch/fft_irfft2 (->tensor t)
                     (long-array s)
                     (long-array (if (number? dim) [dim] dim))
                     (->string-view-optional norm))))

(defn fftn
  ([t]
   (torch/fft_fftn (->tensor t)))
  ([t s]
   (torch/fft_fftn (->tensor t)
                   (long-array s)
                   (long-array (range (- (count s)) 0))
                   (->string-view-optional nil)))
  ([t s dim]
   (torch/fft_fftn (->tensor t)
                   (long-array s)
                   (long-array (if (number? dim) [dim] dim))
                   (->string-view-optional nil)))
  ([t s dim norm]
   (torch/fft_fftn (->tensor t)
                   (long-array s)
                   (long-array (if (number? dim) [dim] dim))
                   (->string-view-optional norm))))

(defn ifftn
  ([t]
   (torch/fft_ifftn (->tensor t)))
  ([t s]
   (torch/fft_ifftn (->tensor t)
                    (long-array s)
                    (long-array (range (- (count s)) 0))
                    (->string-view-optional nil)))
  ([t s dim]
   (torch/fft_ifftn (->tensor t)
                    (long-array s)
                    (long-array (if (number? dim) [dim] dim))
                    (->string-view-optional nil)))
  ([t s dim norm]
   (torch/fft_ifftn (->tensor t)
                    (long-array s)
                    (long-array (if (number? dim) [dim] dim))
                    (->string-view-optional norm))))

(defn rfftn
  ([t]
   (torch/fft_rfftn (->tensor t)))
  ([t s]
   (torch/fft_rfftn (->tensor t)
                    (long-array s)
                    (long-array (range (- (count s)) 0))
                    (->string-view-optional nil)))
  ([t s dim]
   (torch/fft_rfftn (->tensor t)
                    (long-array s)
                    (long-array (if (number? dim) [dim] dim))
                    (->string-view-optional nil)))
  ([t s dim norm]
   (torch/fft_rfftn (->tensor t)
                    (long-array s)
                    (long-array (if (number? dim) [dim] dim))
                    (->string-view-optional norm))))

(defn irfftn
  ([t]
   (torch/fft_irfftn (->tensor t)))
  ([t s]
   (torch/fft_irfftn (->tensor t)
                     (long-array s)
                     (long-array (range (- (count s)) 0))
                     (->string-view-optional nil)))
  ([t s dim]
   (torch/fft_irfftn (->tensor t)
                     (long-array s)
                     (long-array (if (number? dim) [dim] dim))
                     (->string-view-optional nil)))
  ([t s dim norm]
   (torch/fft_irfftn (->tensor t)
                     (long-array s)
                     (long-array (if (number? dim) [dim] dim))
                     (->string-view-optional norm))))

(defn fftshift
  ([t]
   (torch/fft_fftshift (->tensor t)))
  ([t dim]
   (torch/fft_fftshift (->tensor t)
                       (long-array (if (number? dim) [dim] dim)))))

(defn ifftshift
  ([t]
   (torch/fft_ifftshift (->tensor t)))
  ([t dim]
   (torch/fft_ifftshift (->tensor t)
                        (long-array (if (number? dim) [dim] dim)))))

(defn fftfreq
  ([n]
   (torch/fft_fftfreq (clojure.core/long n))))

(defn rfftfreq
  ([n]
   (torch/fft_rfftfreq (clojure.core/long n))))

(defn diff
  ([t]
   (torch/diff (->tensor t))))

(defn gradient
  ([t]
   (->vector (torch/gradient (->tensor t))))
  ([t spacing]
   (->vector (torch/gradient (->tensor t)
                             (->tensor-vector spacing))))
  ([t spacing dim]
   (->vector (torch/gradient (->tensor t)
                             (->tensor-vector spacing)
                             (long-array (if (number? dim) [dim] dim))))))

(defn histc
  ([t]
   (torch/histc (->tensor t)))
  ([t bins min-val max-val]
   (torch/histc (->tensor t)
                (clojure.core/long bins)
                (clojure.core/double min-val)
                (clojure.core/double max-val))))

(defn nan-to-num
  ([t]
   (torch/nan_to_num (->tensor t)))
  ([t {:keys [nan posinf neginf]}]
   (let [->opt (fn [x]
                 (if (some? x)
                   (DoubleOptional. (clojure.core/double x))
                   (DoubleOptional.)))]
     (torch/nan_to_num (->tensor t)
                       (->opt nan)
                       (->opt posinf)
                       (->opt neginf)))))

(defn i0 [t] (torch/i0 (->tensor t)))
(defn i1 [t] (torch/i1 (->tensor t)))
(defn sinc [t] (torch/sinc (->tensor t)))
(defn xlogy [x y] (torch/xlogy (->tensor x) (->tensor y)))
(defn xlog1py [x y] (torch/xlog1py (->tensor x) (->tensor y)))
(defn zeta [x q] (torch/zeta (->tensor x) (->tensor q)))

;; --- Extended Linalg ---

(defn linalg-norm
  "Computes a vector or matrix norm.

   Parameters:
   - t: The input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-norm (ones [3]))
   ```"
  [t] (torch/linalg_norm (->tensor t)))

(defn linalg-vector-norm
  "Computes a vector norm.

   Parameters:
   - t: The input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-vector-norm (ones [3]))
   ```"
  [t] (torch/linalg_vector_norm (->tensor t)))

(defn linalg-matrix-norm
  "Computes a matrix norm.

   Parameters:
   - t: The input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-matrix-norm (ones [3 3]))
   ```"
  [t] (torch/linalg_matrix_norm (->tensor t)))

(defn linalg-inv
  "Computes the inverse of a square matrix or a batch of square matrices.

   Parameters:
   - t: The input Tensor of shape (*, n, n).

   Returns: A new Tensor (inverse).

   Example:
   ```clojure
   (linalg-inv (eye 3))
   ```"
  [t] (torch/linalg_inv (->tensor t)))

(defn linalg-det
  "Computes the determinant of a square matrix or a batch of square matrices.

   Parameters:
   - t: The input Tensor of shape (*, n, n).

   Returns: A new Tensor (determinant).

   Example:
   ```clojure
   (linalg-det (eye 3))
   ```"
  [t] (torch/linalg_det (->tensor t)))

(defn- tuple2->vector [pair]
  [(.get0 pair) (.get1 pair)])

(defn- tuple3->vector [triple]
  [(.get0 triple) (.get1 triple) (.get2 triple)])

(defn- tuple4->map [quad]
  {:solution (.get0 quad)
   :residuals (.get1 quad)
   :rank (.get2 quad)
   :singular-values (.get3 quad)})

(defn- ->double-optional [x]
  (if (some? x)
    (DoubleOptional. (clojure.core/double x))
    (DoubleOptional.)))

(defn- ->driver-string [driver]
  (cond
    (keyword? driver) (name driver)
    (string? driver) driver
    :else (throw (IllegalArgumentException.
                  (str "Unsupported lstsq/svd driver type: " (type driver))))))

(defn- ->driver-optional [driver]
  (if (some? driver)
    (StringViewOptional. (->driver-string driver))
    (StringViewOptional.)))

(defn- order->cond-arg [ord]
  (cond
    (number? ord) (ScalarOptional. (Scalar. (clojure.core/double ord)))
    (keyword? ord) (str (name ord))
    (string? ord) ord
    :else (throw (IllegalArgumentException.
                  (str "Unsupported cond order type: " (type ord))))))

(defn histogram
  ([t]
   (tuple2->vector (torch/histogram (->tensor t))))
  ([t bins]
   (tuple2->vector (torch/histogram (->tensor t) (->tensor bins)))))

(defn quantile
  ([t q]
   (torch/quantile (->tensor t) (->tensor q)))
  ([t q dim]
   (torch/quantile (->tensor t)
                   (->tensor q)
                   (LongOptional. (clojure.core/long dim))
                   false
                   "linear"))
  ([t q dim keepdim interpolation]
   (torch/quantile (->tensor t)
                   (->tensor q)
                   (LongOptional. (clojure.core/long dim))
                   (clojure.core/boolean keepdim)
                   (clojure.core/str interpolation))))

(defn nanquantile
  ([t q]
   (torch/nanquantile (->tensor t) (->tensor q)))
  ([t q dim]
   (torch/nanquantile (->tensor t)
                      (->tensor q)
                      (LongOptional. (clojure.core/long dim))
                      false
                      "linear"))
  ([t q dim keepdim interpolation]
   (torch/nanquantile (->tensor t)
                      (->tensor q)
                      (LongOptional. (clojure.core/long dim))
                      (clojure.core/boolean keepdim)
                      (clojure.core/str interpolation))))

(defn cdist
  ([x1 x2]
   (torch/cdist (->tensor x1) (->tensor x2))))

(defn pdist
  ([x]
   (torch/pdist (->tensor x)))
  ([x p]
   (torch/pdist (->tensor x) (clojure.core/double p))))

(defn searchsorted
  ([sorted-seq values]
   (torch/searchsorted (->tensor sorted-seq) (->tensor values))))

(defn bucketize
  ([input boundaries]
   (torch/bucketize (->tensor input) (->tensor boundaries))))

(defn bincount
  ([input]
   (torch/bincount (->tensor input))))

(defn scatter-reduce
  ([input dim index src reduce-op]
   (torch/scatter_reduce (->tensor input)
                         (clojure.core/long dim)
                         (->tensor index)
                         (->tensor src)
                         (clojure.core/str reduce-op)))
  ([input dim index src reduce-op include-self]
   (torch/scatter_reduce (->tensor input)
                         (clojure.core/long dim)
                         (->tensor index)
                         (->tensor src)
                         (clojure.core/str reduce-op)
                         (clojure.core/boolean include-self))))

(defn index-reduce
  ([input dim index source reduce-op]
   (torch/index_reduce (->tensor input)
                       (clojure.core/long dim)
                       (->tensor index)
                       (->tensor source)
                       (clojure.core/str reduce-op)))
  ([input dim index source reduce-op include-self]
   (torch/index_reduce (->tensor input)
                       (clojure.core/long dim)
                       (->tensor index)
                       (->tensor source)
                       (clojure.core/str reduce-op)
                       (clojure.core/boolean include-self))))

(defn linalg-slogdet
  "Computes the sign and natural logarithm of the absolute value of the 
   determinant of a square matrix or a batch of square matrices.

   Parameters:
   - t: The input Tensor.

   Returns: A vector [sign log-abs-det] (Tensors).

   Example:
   ```clojure
   (linalg-slogdet (eye 3))
   ```"
  [t]
  (tuple2->vector (torch/linalg_slogdet (->tensor t))))

(defn linalg-qr
  "Computes the QR decomposition of a matrix or a batch of matrices.

   Parameters:
   - t: The input Tensor of shape (*, m, n).
   - mode (optional): Controls the form of the returned decomposition (default: \"reduced\").

   Returns: A vector [Q R] (Tensors).

   Example:
   ```clojure
   (linalg-qr (ones [3 3]))
   ```"
  ([t] (tuple2->vector (torch/linalg_qr (->tensor t))))
  ([t mode] (tuple2->vector (torch/linalg_qr (->tensor t) (clojure.core/str mode)))))

(defn linalg-cholesky
  "Computes the Cholesky decomposition of a complex Hermitian or real symmetric 
   positive-definite matrix or a batch of such matrices.

   Parameters:
   - t: The input Tensor of shape (*, n, n).
   - upper? (optional): Whether to return the upper triangular matrix (default: false).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-cholesky (eye 3))
   ```"
  ([t] (torch/linalg_cholesky (->tensor t)))
  ([t upper?] (torch/linalg_cholesky (->tensor t) (clojure.core/boolean upper?))))

(defn linalg-solve
  "Computes the solution of a square system of linear equations with a unique solution.

   Parameters:
   - a: The input Tensor of shape (*, n, n).
   - b: The input Tensor of shape (*, n, k).
   - left? (optional): Whether to solve AX=B (true, default) or XA=B (false).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-solve (eye 3) (ones [3 1]))
   ```"
  ([a b] (torch/linalg_solve (->tensor a) (->tensor b)))
  ([a b left?] (torch/linalg_solve (->tensor a) (->tensor b) (clojure.core/boolean left?))))

(defn linalg-pinv
  "Computes the Moore-Penrose pseudo-inverse of a matrix or a batch of matrices.

   Parameters:
   - t: The input Tensor.
   - rcond (optional): Relative tolerance for singular values.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-pinv (ones [3 2]))
   ```"
  ([t] (torch/linalg_pinv (->tensor t)))
  ([t rcond] (torch/linalg_pinv (->tensor t) (clojure.core/double rcond))))

(defn linalg-matrix-power
  "Computes the n-th power of a square matrix or a batch of square matrices.

   Parameters:
   - t: The input Tensor.
   - n: The integer exponent (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-matrix-power (eye 3) 2)
   ```"
  [t n]
  (torch/linalg_matrix_power (->tensor t) (clojure.core/long n)))

(defn linalg-svd
  "Computes the singular value decomposition (SVD) of a matrix or a batch of matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map of options:
     - :full-matrices (Boolean): Whether to compute the full-sized decomposition (default: true).
     - :driver (Keyword/String): The name of the LAPACK driver to use.

   Returns: A vector [U S Vh] (Tensors).

   Example:
   ```clojure
   (linalg-svd (ones [3 2]))
   ```"
  ([a]
   (tuple3->vector (torch/linalg_svd (->tensor a))))
  ([a {:keys [full-matrices driver]
       :or {full-matrices true}}]
   (tuple3->vector
    (torch/linalg_svd (->tensor a)
                      (clojure.core/boolean full-matrices)
                      (->driver-optional driver)))))

(defn linalg-svdvals
  "Computes the singular values of a matrix or a batch of matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map of options:
     - :driver (Keyword/String): The name of the LAPACK driver to use.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-svdvals (ones [3 2]))
   ```"
  ([a] (torch/linalg_svdvals (->tensor a)))
  ([a {:keys [driver]}]
   (torch/linalg_svdvals (->tensor a) (->driver-optional driver))))

(defn linalg-eig
  "Computes the eigenvalue decomposition of a square matrix or a batch of square matrices.

   Parameters:
   - a: The input Tensor.

   Returns: A vector [eigenvalues eigenvectors] (Tensors).

   Example:
   ```clojure
   (linalg-eig (eye 3))
   ```"
  [a]
  (tuple2->vector (torch/linalg_eig (->tensor a))))

(defn linalg-eigvals
  "Computes the eigenvalues of a square matrix or a batch of square matrices.

   Parameters:
   - a: The input Tensor.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-eigvals (eye 3))
   ```"
  [a]
  (torch/linalg_eigvals (->tensor a)))

(defn linalg-eigh
  "Computes the eigenvalue decomposition of a complex Hermitian or real symmetric 
   matrix or a batch of such matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map of options:
     - :uplo (Keyword/String): Whether to use the upper ('U') or lower ('L', default) triangle.

   Returns: A vector [eigenvalues eigenvectors] (Tensors).

   Example:
   ```clojure
   (linalg-eigh (eye 3))
   ```"
  ([a] (tuple2->vector (torch/linalg_eigh (->tensor a))))
  ([a {:keys [uplo]}]
   (tuple2->vector (torch/linalg_eigh (->tensor a)
                                      (if uplo (name uplo) "L")))))

(defn linalg-eigvalsh
  "Computes the eigenvalues of a complex Hermitian or real symmetric 
   matrix or a batch of such matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map of options:
     - :uplo (Keyword/String): Whether to use the upper ('U') or lower ('L', default) triangle.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-eigvalsh (eye 3))
   ```"
  ([a] (torch/linalg_eigvalsh (->tensor a)))
  ([a {:keys [uplo]}]
   (torch/linalg_eigvalsh (->tensor a)
                          (if uplo (name uplo) "L"))))

(defn linalg-lu
  "Computes the LU decomposition with partial pivoting of a matrix or a batch of matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map of options:
     - :pivot (Boolean): Whether to compute the LU decomposition with pivoting (default: true).

   Returns: A vector [P L U] (Tensors).

   Example:
   ```clojure
   (linalg-lu (ones [3 3]))
   ```"
  ([a]
   (tuple3->vector (torch/linalg_lu (->tensor a))))
  ([a {:keys [pivot] :or {pivot true}}]
   (tuple3->vector (torch/linalg_lu (->tensor a) (clojure.core/boolean pivot)))))

(defn linalg-lu-factor
  "Computes the LU factorization of a matrix or a batch of matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map of options:
     - :pivot (Boolean): Whether to compute the LU decomposition with pivoting (default: true).

   Returns: A vector [LU pivots] (Tensors).

   Example:
   ```clojure
   (linalg-lu-factor (ones [3 3]))
   ```"
  ([a]
   (tuple2->vector (torch/linalg_lu_factor (->tensor a))))
  ([a {:keys [pivot] :or {pivot true}}]
   (tuple2->vector (torch/linalg_lu_factor (->tensor a) (clojure.core/boolean pivot)))))

(defn linalg-lu-solve
  "Computes the solution of a square system of linear equations from an LU factorization.

   Parameters:
   - lu-factorized: The LU factorization (from `linalg-lu-factor`).
   - pivots: The pivots (from `linalg-lu-factor`).
   - b: The right-hand side Tensor.
   - opts (optional): A map of options:
     - :left (Boolean): Whether to solve AX=B (true, default) or XA=B (false).
     - :adjoint (Boolean): Whether to solve with the adjoint matrix (default: false).

   Returns: A new Tensor.

   Example:
   ```clojure
   (let [[lu p] (linalg-lu-factor (eye 3))]
     (linalg-lu-solve lu p (ones [3 1])))
   ```"
  ([lu-factorized pivots b]
   (torch/linalg_lu_solve (->tensor lu-factorized) (->tensor pivots) (->tensor b)))
  ([lu-factorized pivots b {:keys [left adjoint]
                            :or {left true adjoint false}}]
   (torch/linalg_lu_solve (->tensor lu-factorized)
                          (->tensor pivots)
                          (->tensor b)
                          (clojure.core/boolean left)
                          (clojure.core/boolean adjoint))))

(defn linalg-ldl-factor
  "Computes the LDL factorization of a complex Hermitian or real symmetric matrix or a batch of such matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map of options:
     - :hermitian (Boolean): Whether to consider the input as Hermitian (default: true).

   Returns: A vector [LD pivots] (Tensors).

   Example:
   ```clojure
   (linalg-ldl-factor (eye 3))
   ```"
  ([a]
   (tuple2->vector (torch/linalg_ldl_factor (->tensor a))))
  ([a {:keys [hermitian] :or {hermitian true}}]
   (tuple2->vector (torch/linalg_ldl_factor (->tensor a) (clojure.core/boolean hermitian)))))

(defn linalg-ldl-solve
  "Computes the solution of a square system of linear equations from an LDL factorization.

   Parameters:
   - ldl: The LD factorization (from `linalg-ldl-factor`).
   - pivots: The pivots (from `linalg-ldl-factor`).
   - b: The right-hand side Tensor.
   - opts (optional): A map of options:
     - :hermitian (Boolean): Whether to consider the input as Hermitian (default: true).

   Returns: A new Tensor.

   Example:
   ```clojure
   (let [[ldl p] (linalg-ldl-factor (eye 3))]
     (linalg-ldl-solve ldl p (ones [3 1])))
   ```"
  ([ldl pivots b]
   (torch/linalg_ldl_solve (->tensor ldl) (->tensor pivots) (->tensor b)))
  ([ldl pivots b {:keys [hermitian] :or {hermitian true}}]
   (torch/linalg_ldl_solve (->tensor ldl)
                           (->tensor pivots)
                           (->tensor b)
                           (clojure.core/boolean hermitian))))

(defn linalg-solve-triangular
  "Computes the solution of a triangular system of linear equations.

   Parameters:
   - a: The input triangular Tensor.
   - b: The right-hand side Tensor.
   - upper: Whether `a` is upper triangular (Boolean).
   - opts (optional): A map of options:
     - :left (Boolean): Whether to solve AX=B (true, default) or XA=B (false).
     - :unitriangular (Boolean): Whether `a` is assumed to be unit triangular (default: false).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-solve-triangular (eye 3) (ones [3 1]) true)
   ```"
  ([a b upper]
   (torch/linalg_solve_triangular (->tensor a) (->tensor b) (clojure.core/boolean upper)))
  ([a b upper {:keys [left unitriangular]
               :or {left true unitriangular false}}]
   (torch/linalg_solve_triangular (->tensor a)
                                  (->tensor b)
                                  (clojure.core/boolean upper)
                                  (clojure.core/boolean left)
                                  (clojure.core/boolean unitriangular))))

(defn linalg-matrix-rank
  "Computes the numerical rank of a matrix or a batch of matrices.

   Parameters:
   - a: The input Tensor.
   - tol-or-opts (optional): Tolerance value (Number/Tensor) or a map of options:
     - :atol (Number): Absolute tolerance.
     - :rtol (Number): Relative tolerance.
     - :hermitian (Boolean): Whether the input is Hermitian (default: false).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-matrix-rank (eye 3))
   ```"
  ([a]
   (torch/linalg_matrix_rank (->tensor a)))
  ([a tol-or-opts]
   (if (map? tol-or-opts)
     (let [{:keys [atol rtol hermitian]
            :or {hermitian false}} tol-or-opts]
       (torch/linalg_matrix_rank (->tensor a)
                                 (->double-optional atol)
                                 (->double-optional rtol)
                                 (clojure.core/boolean hermitian)))
     (if (instance? Tensor tol-or-opts)
       (torch/linalg_matrix_rank (->tensor a) tol-or-opts)
       (torch/linalg_matrix_rank (->tensor a) (clojure.core/double tol-or-opts)))))
  ([a tol {:keys [hermitian] :or {hermitian false}}]
   (if (instance? Tensor tol)
     (torch/linalg_matrix_rank (->tensor a) tol (clojure.core/boolean hermitian))
     (torch/linalg_matrix_rank (->tensor a) (clojure.core/double tol) (clojure.core/boolean hermitian)))))

(defn linalg-cond
  "Computes the condition number of a matrix or a batch of matrices.

   Parameters:
   - a: The input Tensor.
   - ord (optional): Order of the norm (default: nil).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-cond (eye 3))
   ```"
  ([a]
   (torch/linalg_cond (->tensor a)))
  ([a ord]
   (torch/linalg_cond (->tensor a) (order->cond-arg ord))))

(defn linalg-lstsq
  "Computes a solution to the least squares problem of a system of linear equations.

   Parameters:
   - a: The input Tensor of shape (*, m, n).
   - b: The input Tensor of shape (*, m, k).
   - opts (optional): A map of options:
     - :rcond (Number): Cut-off ratio for small singular values.
     - :driver (Keyword/String): The name of the LAPACK driver to use.

   Returns: A map with :solution, :residuals, :rank, :singular-values.

   Example:
   ```clojure
   (linalg-lstsq (ones [3 2]) (ones [3 1]))
   ```"
  ([a b]
   (tuple4->map (torch/linalg_lstsq (->tensor a) (->tensor b))))
  ([a b {:keys [rcond driver]}]
   (tuple4->map
    (torch/linalg_lstsq (->tensor a)
                        (->tensor b)
                        (->double-optional rcond)
                        (->driver-optional driver)))))

(defn linalg-solve-ex
  "Computes the solution of a square system of linear equations. 
   Includes a check for errors instead of throwing.

   Parameters:
   - a: The input Tensor of shape (*, n, n).
   - b: The input Tensor of shape (*, n, k).
   - opts (optional): A map of options:
     - :left (Boolean): Whether to solve AX=B (true, default) or XA=B (false).
     - :check-errors (Boolean): Whether to check for errors (default: false).

   Returns: A vector [solution info] (Tensors).

   Example:
   ```clojure
   (linalg-solve-ex (eye 3) (ones [3 1]))
   ```"
  ([a b]
   (tuple2->vector (torch/linalg_solve_ex (->tensor a) (->tensor b))))
  ([a b {:keys [left check-errors]
         :or {left true check-errors false}}]
   (tuple2->vector
    (torch/linalg_solve_ex (->tensor a)
                           (->tensor b)
                           (clojure.core/boolean left)
                           (clojure.core/boolean check-errors)))))

(defn linalg-inv-ex
  "Computes the inverse of a square matrix or a batch of square matrices.
   Includes a check for errors instead of throwing.

   Parameters:
   - a: The input Tensor of shape (*, n, n).
   - opts (optional): A map of options:
     - :check-errors (Boolean): Whether to check for errors (default: false).

   Returns: A vector [inverse info] (Tensors).

   Example:
   ```clojure
   (linalg-inv-ex (eye 3))
   ```"
  ([a]
   (tuple2->vector (torch/linalg_inv_ex (->tensor a))))
  ([a {:keys [check-errors] :or {check-errors false}}]
   (tuple2->vector
    (torch/linalg_inv_ex (->tensor a) (clojure.core/boolean check-errors)))))

(defn linalg-cholesky-ex
  "Computes the Cholesky decomposition.
   Includes a check for errors instead of throwing.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map of options:
     - :upper (Boolean): Whether to return the upper triangular matrix (default: false).
     - :check-errors (Boolean): Whether to check for errors (default: false).

   Returns: A vector [L info] (Tensors).

   Example:
   ```clojure
   (linalg-cholesky-ex (eye 3))
   ```"
  ([a]
   (tuple2->vector (torch/linalg_cholesky_ex (->tensor a))))
  ([a {:keys [upper check-errors]
       :or {upper false check-errors false}}]
   (tuple2->vector
    (torch/linalg_cholesky_ex (->tensor a)
                              (clojure.core/boolean upper)
                              (clojure.core/boolean check-errors)))))

(defn linalg-lu-factor-ex
  "Computes the LU factorization of a matrix or a batch of matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map with :pivot (Boolean, default: true) and :check-errors (Boolean, default: false).

   Returns: A vector of three Tensors [LU, pivots, info].

   Example:
   ```clojure
   (linalg-lu-factor-ex (rand [3 3]))
   ```"
  ([a]
   (tuple3->vector (torch/linalg_lu_factor_ex (->tensor a))))
  ([a {:keys [pivot check-errors]
       :or {pivot true check-errors false}}]
   (tuple3->vector
    (torch/linalg_lu_factor_ex (->tensor a)
                               (clojure.core/boolean pivot)
                               (clojure.core/boolean check-errors)))))

(defn linalg-ldl-factor-ex
  "Computes the LDL factorization of a symmetric or Hermitian matrix or a batch of matrices.

   Parameters:
   - a: The input Tensor.
   - opts (optional): A map with :hermitian (Boolean, default: true) and :check-errors (Boolean, default: false).

   Returns: A vector of three Tensors [LD, pivots, info].

   Example:
   ```clojure
   (linalg-ldl-factor-ex (rand [3 3]))
   ```"
  ([a]
   (tuple3->vector (torch/linalg_ldl_factor_ex (->tensor a))))
  ([a {:keys [hermitian check-errors]
       :or {hermitian true check-errors false}}]
   (tuple3->vector
    (torch/linalg_ldl_factor_ex (->tensor a)
                                (clojure.core/boolean hermitian)
                                (clojure.core/boolean check-errors)))))

(defn linalg-tensorinv
  "Computes the multi-dimensional inverse of a tensor.

   Parameters:
   - a: The input Tensor.
   - ind (optional): The number of first dimensions that form the left-hand side of the inverse (Long).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-tensorinv (rand [4 4 4 4]) 2)
   ```"
  ([a] (torch/linalg_tensorinv (->tensor a)))
  ([a ind] (torch/linalg_tensorinv (->tensor a) (clojure.core/long ind))))

(defn linalg-tensorsolve
  "Solves the tensor system of linear equations `a x = b`.

   Parameters:
   - a: The left-hand side Tensor.
   - b: The right-hand side Tensor.
   - dims (optional): Dimensions of `a` to be moved to the right (sequence of Longs).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-tensorsolve (rand [2 3 2 3]) (rand [2 3]))
   ```"
  ([a b]
   (torch/linalg_tensorsolve (->tensor a) (->tensor b)))
  ([a b dims]
   (torch/linalg_tensorsolve (->tensor a) (->tensor b) (long-array dims))))

(defn linalg-multi-dot
  "Efficiently multiplies two or more 2-D tensors by choosing the optimal order of multiplications.

   Parameters:
   - tensors: A sequence of Tensors.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-multi-dot [(rand [2 3]) (rand [3 4]) (rand [4 5])])
   ```"
  [tensors]
  (torch/linalg_multi_dot (->tensor-vector tensors)))

(defn linalg-vecdot
  "Computes the dot product of two batches of vectors along a dimension.

   Parameters:
   - a: First input Tensor.
   - b: Second input Tensor.
   - dim (optional): Dimension along which to compute the dot product (Long, default: -1).

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-vecdot (rand [2 3]) (rand [2 3]) 1)
   ```"
  ([a b]
   (torch/linalg_vecdot (->tensor a) (->tensor b)))
  ([a b dim]
   (torch/linalg_vecdot (->tensor a) (->tensor b) (clojure.core/long dim))))

(defn linalg-vander
  "Generates a Vandermonde matrix of a 1-D input tensor.

   Parameters:
   - x: The input 1-D Tensor.
   - n (optional): Number of columns in the output (Long).

   Returns: A new Matrix (2-D Tensor).

   Example:
   ```clojure
   (linalg-vander (tensor [1 2 3 4]) 3)
   ```"
  ([x]
   (torch/linalg_vander (->tensor x)))
  ([x n]
   (torch/linalg_vander (->tensor x) (LongOptional. (clojure.core/long n)))))

(defn linalg-householder-product
  "Computes the product of Householder matrices.

   Parameters:
   - input: A Tensor containing the Householder reflections.
   - tau: A Tensor containing the Householder coefficients.

   Returns: A new Tensor.

   Example:
   ```clojure
   (linalg-householder-product (rand [3 3]) (rand [2]))
   ```"
  [input tau]
  (torch/linalg_householder_product (->tensor input) (->tensor tau)))

(defn lu-unpack
  "Unpacks the LU factorization and pivots into L, U, and P matrices.

   Parameters:
   - lu-data: The LU factorization Tensor.
   - pivots: The pivots Tensor.
   - opts (optional): A map with :unpack-data (Boolean, default: true) and :unpack-pivots (Boolean, default: true).

   Returns: A vector of three Tensors [P, L, U].

   Example:
   ```clojure
   (let [[lu p] (linalg-lu-factor-ex A)] (lu-unpack lu p))
   ```"
  ([lu-data pivots]
   (tuple3->vector (torch/lu_unpack (->tensor lu-data) (->tensor pivots))))
  ([lu-data pivots {:keys [unpack-data unpack-pivots]
                    :or {unpack-data true unpack-pivots true}}]
   (tuple3->vector
    (torch/lu_unpack (->tensor lu-data)
                     (->tensor pivots)
                     (clojure.core/boolean unpack-data)
                     (clojure.core/boolean unpack-pivots)))))

(defn clip-grad-norm-raw
  "Thin wrapper around native clip_grad_norm_ for TensorVector."
  [params max-norm]
  (torch/clip_grad_norm_ params (clojure.core/double max-norm) 2.0 false))

;; --- Utilities ---

(defn- walk-and-serialize [archive prefix thing]
  (cond
    (instance? Tensor thing)
    (.write archive (clojure.core/str prefix) ^Tensor thing)

    (instance? org.bytedeco.pytorch.Module thing)
    (if (empty? prefix)
      (.save ^org.bytedeco.pytorch.Module thing archive)
      (with-open [sub-archive (org.bytedeco.pytorch.OutputArchive.)]
        (.save ^org.bytedeco.pytorch.Module thing sub-archive)
        (.write archive (clojure.core/str prefix) sub-archive)))

    (map? thing)
    (doseq [[k v] thing]
      (let [k-str (if (keyword? k) (name k) (str k))]
        (walk-and-serialize archive (if (empty? prefix) k-str (clojure.core/str prefix "." k-str)) v)))

    (vector? thing)
    (doseq [[i v] (map-indexed vector thing)]
      (walk-and-serialize archive (if (empty? prefix) (str i) (clojure.core/str prefix "." i)) v))))

(defn- walk-and-deserialize [archive prefix thing]
  (cond
    (instance? Tensor thing)
    (do (.read archive (clojure.core/str prefix) ^Tensor thing) thing)

    (instance? org.bytedeco.pytorch.Module thing)
    (if (empty? prefix)
      (do (.load ^org.bytedeco.pytorch.Module thing archive) thing)
      (with-open [sub-archive (org.bytedeco.pytorch.InputArchive.)]
        (.read archive (clojure.core/str prefix) sub-archive)
        (.load ^org.bytedeco.pytorch.Module thing sub-archive)
        thing))

    (map? thing)
    (do (doseq [[k v] thing]
          (let [k-str (if (keyword? k) (name k) (str k))]
            (walk-and-deserialize archive (if (empty? prefix) k-str (clojure.core/str prefix "." k-str)) v)))
        thing)

    (vector? thing)
    (do (doseq [[i v] (map-indexed vector thing)]
          (walk-and-deserialize archive (if (empty? prefix) (str i) (clojure.core/str prefix "." i)) v))
        thing)))

(defn save
  "Saves a tensor, native module, or custom Clojure model to disk.

   Parameters:
   - model-or-tensor: The object to save.
   - filename: The destination path (String).

   Returns: nil.

   Example:
   ```clojure
   (save (ones [2 2]) \"model.pt\")
   ```"
  [model-or-tensor filename]
  (with-open [archive (org.bytedeco.pytorch.OutputArchive.)]
    (walk-and-serialize archive "" model-or-tensor)
    (.save_to archive (clojure.core/str filename))))

(defn load
  "Loads state from disk into a tensor, native module, or custom Clojure model.

   Parameters:
   - model-or-tensor: The object to load the state into.
   - filename: The source path (String).

   Returns: The loaded object.

   Example:
   ```clojure
   (load (ones [2 2]) \"model.pt\")
   ```"
  [model-or-tensor filename]
  (with-open [archive (org.bytedeco.pytorch.InputArchive.)]
    (.load_from archive (clojure.core/str filename))
    (walk-and-deserialize archive "" model-or-tensor)))

(defn jit-load
  "Loads a TorchScript module from disk.

   Parameters:
   - filename: The source path (String).
   - opts (optional): A map of options:
     - :device (Keyword/String): The device to load the module on.

   Returns: A JitModule.

   Example:
   ```clojure
   (jit-load \"module.pt\" {:device :cuda})
   ```"
  ([filename]
   (torch/load (clojure.core/str filename)))
  ([filename {:keys [device]}]
   (if device
     (torch/load
      (clojure.core/str filename)
      (DeviceOptional. (Device. (clojure.core/str (name device))))
      false)
     (jit-load filename))))

(defn jit-save
  "Saves a TorchScript JitModule to disk.

   Parameters:
   - jit-module: The JitModule to save.
   - filename: The destination path (String).

   Returns: The filename (String).

   Example:
   ```clojure
   (jit-save module \"module.pt\")
   ```"
  [jit-module filename]
  (when-not (instance? JitModule jit-module)
    (throw (IllegalArgumentException.
            (str "jit-save expects org.bytedeco.pytorch.JitModule, got " (type jit-module)))))
  (.save ^JitModule jit-module (clojure.core/str filename))
  filename)

(defn jit-forward
  "Executes the forward pass of a TorchScript module with the given arguments.
   Tries `forward` first, then falls back to `apply` for modules without a `forward` method.

   Parameters:
   - jit-module: The JitModule to run.
   - args: A sequence of Tensors or IValues.

   Returns: A Tensor or IValue.

   Example:
   ```clojure
   (jit-forward module [(ones [1 3])])
   ```"
  [jit-module args]
  (when-not (instance? JitModule jit-module)
    (throw (IllegalArgumentException.
            (str "jit-forward expects org.bytedeco.pytorch.JitModule, got " (type jit-module)))))
  (let [iv-args (IValueVector.)]
    (doseq [a args]
      (.push_back iv-args (if (instance? IValue a) a (IValue. (->tensor a)))))
    (let [out (try
                (.forward ^JitModule jit-module iv-args)
                (catch Throwable forward-e
                  (try
                    (.apply ^JitModule jit-module iv-args)
                    (catch Throwable apply-e
                      (throw
                       (ex-info "JIT module invocation failed for both forward and apply."
                                {:forward-error (.getMessage forward-e)
                                 :apply-error (.getMessage apply-e)}
                                apply-e))))))]
      (if (.isTensor ^IValue out)
        (.toTensor ^IValue out)
        out))))

(defn jit-trace
  "Not available in JavaCPP PyTorch presets as a high-level eager-module API."
  [& _]
  (throw (UnsupportedOperationException.
          "jit-trace is not exposed as a high-level eager-module API in this JavaCPP preset.")))

(defn jit-script
  "Not available in JavaCPP PyTorch presets as a high-level eager-module API."
  [& _]
  (throw (UnsupportedOperationException.
          "jit-script is not exposed as a high-level eager-module API in this JavaCPP preset.")))

(defn onnx-export
  "ONNX export is not exposed as a high-level API in this JavaCPP preset."
  [& _]
  (throw (UnsupportedOperationException.
          "onnx-export is not exposed as a high-level API in this JavaCPP preset.")))

(defmulti -seq type)
(defmethod -seq org.bytedeco.pytorch.Tensor [t]
  (let [s (size t)]
    (if (and (not (empty? s)) (pos? (first s)))
      (map #(ix t %) (range (first s)))
      nil)))

(defn tseq
  "Returns a Clojure sequence of slices along the first dimension.

   Parameters:
   - t: The input Tensor.

   Returns: A sequence of Tensors.

   Example:
   ```clojure
   (tseq (ones [3 2]))
   ```"
  [t] (-seq t))

(defn tensor-string
  "Returns a string representation of the tensor, including its data, shape, and options.

   Parameters:
   - t: The input Tensor.

   Returns: A String.

   Example:
   ```clojure
   (tensor-string (ones [2 2]))
   ```"
  [t]
  (let [t-cpu (.to (.contiguous (->tensor t)) (Device. "cpu") (torch/kFloat) false false (MemoryFormatOptional.))
        shape (size t-cpu)
        stype (.toString (.scalar_type t-cpu))
        dtype-kw (keyword (str/lower-case stype))
        info (get tech-dtype-info (keyword stype))
        address (.address (.data_ptr t-cpu))
        numel (.numel t-cpu)
        requires-grad (.requires_grad (->tensor t))]
    (with-out-str
      (print "(torch/tensor ")
      (if (and info (> numel 0))
        (let [buffer (native/wrap-address address (* numel (:bytes info)) (:kw info) :little-endian t-cpu)
              t-view (dtt/construct-tensor buffer (dims/dimensions (if (empty? shape) [1] shape)))
              ;; Strip the #tech.v3.tensor header from the view string
              data-str (str/replace (clojure.core/str t-view) #"^#tech\.v3\.tensor<.*?>\[.*\]\n" "")]
          (print (str/trim data-str)))
        (print "[]"))

      (let [opts (cond-> {}
                   (not= dtype-kw :float32) (assoc :dtype dtype-kw)
                   requires-grad (assoc :requires-grad true))]
        (when (seq opts)
          (print " ")
          (print opts)))
      (print ")"))))

(defn tprint
  "Minimalist, shape-aware printer for REPL usage.

   Parameters:
   - t: The input Tensor.

   Returns: nil.

   Example:
   ```clojure
   (tprint (ones [2 2]))
   ```"
  [t]
  (print (tensor-string t))
  nil)

(defmethod print-method org.bytedeco.pytorch.Tensor [t ^java.io.Writer w]
  (.write w (tensor-string t)))

(defn top-p
  "Nucleus (Top-p) sampling.
   Returns indices sampled from the top-p probability mass.

   Parameters:
   - logits: The input Tensor of logits.
   - p: The probability threshold (Number).

   Returns: A Tensor containing the sampled indices.

   Example:
   ```clojure
   (top-p (tensor [0.1 0.5 0.4]) 0.9)
   ```"
  [logits p]
  (let [probs (softmax logits -1)
        [sorted-probs sorted-indices] (sort probs :dim -1 :descending true)
        cum-probs (cumsum sorted-probs -1)
        ;; mask out tokens with cum_probs > p, but KEEP the first one
        mask (gt cum-probs (clojure.core/double p))
        ;; Shift mask: mask[1:] = mask[:-1], mask[0] = false
        [B _T] (size mask)
        mask-shifted (cat [(zeros [B 1] {:dtype :bool}) (ix mask :_ [0 (dec _T)])] 1)
        indices-to-remove mask-shifted
        sorted-probs-filtered (masked-fill sorted-probs indices-to-remove 0.0)
        ;; Re-normalize
        sorted-probs-norm (div sorted-probs-filtered (sum sorted-probs-filtered -1 :keepdim true))
        ;; Sample from sorted
        next-token-sorted (multinomial sorted-probs-norm 1)
        ;; Gather the original index
        next-token (gather sorted-indices 1 next-token-sorted)]
    next-token))
