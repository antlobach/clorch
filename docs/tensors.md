# Tensors & Operations

Tensors are the fundamental data structures in Clorch. They are multidimensional arrays that live in native C++ memory and support automatic differentiation.

## Creation (Factory Methods)

### From Clojure Data
You can create tensors from vectors, nested vectors, or sequences.

```clojure
(require '[clorch.torch :as t])

;; 1D tensor
(t/tensor [1 2 3])

;; 2D tensor
(t/tensor [[1 2] [3 4]])

;; Specify dtype
(t/tensor [1 2 3] {:dtype :float64})

;; Enable gradients
(t/tensor [1.0 2.0] {:requires-grad true})
```

### Standard Initializers
Clorch supports the full suite of LibTorch factory methods.

```clojure
;; Zeros and Ones
(t/zeros [3 3])
(t/ones [2 5])

;; Identity Matrix
(t/eye 3)

;; Range
(t/arange 10)       ;; 0 to 9
(t/arange 1 11)     ;; 1 to 10
(t/arange 0 1 0.1)  ;; 0 to 0.9 with step 0.1

;; Spacing
(t/linspace 0 10 5) ;; [0.0, 2.5, 5.0, 7.5, 10.0]
(t/logspace 0 2 3)  ;; [1.0, 10.0, 100.0] (base 10 by default)

;; Constant Value
(t/full [2 2] 3.14)

;; Random
(t/randn [3 3])     ;; Normal distribution (mean 0, std 1)
(t/rand-int 0 10 [5]) ;; Random integers between 0 and 10
(t/randperm 5)      ;; Random permutation of [0, 1, 2, 3, 4]
```

## Basic Operations

### Element-wise Math
Most operations work exactly like their Clojure or PyTorch counterparts.

```clojure
(def a (t/tensor [1 2 3]))
(def b (t/tensor [4 5 6]))

(t/add a b) ;; [5, 7, 9]
(t/sub a b) ;; [-3, -3, -3]
(t/mul a b) ;; [4, 10, 18]
(t/div a b) ;; [0.25, 0.4, 0.5]

;; Scalar broadcasting
(t/add a 10) ;; [11, 12, 13]
```

### Linear Algebra
```clojure
(def m1 (t/randn [3 5]))
(def m2 (t/randn [5 2]))

(t/matmul m1 m2) ;; [3 2]
(t/mm m1 m2)     ;; Shorthand for matmul

(def v1 (t/tensor [1 2 3]))
(def v2 (t/tensor [4 5 6]))
(t/outer v1 v2)  ;; Outer product
```

### Reductions
```clojure
(def x (t/tensor [[1 2] [3 4]]))

(t/sum x)    ;; 10.0
(t/mean x)   ;; 2.5
(t/max x)    ;; 4.0
(t/min x)    ;; 1.0

;; Along dimensions
(t/sum x 0)  ;; [4, 6]
(t/sum x 1)  ;; [3, 7]

;; Keep dimensions
(t/sum x 1 :keepdim true) ;; [[3], [7]]
```

## Shape Management

### Reshaping & Viewing
Clorch supports the standard PyTorch `reshape` and `view` APIs, including dimension inference via `-1`.

```clojure
(def x (t/arange 12))

;; Standard reshape
(t/reshape x [3 4])

;; Dimension inference
(t/reshape x [2 -1]) ;; Results in [2 6]

;; View (shares memory, requires contiguous tensor)
(t/view x [4 3])
```

### Flatten & Unflatten
```clojure
(def x (t/randn [1 3 4 4]))

;; Flatten all but batch
(t/flatten x 1) ;; [1 48]

;; Unflatten a dimension
(def y (t/randn [10 12]))
(t/unflatten y 1 [3 4]) ;; [10 3 4]
```

### Squeezing & Expansion
```clojure
(def x (t/randn [1 10 1]))

(t/unsqueeze x 0) ;; [1 1 10 1]
(t/expand (t/tensor [1 2 3]) [3 3]) ;; Expand to [3 3] matrix
```

## Utility Methods

### Type Conversion
```clojure
(def x (t/tensor [1 2 3]))

(t/to-float x) ;; Convert to Float32
(t/to-long x)  ;; Convert to Int64
(t/to x :float64) ;; Explicit keyword dtype
```

### Device Placement
Clorch selects an available CPU or CUDA native backend when `clorch.torch` loads, but tensors remain on CPU unless you place them explicitly. Resolve one application-level device and use it consistently:

```clojure
(require '[clorch.cuda :as cuda]
         '[clorch.nn :as nn])

(def device (if (cuda/available?) :cuda :cpu))
(def x (t/randn [32 128] {:device device}))
(def y (t/to existing-tensor device))
```

Move models with `(nn/to model device)`. A CUDA model and its input tensors must use the same device.

### Inspection
```clojure
(t/size x)  ;; Get shape vector [3]
(t/dtype x) ;; Get dtype keyword :int64
(t/item-float (t/tensor [3.14])) ;; Extract single value as JVM float
```

## Logic & Search

### Comparisons
```clojure
(t/eq a b) ;; Element-wise equality
(t/gt a b) ;; Greater than
```

### Conditional Selection
```clojure
;; (where condition x y)
(t/where (t/gt x 0) x (t/zeros [10]))
```

### Searching
```clojure
(t/argmax x)     ;; Index of max value
(t/argmin x)     ;; Index of min value
(t/nonzero x)    ;; Indices of non-zero elements
(t/topk x 5)     ;; Top 5 values and their indices
(t/sort x :descending true) ;; Sorted values and indices
```

## Sampling (Generative AI)

### Multinomial
```clojure
;; Sample 1 index from a probability distribution
(t/multinomial (t/tensor [0.1 0.8 0.1]) 1)
```

### Nucleus (Top-p) Sampling
A built-in utility for high-quality text generation. It filters the probability mass to the top `p` percentile before sampling.

```clojure
(t/top-p (t/tensor [[0.1 0.8 0.1]]) 0.9)
```
