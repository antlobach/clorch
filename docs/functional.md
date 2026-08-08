# Functional API

The `clorch.nn.functional` namespace (aliased as `F`) contains functional versions of standard layers and utilities. These functions are stateless and expect weights and biases to be passed explicitly where applicable.

## Core Operations

### Linear & Convolution
```clojure
(require '[clorch.nn.functional :as F])

;; Linear
(F/linear input weight bias)

;; Convolution (1D, 2D, 3D)
(F/conv2d input weight :bias bias :stride 1 :padding 0)
```

### Pooling
```clojure
(F/max-pool2d input 2)
(F/avg-pool2d input 2)
(F/adaptive-avg-pool2d input [7 7])
```

### Normalization
```clojure
(F/layer-norm input normalized-shape :weight weight :bias bias :eps eps)
(F/group-norm input num-groups :weight weight :bias bias :eps eps)
```

## Regularization & Utilities

### Dropout
```clojure
;; p is the probability of an element to be zeroed
(F/dropout x 0.5 :training? true)

;; Spatial Dropout
(F/dropout2d x 0.5 :training? true)
```

### Interpolation & Padding
```clojure
;; Resize image-like tensors
(F/interpolate input :size [224 224] :mode :nearest)

;; Advanced Padding
;; pad is a vector of [left right top bottom ...]
(F/pad input [1 1 1 1] :mode :reflect)
```

## Loss Functions
See the [Loss Functions](losses.md) guide for a full list.

```clojure
(F/mse-loss input target)
(F/cross-entropy logits targets)
```

## When to use Functional vs Modules?
*   **Use Modules (`nn/`)**: For standard layers with learnable parameters. Modules handle parameter registration and state management (training/eval mode) automatically.
*   **Use Functional (`F/`)**: For stateless operations (activations, pooling) or when you need total manual control over weights (e.g., custom meta-learning loops or weights sharing).
