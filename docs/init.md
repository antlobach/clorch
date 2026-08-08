# Parameter Initialization

The `clorch.nn.init` namespace provides functions to initialize tensor values. While Clorch layers come with sensible defaults, explicit initialization is often required for custom modules or specific research requirements.

## Basic Initializers

These functions modify the input tensor in-place.

```clojure
(require '[clorch.torch :as t]
         '[clorch.nn.init :as init])

(def x (t/zeros [3 3]))

(init/constant! x 3.14)
(init/ones! x)
(init/zeros! x)
(init/uniform! x -1.0 1.0)
(init/normal! x 0.0 0.01)
```

## Research-Standard Initializers

### Xavier (Glorot) Initialization
Designed for layers with symmetric activation functions (like `tanh` or `sigmoid`).

```clojure
;; Xavier Uniform
(init/xavier-uniform! weight)

;; Xavier Normal
(init/xavier-normal! weight)
```

### Kaiming (He) Initialization
Designed for layers with non-symmetric activations like `relu` or `leaky-relu`. Essential for training deep networks (e.g., ResNets).

```clojure
;; Kaiming Uniform
(init/kaiming-uniform! weight :non-linearity :relu)

;; Kaiming Normal
(init/kaiming-normal! weight :non-linearity :leaky-relu)
```

**Options for Kaiming:**
*   `:a`: Negative slope of the rectifier (used with `:leaky-relu`).
*   `:mode`: Either `:fan-in` (default) or `:fan-out`. `:fan-in` preserves the magnitude of the variance of the weights in the forward pass. `:fan-out` preserves the magnitudes in the backward pass.
*   `:non-linearity`: The non-linear function. Supported: `:relu`, `:leaky-relu`, `:tanh`, `:sigmoid`, `:linear`.

## Usage in Custom Models

It is common practice to initialize weights in the constructor of a model.

```clojure
(nn/defmodel MyModel [in out]
  [l1 (nn/linear in out)]
  (forward [x]
    (nn/forward l1 x)))

(def model (MyModel 128 64))
(init/kaiming-normal! (.weight (:l1 model)) :non-linearity :relu)
(init/zeros! (.bias (:l1 model)))
```
