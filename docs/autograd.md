# Automatic Differentiation (Autograd)

The `clorch.autograd` namespace provides the engine for computing gradients automatically. It is a thin wrapper around LibTorch's native autograd system.

## Basic Usage

### Enabling Gradients
Tensors must have `:requires-grad true` to be tracked by the autograd engine.

```clojure
(require '[clorch.torch :as t]
         '[clorch.autograd :as autograd])

(def x (t/tensor [2.0] {:requires-grad true}))
(def y (t/pow x 2)) ;; y = x^2
```

### Computing Gradients
Use `backward` to compute the gradient of a scalar tensor.

```clojure
(autograd/backward y)

;; Access the gradient
(autograd/grad x) ;; → [4.0] (d/dx x^2 = 2x = 2*2 = 4)
```

## Advanced Autograd

### Detaching from Graph
If you want to move a tensor out of the computation graph (e.g., for calculating statistics without tracking gradients), use `detach`.

```clojure
(def detached (autograd/detach y))
```

### Disabling Tracking (`no-grad`)
Use the `no-grad` macro to execute a block of code without tracking gradients. This is essential for inference and validation loops to save memory and compute.

```clojure
(autograd/no-grad
  (let [pred (nn/forward model x)]
    (calculate-accuracy pred targets)))
```

### Manual Gradient Control
```clojure
;; Explicitly set requires-grad on an existing tensor
(autograd/set-requires-grad x true)
```

## Training Loop Integration

In a typical training loop, you zero out gradients, compute the loss, perform backpropagation, and then update parameters using an optimizer.

```clojure
(optim/zero-grad optimizer) ;; Clear previous gradients
(autograd/backward loss)  ;; Compute current gradients
(optim/step optimizer)    ;; Update weights
```

Run the complete step inside a per-batch `with-torch` scope. See [Memory Management](memory.md#canonical-training-loop).
