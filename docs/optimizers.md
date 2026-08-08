# Optimizers

The `clorch.optim` namespace provides standard optimization algorithms. These are thin wrappers around LibTorch's C++ implementations, offering high performance and low JVM overhead.

## Standard Optimizers

### SGD (Stochastic Gradient Descent)
```clojure
(require '[clorch.optim :as optim])

(def opt (optim/sgd (nn/parameters model) 
                    :lr 0.01 
                    :momentum 0.9))
```

### Adam
The most popular adaptive optimizer.
```clojure
(def opt (optim/adam (nn/parameters model) 
                     :lr 0.001 
                     :betas [0.9 0.999]))
```

### AdamW
A variant of Adam that decouples weight decay from the gradient update. Recommended for Transformer and LLM training.
```clojure
(def opt (optim/adamw (nn/parameters model) 
                      :lr 3e-4 
                      :weight-decay 0.01))
```

## Additional Optimizers

*   **RMSprop**: `(optim/rmsprop params :lr 0.01 :alpha 0.99)`
*   **Adagrad**: `(optim/adagrad params :lr 0.01)`

## Lifecycle API

### Resetting Gradients
Before computing the loss, always clear previous gradients.
```clojure
(optim/zero-grad opt)
```

### Updating Weights
After performing backpropagation (`autograd/backward`), update the parameters.
```clojure
(optim/step opt)
```

## Training Step Example

```clojure
(defn train-step [model optimizer batch]
  (optim/zero-grad optimizer)
  (let [pred (nn/forward model (:data batch))
        loss (F/mse-loss pred (:target batch))]
    (autograd/backward loss)
    (optim/step optimizer)
    (t/item-float loss)))
```

Call `train-step` from the per-batch scope shown in [Memory Management](memory.md#canonical-training-loop).
