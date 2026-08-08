# Loss Functions

Clorch implements the most common loss functions used in supervised learning. All loss functions are available in the `clorch.nn.functional` namespace.

## Standard Losses

### Mean Squared Error (MSE)
Standard loss for regression tasks.
```clojure
(require '[clorch.nn.functional :as F])

(F/mse-loss input target)
```

### L1 Loss (Mean Absolute Error)
More robust to outliers than MSE.
```clojure
(F/l1-loss input target)
```

### Smooth L1 Loss (Huber Loss)
Combines MSE and L1; stable and robust.
```clojure
(F/smooth-l1-loss input target)
```

## Classification Losses

### Cross Entropy
Combines `LogSoftmax` and `NLLLoss` in one single class. Used for multi-class classification.
```clojure
(F/cross-entropy logits targets)
```

### NLL Loss (Negative Log Likelihood)
Used for multi-class classification (expects log-probabilities as input).
```clojure
(F/nll-loss input target)
```

### Binary Cross Entropy (BCE)
Used for binary classification.
```clojure
;; Expects probabilities (after sigmoid)
(F/bce-loss input target)

;; More numerically stable: combines sigmoid + BCE
(F/bce-with-logits-loss input target)
```

## Usage in Training Loop

```clojure
(def loss (F/cross-entropy pred target))
(autograd/backward loss)
```

Loss tensors retain their autograd graphs. Extract required JVM values and let the per-batch scope close after the optimizer step; see [Memory Management](memory.md#canonical-training-loop).
