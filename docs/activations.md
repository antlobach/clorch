# Activation Functions

Clorch provides a comprehensive set of activation functions in both the stateful `clorch.nn` namespace (for use in modules) and the functional `clorch.nn.functional` namespace.

## Standard Activations

| Feature | `clorch.nn` | `clorch.nn.functional` |
| :--- | :--- | :--- |
| ReLU | `(nn/relu)` | `(F/relu x)` |
| Sigmoid | `(nn/sigmoid)` | `(F/sigmoid x)` |
| Tanh | `(nn/tanh)` | `(F/tanh x)` |
| GeLU | `(nn/gelu)` | `(F/gelu x)` |
| SiLU (Swish) | `(nn/silu)` | `(F/silu x)` |
| Softmax | `(nn/softmax dim)` | `(F/softmax x dim)` |

## Enhanced & Specialized

### LeakyReLU
```clojure
;; Stateful
(nn/leaky-relu 0.1)

;; Functional
(F/leaky-relu x 0.1)
```

### PReLU (Parametric ReLU)
Learns the slope of the negative part.
```clojure
;; Stateful
(nn/prelu {:num-parameters 1 :init 0.25})

;; Functional (requires manual weight management)
(F/prelu x weight)
```

### ELU (Exponential Linear Unit)
```clojure
(nn/elu 1.0)
(F/elu x 1.0)
```

### GLU (Gated Linear Unit)
```clojure
(nn/glu -1)
(F/glu x -1)
```

## Shrinkage Functions
Useful for sparse representations.
*   **Hardshrink**: `(nn/hardshrink 0.5)`
*   **Softshrink**: `(nn/softshrink 0.5)`
*   **Tanhshrink**: `(nn/tanhshrink)`

## Modern & Experimental
*   **Mish**: `(nn/mish)` / `(F/mish x)`
*   **Hardswish**: `(nn/hardswish)` / `(F/hardswish x)`
*   **Hardsigmoid**: `(nn/hardsigmoid)` / `(F/hardsigmoid x)`
*   **Hardtanh**: `(nn/hardtanh min max)` / `(F/hardtanh x min max)`

## Comparison Table

| Activation | Range | Usage |
| :--- | :--- | :--- |
| `ReLU` | [0, ∞) | Most common hidden layer activation |
| `Sigmoid` | (0, 1) | Binary classification, probability |
| `Tanh` | (-1, 1) | Centered output, RNNs |
| `Softmax` | (0, 1) | Multi-class classification |
| `GeLU` | (-0.17, ∞) | Transformers (BERT, GPT) |
| `SiLU` | (-0.28, ∞) | Modern vision and language models |
| `Softplus` | (0, ∞) | Smooth approximation of ReLU |
