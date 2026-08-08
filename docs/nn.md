# Neural Networks

The `clorch.nn` namespace provides high-level modules for building neural networks. It leverages LibTorch's C++ frontend for native performance while maintaining Clojure's idiomatic data-first philosophy.

## Common Layers

### Linear (Dense) Layers
```clojure
(require '[clorch.nn :as nn])

;; Linear(in_features=10, out_features=5)
(nn/linear 10 5)

;; Linear without bias
(nn/linear 10 5 :bias false)
```

### Convolutional Layers
Clorch supports 1D, 2D, and 3D convolutions and their transpose variants.

```clojure
;; Conv2d(in_channels=3, out_channels=16, kernel_size=3)
(nn/conv2d 3 16 3)

;; Advanced parameters
(nn/conv2d 3 16 [3 5] :stride 2 :padding 1 :dilation 2 :groups 1)

;; Transpose Convolution
(nn/conv-transpose2d 16 3 3 :stride 2)
```

### Recurrent Layers
Directly wraps LibTorch's optimized C++ RNN implementations.

```clojure
;; LSTM(input_size=10, hidden_size=20)
(nn/lstm 10 20)

;; Bidirectional with multiple layers
(nn/gru 10 20 :num-layers 2 :bidirectional true :dropout 0.1)
```

### Normalization
```clojure
(nn/batchnorm2d 64)
(nn/layernorm 128)
(nn/rmsnorm 512) ;; High-performance RMSNorm for LLMs
(nn/groupnorm 8 64)
```

### Pooling & Padding
```clojure
;; Pooling
(nn/max-pool2d 2)
(nn/avg-pool2d 2)
(nn/adaptive-avg-pool2d [7 7])

;; Padding
(nn/zeropad2d 1)
(nn/reflection-pad2d 2)
(nn/constant-pad2d 1 3.14)
```

## Containers & Utilities

### Sequential
A simple linear stack of layers.

```clojure
(def model (nn/sequential
             (nn/linear 10 20)
             (nn/relu)
             (nn/dropout 0.5)
             (nn/linear 20 1)))
```

### Custom Models with `defmodel`

`defmodel` creates a reusable model constructor without requiring a manual `defrecord` or `IModule` implementation. It has three parts:

1. A model name and constructor arguments.
2. A binding vector of field/value pairs created when the model is instantiated.
3. A `forward` form that can refer to those fields directly.

```clojure
(require '[clorch.torch :as t]
         '[clorch.nn :as nn]
         '[clorch.nn.functional :as F])

(nn/defmodel MyClassifier [in-dim hidden-dim num-classes]
  [l1 (nn/linear in-dim hidden-dim)
   l2 (nn/linear hidden-dim num-classes)]
  (forward [x]
    (nn/forward l2 (F/relu (nn/forward l1 x)))))

(def classifier (MyClassifier 784 256 10))
(def logits (nn/forward classifier (t/randn [32 784])))
(t/size logits) ; => [32 10]
```

Fields containing modules, parameters, tensors, vectors, or maps participate in recursive model operations. `nn/train`, `nn/to`, `nn/parameters`, and state-dictionary functions traverse them automatically. The generated model is also callable:

```clojure
(classifier (t/randn [32 784]))
```

Use [`examples/simple.clj`](https://github.com/antlobach/clorch/blob/main/examples/simple.clj) for a minimal runnable model and [`examples/llms_from_scratch.clj`](https://github.com/antlobach/clorch/blob/main/examples/llms_from_scratch.clj) for nested custom modules.

## Lifecycle API

### Mode Management
```clojure
(nn/train model true)  ;; Set to training mode (enables dropout/batchnorm updates)
(nn/train model false) ;; Set to evaluation mode
```

### Device & Dtype
```clojure
(nn/to model :cuda)    ;; Move entire model to GPU
(nn/to model :float64) ;; Convert all parameters to Double
```

### State & Parameters
```clojure
;; Get all parameters as a native TensorVector
(nn/parameters model)

;; Get a nested map of all weights/biases
(nn/state-dict model)

;; Load weights from a state dict
(nn/load-state-dict model saved-sd)
```

## Model Inspection

### Summary (`nn/summary`)
Clorch provides a PyTorch-style summary tool that traces execution to show actual output shapes and parameter counts for every layer. 

Unlike static analyzers, `nn/summary` performing a **dry run** forward pass. This means it captures the exact shapes resulting from your specific model logic, including complex reshaping and broadcasting.

#### Usage
```clojure
;; 1. Simple input shape
(nn/summary model [1 784])

;; 2. Complex input (e.g. for LLMs or models with multiple arguments)
(nn/summary gpt {:idx (torch/zeros [8 4] {:dtype :int64}) 
                 :mask (torch/ones [4 4])})
```

#### How it Works
When you call `summary`, Clorch:
1.  Sets the model to `eval` mode.
2.  Wraps execution in `autograd/no-grad`.
3.  Binds a dynamic tracer that intercepts every `nn/forward` call.
4.  Captures the type, parameter count, and output shape of every layer encountered during the pass.

**Example Output:**
```text
----------------------------------------------------------------
Layer (type)                   Output Shape         Param #   
================================================================
Linear                         [1 20]               220       
ReLU                           [1 20]               0         
Linear                         [1 5]                105       
PersistentVector               [1 5]                325       
================================================================
Total params: 325
Trainable params: 325
Non-trainable params: 0
----------------------------------------------------------------
```
