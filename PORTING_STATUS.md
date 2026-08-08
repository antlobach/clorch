# Clorch: PyTorch to Clojure Port

This document tracks what's been ported from PyTorch to Clorch, and what still needs to be implemented.

---

## Implemented Features

### Core Tensor Operations (`clorch.torch`)

| Feature | Status |
|---------|--------|
| **Tensor Creation** | [x] |
| - `tensor` (from data) | [x] |
| - `randn`, `rand`, `rand-int`, `randperm` | [x] |
| - `ones`, `zeros`, `empty`, `full` | [x] |
| - `eye`, `linspace`, `logspace`, `arange` | [x] |
| - `bernoulli`, `manual-seed` | [x] |
| **Memory Management** | [x] |
| - `with-torch` macro | [x] |
| **Data Types** | [x] |
| - `dtype`, `to`, `to-float`, `to-long` | [x] |
| **Math Operations** | [x] |
| - `add`, `sub`, `mul`, `div` | [x] |
| - `pow`, `rsqrt`, `sqrt` | [x] |
| - `cos`, `sin`, `tan` | [x] |
| - `asin`, `acos`, `atan`, `atan2` | [x] |
| - `sinh`, `cosh`, `tanh` | [x] |
| - `asinh`, `acosh`, `atanh` | [x] |
| - `exp`, `expm1`, `exp2` | [x] |
| - `log`, `log1p`, `log2`, `log10` | [x] |
| - `floor`, `ceil`, `round`, `trunc`, `frac` | [x] |
| - `fmod`, `remainder` | [x] |
| - `abs`, `sign`, `neg`, `reciprocal` | [x] |
| - `matmul`, `bmm`, `mm` | [x] |
| - `dot`, `vdot`, `inner`, `outer` | [x] |
| - `softmax`, `clamp`, `clip` | [x] |
| **Reduction Operations** | [x] |
| - `sum`, `mean`, `var` | [x] |
| - `max`, `min`, `argmax`, `argmin` | [x] |
| - `topk`, `sort`, `argsort` | [x] |
| - `gather`, `cumsum`, `logsumexp` | [x] |
| - `all`, `any`, `nonzero`, `count-nonzero` | [x] |
| **Shape Operations** | [x] |
| - `size`, `view`, `reshape` | [x] |
| - `transpose`, `T`, `swapaxes`, `movedim` | [x] |
| - `unsqueeze`, `unflatten`, `flatten` | [x] |
| - `expand`, `tile`, `repeat` | [x] |
| - `cat`, `stack`, `unbind`, `split`, `chunk` | [x] |
| - `column-stack`, `row-stack`, `vstack`, `hstack`, `dstack` | [x] |
| **Slicing & Indexing** | [x] |
| - `ix` (ergonomic indexer) | [x] |
| - `select`, `index-select`, `masked-fill` | [x] |
| - `take-along-dim`, `gather`, `scatter-reduce` | [x] |
| **Linear Algebra (`clorch.torch/linalg-`)** | [x] |
| - `cholesky`, `inv`, `det`, `svd`, `qr` | [x] |
| - `matrix-exp`, `matrix-power`, `norm` | [x] |
| - `solve`, `lstsq`, `eig`, `eigh` | [x] |
| **Comparison** | [x] |
| - `eq`, `gt`, `lt`, `ge`, `le` | [x] |
| - `where`, `isclose`, `allclose` | [x] |
| - `isnan`, `isinf`, `isfinite` | [x] |
| **Other Operations** | [x] |
| - `tril`, `triu`, `diag`, `diagonal`, `trace` | [x] |
| - `flip`, `roll`, `rot90`, `cross` | [x] |
| - `meshgrid`, `broadcast-to`, `broadcast-tensors` | [x] |
| - `erf`, `erfc`, `erfinv`, `digamma`, `lgamma` | [x] |
| - `multinomial`, `bernoulli` | [x] |
| - `item-float` | [x] |
| **RoPE (Rotary Position Embedding)** | [x] |
| - `precompute-rope-freqs` | [x] |
| - `apply-rope` | [x] |
| **Save/Load / JIT** | [x] |
| - `save`, `load` | [x] |
| - `jit-load`, `jit-save`, `jit-forward` | [x] |
| **Printing** | [x] |
| - `tprint`, `tensor-string` | [x] |

---

### Autograd (`clorch.autograd`)

| Feature | Status |
|---------|--------|
| `set-requires-grad` | [x] |
| `detach` | [x] |
| `backward` | [x] |
| `grad` | [x] |
| `no-grad` macro | [x] |

---

### Neural Networks (`clorch.nn`)

| Feature | Status |
|---------|--------|
| **Core Protocols** | [x] |
| - `IModule` protocol | [x] |
| - `Parameter` handling | [x] |
| **Lifecycle** | [x] |
| - `forward`, `train`, `to` | [x] |
| - `parameters`, `named-parameters` | [x] |
| - `zero-grad` | [x] |
| **State Management** | [x] |
| - `state-dict`, `load-state-dict` | [x] |
| - `save-weights`, `load-weights` | [x] |
| **Introspection** | [x] |
| - `summary`, `apply`, `modules` | [x] |
| **Linear Layers** | [x] |
| - `linear` | [x] |
| **Convolutional Layers** | [x] |
| - `conv1d`, `conv2d`, `conv3d` | [x] |
| - `conv-transpose1d`, `conv-transpose2d`, `conv-transpose3d` | [x] |
| **Normalization Layers** | [x] |
| - `batchnorm1d`, `batchnorm2d`, `batchnorm3d` | [x] |
| - `groupnorm` | [x] |
| - `instancenorm1d`, `instancenorm2d`, `instancenorm3d` | [x] |
| - `layernorm` | [x] |
| - `rmsnorm` | [x] |
| **Pooling Layers** | [x] |
| - `max-pool1d`, `max-pool2d`, `max-pool3d` | [x] |
| - `avg-pool1d`, `avg-pool2d`, `avg-pool3d` | [x] |
| - `adaptive-max-pool1d/2d/3d` | [x] |
| - `adaptive-avg-pool1d/2d/3d` | [x] |
| **Padding Layers** | [x] |
| - `reflection-pad1d/2d/3d` | [x] |
| - `replication-pad1d/2d/3d` | [x] |
| - `zeropad1d/2d/3d` | [x] |
| - `constant-pad1d/2d/3d` | [x] |
| **Recurrent Layers** | [x] |
| - `rnn`, `lstm`, `gru` | [x] |
| **Embedding** | [x] |
| - `embedding` | [x] |
| - `embedding-from-pretrained` | [x] |
| **Activation Functions** | [x] |
| - `relu`, `relu6`, `leaky-relu` | [x] |
| - `gelu`, `tanh`, `sigmoid`, `silu`, `mish` | [x] |
| - `hardswish`, `hardsigmoid`, `hardtanh` | [x] |
| - `elu`, `selu`, `celu`, `softplus`, `softsign` | [x] |
| - `log-softmax`, `log-sigmoid` | [x] |
| **Dropout** | [x] |
| - `dropout`, `dropout2d` | [x] |
| - `alpha-dropout`, `feature-alpha-dropout` | [x] |
| **Other Layers** | [x] |
| - `flatten`, `unflatten`, `identity`, `bilinear` | [x] |
| - `cosine-similarity`, `pairwise-distance` | [x] |
| - `pixel-shuffle`, `pixel-unshuffle`, `upsample` | [x] |
| **Model Building** | [x] |
| - `sequential` | [x] |
| - `defmodel` macro | [x] |
| - `generate` (autoregressive) | [x] |
| **Custom Models** | [x] |
| - `SwiGLU` | [x] |
| - `GroupedQueryAttention` | [x] |

---

### Functional API (`clorch.nn.functional`)

| Feature | Status |
|---------|--------|
| **Activations** | [x] |
| All activations from nn module | [x] |
| **Pooling** | [x] |
| All pooling functions | [x] |
| **Linear/Conv** | [x] |
| - `linear`, `conv1d`, `conv2d`, `conv3d` | [x] |
| **Normalization** | [x] |
| - `batch-norm`, `layer-norm`, `group-norm` | [x] |
| **Image Operations** | [x] |
| - `pixel-shuffle`, `pixel-unshuffle`, `pad`, `interpolate` | [x] |
| **Loss Functions** | [x] |
| - `mse-loss`, `l1-loss`, `cross-entropy`, `nll-loss` | [x] |
| - `bce-loss`, `bce-with-logits-loss` | [x] |
| **Similarity** | [x] |
| - `cosine-similarity`, `pairwise-distance` | [x] |

---

### Optimizers (`clorch.optim`)

| Feature | Status |
|---------|--------|
| `sgd`, `adam`, `adamw`, `rmsprop`, `adagrad` | [x] |
| `zero-grad`, `step` | [x] |

---

### Initialization (`clorch.nn.init`)

| Feature | Status |
|---------|--------|
| `xavier-uniform!`, `xavier-normal!` | [x] |
| `kaiming-uniform!`, `kaiming-normal!` | [x] |
| `normal!`, `uniform!`, `constant!`, `zeros!`, `ones!` | [x] |

---

### Data Handling (`clorch.data`)

| Feature | Status |
|---------|--------|
| `IDataset` protocol, `dataset`, `defdataset` | [x] |
| `dataloader` (Thread & Process workers) | [x] |
| `tensor-dataset` | [x] |

---

### CUDA Support (`clorch.cuda`)

| Feature | Status |
|---------|--------|
| `available?` | [x] |

---

## What's Not Yet Implemented

### Specialized Tensors & Ops

| Feature | Status |
|---------|--------|
| Complex, Quantized tensors | [x] |
| Sparse tensors | ❌ |
| BFloat16 support | [x] |
| `polygamma`, bitwise ops, shifts | [x] |
| `lerp`, `addcmul`, `addcdiv` | [x] |

### Autograd Extensions

| Feature | Status |
|---------|--------|
| Custom autograd functions, `gradcheck` | ❌ |
| `inference_mode`, `set-grad-enabled` | ❌ |

### Specialized NN Layers

| Feature | Status |
|---------|--------|
| **Native Transformer Layers** | ❌ |
| (Clojure implementations exist in examples) | |
| **Container Modules** | ❌ |
| - `ModuleList`, `ModuleDict`, `ParameterList` | ❌ |
| **Utilities** | ❌ |
| - `clip-grad-value!`, `register-hook` | ❌ |

### Optimizer Extras

| Feature | Status |
|---------|--------|
| `ASGD`, `Adamax`, `NAdam`, `RAdam`, `LBFGS` | ❌ |
| Learning rate schedulers | ❌ |
| `state_dict` for optimizers | ❌ |

---

## Summary

| Category | Implemented | Not Implemented |
|----------|-------------|-----------------|
| Core Tensor Ops | ~170 | ~10 |
| Autograd | 5 | ~5 |
| NN Layers | ~80 | ~20 |
| Optimizers | 7 | ~10 |
| Data | 5 | ~5 |
| CUDA | 1 | ~10 |
| **Total** | **~268** | **~60** |

---

*Last updated: 2026-03-13*
