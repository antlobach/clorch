---
layout: default
title: Home
---

# Clorch

**Clorch** is a high-performance deep learning library for Clojure with CPU and NVIDIA CUDA GPU execution. Its tensor, autograd, neural-network, optimizer, data, and device APIs follow PyTorch's names and behavior closely while remaining idiomatic and REPL-friendly.

---

## Why Clorch?

*   **Native Speed**: All heavy computation is executed in C++ via optimized LibTorch kernels.
*   **REPL First**: Designed for iterative development with shape-aware printing and ergonomic slicing.
*   **Broad PyTorch Coverage**: Supports more than 100 tensor operations, activations, and layers.
*   **Transparent Memory**: Uses deterministic `with-torch` scopes to prevent native memory bloat.
*   **LLM Architecture**: Includes embeddings, RMSNorm, RoPE, grouped-query attention, SwiGLU, KV caching, and autoregressive generation.

---

## Quick Example

```clojure
(require '[clorch.torch :as t]
         '[clorch.nn :as nn]
         '[clorch.nn.functional :as F]
         '[clorch.autograd :as autograd])

;; 1. Ergonomic Indexing
(def x (t/tensor [[1 2 3] [4 5 6]]))
(t/ix x :_ [1 3]) ;; → [[2.0, 3.0], [5.0, 6.0]]

;; 2. Seamless Autograd
(def a (t/tensor [2.0] {:requires-grad true}))
(def b (t/pow a 3))
(autograd/backward b)
(autograd/grad a) ;; → [12.0] (d/da a^3 = 3a^2 = 3*4 = 12)

;; 3. Modern Architectures
(def llama-layer (nn/sequential
                   (nn/rmsnorm 128)
                   (nn/linear 128 512)
                   (nn/silu)))

;; 4. Architecture Summary
(nn/summary llama-layer [1 16 128])
```

---

## Documentation

### Core API
- [**Tensors & Operations**](tensors.md): Creation, math, and shape management.
- [**Distributions**](distributions.md): Sampling, log-probability, moments.
- [**Slicing & Indexing**](slicing.md): Deep dive into the ergonomic `ix` API.
- [**Autograd**](autograd.md): Understanding the automatic differentiation engine.
- [**PyTorch Parity**](pytorch-parity.md): Current measurements, capability comparison, LLM coverage, and roadmap.

### Building Models
- [**Neural Network Modules**](nn.md): Linear, CNN, RNN, and custom architectures.
- [**Functional API**](functional.md): Stateless operations and low-level control.
- [**Activation Functions**](activations.md): Complete list of 25+ supported activations.
- [**Loss Functions**](losses.md): Criteria for regression and classification.

### Operations & Stability
- [**Memory Management**](memory.md): How to use `with-torch` effectively.
- [**Performance & Profiling**](profiling.md): Tips for high-speed training and leak detection.

---

## Running Examples
Check out the `examples/` directory for full implementations:
*   `autograd_tutorial.clj`: Basic math and gradients.
*   `modern_llama.clj`: Llama-3 style components (RoPE, GQA, SwiGLU).
*   `synthetic.clj`: A complete training loop from data to optimization.
