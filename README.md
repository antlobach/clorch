# Clorch

<p align="center">
  <img src="clorch-logo.png" alt="Clorch logo">
</p>

<p align="center">
  <a href="https://github.com/antlobach/clorch/actions/workflows/ci.yml"><img src="https://github.com/antlobach/clorch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://antlobach.github.io/clorch/"><img src="https://img.shields.io/badge/docs-online-4051B5?logo=materialformkdocs&logoColor=white" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-4c1.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/Clojure-5881D8?logo=clojure&logoColor=white" alt="Clojure">
  <img src="https://img.shields.io/badge/LibTorch-2.1.2-EE4C2C?logo=pytorch&logoColor=white" alt="LibTorch 2.1.2">
  <img src="https://img.shields.io/badge/CUDA-12.3-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.3">
</p>

**CPU and NVIDIA CUDA GPU acceleration. PyTorch-style API.** Clorch brings PyTorch's tensor, autograd, neural-network, optimizer, data, and device concepts to idiomatic, REPL-friendly Clojure on top of the **LibTorch** C++ engine via Bytedeco's JavaCPP presets. Supported operations have tested numerical parity with PyTorch, and Clorch follows PyTorch's names and behavior closely.

## Highlights

- **LLM-ready architecture:** embeddings, RMSNorm, rotary position embeddings (RoPE), grouped-query attention (GQA), SwiGLU, causal masking, KV caching, and autoregressive generation.
- **Custom models:** compose modules with `nn/defmodel`, including residual networks, GPT-style transformers, and Llama-style blocks.
- **Training stack:** automatic differentiation, neural-network modules, losses, optimizers, data loaders, state dictionaries, and deterministic native-memory scopes.
- **CPU and CUDA:** automatic native-backend selection with explicit PyTorch-style tensor and model placement.
- **Measured parity:** 100% of the current 40 cross-language numerical scenarios pass; tracked feature breadth is approximately 81.7%. See [PyTorch Parity](docs/pytorch-parity.md).

---

## Quick Start

```clojure
(require '[clorch.torch :as t]
         '[clorch.linalg :as linalg]
         '[clorch.nn :as nn]
         '[clorch.nn.functional :as F]
         '[clorch.autograd :as autograd])

;; 1. Ergonomic Indexing (Python-style slicing)
(def x (t/tensor [[1 2 3] [4 5 6]]))
(t/ix x :_ [1 3]) ;; → [[2.0, 3.0], [5.0, 6.0]]

;; 2. Automatic Differentiation
(def a (t/tensor [2.0] {:requires-grad true}))
(def b (t/pow a 3))
(autograd/backward b)
(autograd/grad a) ;; → [12.0] (d/da a^3 = 3a^2 = 12)

;; 3. Neural Networks
(def model (nn/sequential
             (nn/linear 10 20)
             (nn/relu)
             (nn/linear 20 1)))

(def out (nn/forward model (t/randn [1 10])))

;; 4. Custom Models
(nn/defmodel CustomMLP [in hidden out]
  [l1 (nn/linear in hidden)
   l2 (nn/linear hidden out)]
  (forward [x]
    (let [h (F/relu (nn/forward l1 x))]
      (nn/forward l2 h))))

(def custom-model (CustomMLP 10 32 1))
(nn/forward custom-model (t/randn [4 10]))

;; 5. Model Summary (Architecture Overview)
(nn/summary model [1 10])

;; 6. Einstein-notation eDSL over Torch
(require '[clorch.einsum :refer [defein ein]])

(let [A (t/tensor [[1.0 2.0 3.0]
                   [4.0 5.0 6.0]])
      x (t/tensor [10.0 20.0 30.0])]
  (defein y [i] := (* (A i j) (x j)))
  (mapv t/item-float (t/tseq y))) ;; => [140.0 320.0]
```

### Native memory in repeated workloads

Java GC eventually reclaims unreachable Clorch tensors, but it cannot see the size or pressure of LibTorch CPU allocations or CUDA VRAM. Keep long-lived models and optimizers outside the loop, then use one `with-torch` scope per allocating batch or generation step:

```clojure
(doseq [batch dataloader]
  (let [loss-value
        (t/with-torch
          (let [loss (train-step model optimizer batch)]
            (t/item-float loss)))]
    (println "Loss:" loss-value)))
```

End the scope with a JVM scalar or `nil` unless a tensor should escape. See [Memory Management](docs/memory.md) for GC behavior, explicit retention, manual release, and REPL sessions.

---

## Key Features

*   **Native Performance**: Executes heavy tensor compute in C++ via optimized LibTorch kernels.
*   **REPL First**: Designed for iterative development with shape-aware printing and ergonomic slicing.
*   **Broad PyTorch Coverage**: Supports standard PyTorch operations, activations (25+), and layers (CNN, RNN, normalization, etc.).
*   **Transparent Memory**: Uses deterministic `with-torch` scopes to prevent native memory bloat and OOMs.
*   **Modern Components**: Built-in support for Llama-3 style architecture pieces like `RMSNorm`, `RoPE`, and `GQA`.

---

## Visualizing Architectures

Clorch includes a professional model summary tool that traces execution to show actual activation shapes and parameter counts.

```clojure
(nn/summary my-model [1 3 224 224])
```

**Output:**
```text
----------------------------------------------------------------
Layer (type)                   Output Shape         Param #   
================================================================
Conv2d                         [1 64 112 112]       9408      
BatchNorm2d                    [1 64 112 112]       128       
ReLU                           [1 64 112 112]       0         
MaxPool2d                      [1 64 56 56]         0         
...
================================================================
Total params: 25557032
Trainable params: 25557032
Non-trainable params: 0
----------------------------------------------------------------
```

---

## Documentation

Visit our [**Documentation Site**](https://antlobach.github.io/clorch/) for detailed guides:

### Core API
- [**Tensors & Operations**](docs/tensors.md): Creation, math, and shape management.
- [**Slicing & Indexing**](docs/slicing.md): Deep dive into the ergonomic `ix` API.
- [**Autograd**](docs/autograd.md): Understanding the automatic differentiation engine.

### Building Models
- [**Neural Network Modules**](docs/nn.md): Linear, CNN, RNN, and custom architectures.
- [**Functional API**](docs/functional.md): Stateless operations and low-level control.
- [**Activation Functions**](docs/activations.md): Complete list of 25+ supported activations.
- [**Loss Functions**](docs/losses.md): Criteria for regression and classification.

### Operations & Stability
- [**Memory Management**](docs/memory.md): How to use `with-torch` effectively.
- [**Performance & Profiling**](docs/profiling.md): Tips for high-speed training and leak detection.

---

## Repository Development

### 1. Start the nREPL Server

Start the development nREPL server (this automatically configures the JavaCPP platform for your OS/Hardware):

```bash
clj -M:dev
```

*   **Default Port**: `7891`
*   **Environment Variables**: 
    *   `CLORCH_NREPL_PORT`: Customize the bind port.
    *   `CLORCH_NREPL_BIND`: Customize the bind address (default `127.0.0.1`).

### 2. Connect Your Editor

Connect your favorite editor (Cursive, Calva, Conjure) to the running nREPL at `localhost:7891`.

### 3. Usage in REPL

```clojure
(require '[clorch.torch :as t] :reload)
(t/randn [2 3])
```

---

## Running Examples

Check out the `examples/` directory for full verified implementations:
*   **[Autograd Tutorial](examples/autograd_tutorial.clj)**: Basic math and gradients.
*   **[PyTorch Basics](examples/pytorch_basics_tutorial.clj)**: Comprehensive port of official PyTorch "Learn the Basics".
*   **[Modern Llama](examples/modern_llama.clj)**: Llama-3 style components (RoPE, GQA, SwiGLU).
*   **[Synthetic Training](examples/synthetic.clj)**: A complete training loop from data to optimization.

---

## Numerical Accuracy

Clorch is verified against PyTorch (Python) via a rigorous cross-platform comparison suite. To run parity tests:

```bash
./run_comparison.sh
```

Operations covered by the parity suite match their Python reference outputs within the configured tolerances.

---

## CPU and GPU

Clorch sets the JavaCPP platform extension automatically when `clorch.torch` loads:
- Uses GPU (`-gpu`) only when GPU natives are on classpath and NVIDIA hardware is detected.
- Falls back to CPU otherwise.

This same auto-selection also runs when users load Clorch as a dependency (for example, requiring `clorch.torch` in their own app).

Optional overrides:
- `CLORCH_FORCE_CPU=1` forces CPU mode.
- `CLORCH_FORCE_GPU=1` forces GPU mode.

Backend selection only determines whether CUDA operations are available. Tensors and models remain on CPU unless your code places them on CUDA:

```clojure
(require '[clorch.cuda :as cuda]
         '[clorch.torch :as t]
         '[clorch.nn :as nn])

(def device (if (cuda/available?) :cuda :cpu))
(def model-on-device (nn/to model device))
(def input (t/randn [32 128] {:device device}))
```

This mirrors PyTorch: resolve one application-level device, then place models and data explicitly. `CLORCH_FORCE_GPU=1` bypasses detection but cannot compensate for missing CUDA natives, hardware, or a compatible NVIDIA driver.
