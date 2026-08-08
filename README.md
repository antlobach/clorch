# Clorch

<p align="center">
  <img src="clorch-logo-v2.png" alt="Clorch logo" width="760">
</p>

<p align="center">
  <a href="https://github.com/antlobach/clorch/actions/workflows/ci.yml"><img src="https://github.com/antlobach/clorch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://antlobach.github.io/clorch/"><img src="https://img.shields.io/badge/docs-online-4051B5?logo=materialformkdocs&logoColor=white" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-4c1.svg" alt="MIT license"></a>
</p>

Clorch is a Clojure deep-learning library backed by LibTorch. It provides PyTorch-style tensors, autograd, neural-network modules, optimizers, data loading, and explicit CPU/CUDA device placement through a REPL-friendly API.

## Highlights

- **Tensor and training APIs:** tensor operations, autograd, losses, optimizers, data loaders, state dictionaries, and native-memory scopes
- **Model building:** standard layers, custom `nn/defmodel` modules, architecture summaries, and checkpoint loading
- **LLM components:** RMSNorm, RoPE, grouped-query attention, SwiGLU, causal masks, KV caches, and autoregressive generation
- **NanoChat-inspired example:** a compact, single-device Llama-style training and chat demo with checkpointing and optional KV caching, inspired by [Karpathy's NanoChat](https://github.com/karpathy/nanochat)
- **CPU and CUDA:** native LibTorch execution with explicit tensor and model placement

## Install

Create a Clojure project with this `deps.edn`:

```clojure
{:paths ["src"]
 :deps {io.github.antlobach/clorch
        {:git/tag "v0.1.0"
         :git/sha "25a6b1005b7ada7259aaca680e9507d7eb4b03ac"}}}
```

Start a REPL from the project directory:

```bash
clj
```

## Quick start

### Tensors and autograd

```clojure
(require '[clorch.torch :as t]
         '[clorch.autograd :as autograd])

(def x (t/tensor [2.0] {:requires-grad true}))
(def y (t/pow x 3))

(autograd/backward y)
(autograd/grad x) ; => [12.0]
```

### Neural networks

```clojure
(require '[clorch.nn :as nn])

(def model
  (nn/sequential
    (nn/linear 10 20)
    (nn/relu)
    (nn/linear 20 1)))

(nn/forward model (t/randn [4 10]))
```

### Custom models with `defmodel`

```clojure
(require '[clorch.nn.functional :as F])

(nn/defmodel CustomMLP [in hidden out]
  [l1 (nn/linear in hidden)
   l2 (nn/linear hidden out)]
  (forward [x]
    (nn/forward l2 (F/relu (nn/forward l1 x)))))

(def custom-model (CustomMLP 10 32 1))
(t/size (nn/forward custom-model (t/randn [4 10]))) ; => [4 1]
```

Inspect the custom model's layer shapes and parameter counts:

```clojure
(nn/summary custom-model [4 10])
```

```text
----------------------------------------------------------------
Layer (type)                   Output Shape         Param #
================================================================
Linear                         [4 32]               352
Linear                         [4 1]                33
CustomMLPRecord                [4 1]                385
================================================================
Total params: 385
Trainable params: 385
Non-trainable params: 0
----------------------------------------------------------------
```

The constructor arguments configure the model, the binding vector registers modules or parameters, and the `forward` form defines execution. Registered fields participate in `nn/parameters`, `nn/to`, and state dictionaries. Read [Custom Models with `defmodel`](docs/nn.md#custom-models-with-defmodel) or the [minimal source example](examples/simple.clj).

## CPU and CUDA

Clorch selects the native backend when `clorch.torch` loads. It uses CUDA when GPU natives are available and NVIDIA hardware is detected; otherwise it loads the CPU backend.

Backend selection does not move tensors or models. Choose a device once, then place models and data on it:

```clojure
(require '[clorch.cuda :as cuda])

(def device (if (cuda/available?) :cuda :cpu))
(def model-on-device (nn/to model device))
(def input (t/randn [32 10] {:device device}))
```

Environment overrides:

- `CLORCH_FORCE_CPU=1` forces the CPU backend.
- `CLORCH_FORCE_GPU=1` requests the CUDA backend. It still requires compatible natives, hardware, and an NVIDIA driver.

Clorch currently supports single-device training. Distributed training, DDP, FSDP, collectives, and distributed checkpointing are not implemented.

## Native memory

LibTorch allocates tensors outside the JVM heap. The JVM garbage collector cannot measure CPU native memory or CUDA VRAM pressure. Keep models and optimizers outside repeated loops and wrap each allocating batch or generation step in `with-torch`:

```clojure
(doseq [batch dataloader]
  (let [loss-value
        (t/with-torch
          (let [loss (train-step model optimizer batch)]
            (t/item-float loss)))]
    (println "Loss:" loss-value)))
```

Return a JVM scalar or `nil` from the scope unless a tensor must escape. Read [Memory Management](docs/memory.md) for retention, manual release, and long-lived REPL sessions.

## Documentation

Full documentation: **[antlobach.github.io/clorch](https://antlobach.github.io/clorch/)**

| Guide | Covers |
|---|---|
| [Tensors and operations](docs/tensors.md) | Creation, dtypes, shapes, math, and reductions |
| [Slicing and indexing](docs/slicing.md) | `ix`, ranges, masks, and advanced indexing |
| [Autograd](docs/autograd.md) | Gradients, backward passes, detach, and no-grad scopes |
| [Neural networks](docs/nn.md) | Modules, custom models, layers, and summaries |
| [Functional API](docs/functional.md) | Stateless layers, activations, pooling, and losses |
| [Optimizers](docs/optimizers.md) | SGD, Adam, AdamW, RMSprop, and Adagrad |
| [Memory management](docs/memory.md) | Native allocation scopes and pointer lifecycles |
| [PyTorch parity](docs/pytorch-parity.md) | Measured coverage, current gaps, and roadmap |

## AI coding agents

Machine-readable project index: [`llms.txt`](https://antlobach.github.io/clorch/llms.txt)

Clorch follows PyTorch concepts, but it does not expose every PyTorch symbol. Check the linked API documentation or source before translating a Python call.

| PyTorch concept | Clorch namespace |
|---|---|
| `torch` | `clorch.torch` |
| `torch.cuda` | `clorch.cuda` |
| `torch.autograd` | `clorch.autograd` |
| `torch.nn` | `clorch.nn` |
| `torch.nn.functional` | `clorch.nn.functional` |
| `torch.optim` | `clorch.optim` |
| `torch.distributions` | `clorch.distributions` |
| `torch.linalg` | `clorch.linalg` |
| `Dataset` / `DataLoader` | `clorch.data` |

When generating Clorch code:

- Resolve one application-level device, then place models and input tensors explicitly.
- Keep long-lived models and optimizers outside `with-torch`; use one scope per allocating batch or generation step.
- Return a JVM scalar or `nil` from `with-torch` unless a tensor must escape.
- Reuse patterns from the [examples](examples/) instead of inventing wrapper APIs.
- Check [PyTorch Parity](docs/pytorch-parity.md) before assuming support for distributed training, AMP, Flash Attention, or `torch.compile`.

## Examples

- [PyTorch basics](examples/pytorch_basics_tutorial.clj): tensors, datasets, models, optimization, and inference
- [Autograd tutorial](examples/autograd_tutorial.clj): gradients and graph behavior
- [Synthetic training](examples/synthetic.clj): end-to-end model training
- [Modern Llama](examples/modern_llama.clj): RoPE, GQA, SwiGLU, and KV caching
- [Compact Llama chat](examples/nanochat.clj): small-corpus training, checkpointing, and token generation

## Project status

The cross-language comparison suite currently passes 40 of 40 numerical scenarios. The repository's tracked feature catalog marks 268 of approximately 328 capabilities as implemented, or about 81.7%. That catalog does not enumerate the entire upstream PyTorch API.

Read [PyTorch Parity](docs/pytorch-parity.md) for the capability table and roadmap to 100% of a version-pinned target.

## Development

Start the repository nREPL:

```bash
clj -M:dev
```

Run the CPU release suite:

```bash
clojure -Sthreads 1 -M -m clorch.release-check --mode cpu
```

Run all shipped examples:

```bash
scripts/run-examples.sh
```

The nREPL listens on `127.0.0.1:7891` by default. Set `CLORCH_NREPL_PORT` or `CLORCH_NREPL_BIND` to override it.

## License

Clorch is available under the [MIT License](LICENSE).
