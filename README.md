# Clorch

<p align="center">
  <img src="clorch-logo-v2.png" alt="Clorch logo" width="760">
</p>

<p align="center">
  <a href="https://github.com/antlobach/clorch/actions/workflows/ci.yml"><img src="https://github.com/antlobach/clorch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://antlobach.github.io/clorch/"><img src="https://img.shields.io/badge/docs-online-4051B5?logo=materialformkdocs&logoColor=white" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-4c1.svg" alt="MIT license"></a>
</p>

Clorch is a Clojure deep-learning library backed by LibTorch. It provides PyTorch-style tensors, autograd, neural-network modules, optimizers, data loading, explicit CPU/CUDA placement, and NCCL distributed training through a REPL-friendly API.

## Highlights

- **Tensor and training APIs:** tensor operations, autograd, losses, optimizers, data loaders, state dictionaries, AMP, and native-memory scopes
- **Model building:** standard layers, custom `nn/defmodel` modules, architecture summaries, and checkpoint loading
- **Distributed CUDA:** NCCL collectives, managed rank processes, distributed sampling, synchronous DDP, gradient accumulation, and rank-zero checkpoints
- **LLM components:** RMSNorm, RoPE, grouped-query attention, SwiGLU, fused scaled-dot-product attention, causal masks, KV caches, and autoregressive generation
- **NanoChat-inspired example:** a compact, single-device Llama-style training and chat demo with checkpointing and optional KV caching, inspired by [Karpathy's NanoChat](https://github.com/karpathy/nanochat)
- **CPU and CUDA:** native LibTorch execution with explicit tensor and model placement

## Install

Create a Clojure project with this `deps.edn`:

```clojure
{:paths ["src"]
 :deps {io.github.antlobach/clorch
        {:git/tag "v0.2.0"
         :git/sha "07642acdbc522e8aa2a20cd223912247614d2239"}}}
```

Start a REPL from the project directory:

```bash
clj
```

### Supported runtime

| Component | Supported version |
|---|---|
| Java | OpenJDK/Temurin 21 through 25; CI runs the full 21–25 matrix |
| Clojure | Clojure 1.12.x |
| Clojure CLI | A current 1.12.x CLI; CI uses `1.12.5.1664` |
| Native PyTorch | JavaCPP PyTorch `2.10.0-1.5.13` |

```text
Java 21 ┐
Java 22 │
Java 23 ├─ PyTorch 2.10 + JavaCPP 1.5.13 CPU natives
Java 24 │
Java 25 ┘
```

### Multi-GPU distributed training

```text
Java 25
PyTorch/LibTorch 2.10
JavaCPP 1.5.13
CUDA 13.1
cuDNN 9.19
NCCL 2.29.2
2× RTX A5000
```

Use a JDK, not a JRE. Java 24 and 25 should be started with
`--enable-native-access=ALL-UNNAMED` for JavaCPP native loading.

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

## CPU, CUDA, and multi-GPU training

Clorch selects the native backend when `clorch.torch` loads. It uses CUDA when
GPU natives are available and NVIDIA hardware is detected; otherwise it loads
the CPU backend.

Backend selection does not move tensors or models. Choose a device once, then
place models and data on it:

```clojure
(require '[clorch.cuda :as cuda])

(def device (if (cuda/available?) :cuda :cpu))
(def model-on-device (nn/to model device))
(def input (t/randn [32 10] {:device device}))
```

Environment overrides:

- `CLORCH_FORCE_CPU=1` forces the CPU backend.
- `CLORCH_FORCE_GPU=1` requests the CUDA backend. It still requires compatible
  native libraries, hardware, and an NVIDIA driver.

### Multi-GPU requirements

Distributed training uses one worker JVM per GPU and NCCL for communication.
The managed launcher covers multiple GPUs on one Linux host.

- Two or more NVIDIA GPUs visible to the same process
- A working NVIDIA driver with CUDA 13 support
- CUDA 13.1 user-space libraries, including cuBLAS
- cuDNN 9 and NCCL 2; the validated stack uses cuDNN 9.19 and NCCL 2.29.2
- Java 21 through 25 and Clojure 1.12.x
- One distinct CUDA device per rank
- Enough host RAM and CUDA VRAM for one model replica per rank
- A writable checkpoint directory and an available local TCP port

For an Ubuntu host configured with NVIDIA's CUDA package repository, the
required runtime packages are:

```bash
sudo apt-get update
sudo apt-get install cuda-libraries-13-1 libcudnn9-cuda-13 libnccl2
```

Confirm that the host exposes each GPU before starting Clojure:

```bash
nvidia-smi -L
java -version
clojure -Sdescribe
```

Set native-loading variables before the JVM starts. `JAVA_TOOL_OPTIONS` is
inherited by launcher-created worker JVMs:

```bash
export CLORCH_FORCE_GPU=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export JAVA_TOOL_OPTIONS="--enable-native-access=ALL-UNNAMED"

CLOJURE_DISABLE_RLWRAP=1 clojure -M:dev
```

Check CUDA from the REPL:

```clojure
(require '[clorch.cuda :as cuda]
         '[clorch.torch :as t])

{:available (cuda/available?)
 :devices (cuda/device-count)}
;; => {:available true, :devices 2}
```

### Launch the distributed training example

Clone this repository and start its nREPL so the `examples` path is available.
The example below launches ranks for physical devices 0 and 1, waits for both
workers, and returns status plus per-rank logs:

```clojure
(require '[distributed-training :as training])

(def result
  (training/run-local!
   [0 1]
   {:epochs 4
    :sample-count 1024
    :batch-size 32
    :accumulation 2
    :precision :bfloat16
    :checkpoint-path "/checkpoints/clorch-ddp.pt"}))
```

Use `:float16` for autocast with dynamic gradient scaling, or `:bfloat16` for
autocast without a scaler. `:accumulation 2` performs two micro-batches per
optimizer step and suppresses DDP synchronization until the final
micro-batch.

The launcher sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`,
`MASTER_ADDR`, `MASTER_PORT`, and `CUDA_VISIBLE_DEVICES` for every child. Worker
code should place its model and batches on `:cuda`; the launcher maps that
logical device to the assigned physical GPU.

For an application worker, launch a namespace-qualified function:

```clojure
(require '[clorch.distributed :as dist])

(def job
  (dist/launch!
   {:nproc-per-node 2
    :devices [0 1]
    :main 'my.training/train-worker
    :args {:epochs 10}
    :timeout-ms 300000}))

(dist/job-status job)
(dist/await-job! job)
(dist/job-logs job 0)
```

Every rank must execute collectives in the same order and train from a
disjoint `distributed-sampler` partition. Create the model on `:cuda` before
wrapping it with `distributed-data-parallel`, call `data/set-epoch!` each
epoch, and use `ddp/optimizer-step!` instead of stepping the optimizer
directly. Only rank zero writes checkpoints; all ranks participate in save and
restore barriers.

Run the GPU release check after configuring the host:

```bash
CLORCH_FORCE_GPU=1 clojure -Sthreads 1 -M -m clorch.release-check --mode gpu
```

On a host with at least two GPUs, the CUDA smoke includes a two-rank DDP check
that verifies every model parameter remains synchronized. Read
[Distributed CUDA Training](docs/distributed.md) for worker code, collectives,
sampling, AMP, gradient accumulation, failure handling, and checkpoint
restore.

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
| [Distributed training](docs/distributed.md) | NCCL collectives, worker launch, DDP, AMP, sampling, and checkpoints |
| [Memory management](docs/memory.md) | Native allocation scopes and pointer lifecycles |
| [PyTorch parity](docs/pytorch-parity.md) | Measured coverage, current gaps, and roadmap |

## AI coding agents

Machine-readable project index: [`llms.txt`](https://antlobach.github.io/clorch/llms.txt)

Clorch follows PyTorch concepts, but it does not expose every PyTorch symbol. Check the linked API documentation or source before translating a Python call.

| PyTorch concept | Clorch namespace |
|---|---|
| `torch` | `clorch.torch` |
| `torch.cuda` | `clorch.cuda` |
| `torch.amp` | `clorch.amp` |
| `torch.distributed` | `clorch.distributed` |
| `DistributedDataParallel` | `clorch.nn.parallel` |
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
- Read the [Distributed CUDA Training](docs/distributed.md) guide before generating multi-process training code.

## Examples

- [PyTorch basics](examples/pytorch_basics_tutorial.clj): tensors, datasets, models, optimization, and inference
- [Autograd tutorial](examples/autograd_tutorial.clj): gradients and graph behavior
- [Synthetic training](examples/synthetic.clj): end-to-end model training
- [Modern Llama](examples/modern_llama.clj): RoPE, GQA, SwiGLU, and KV caching
- [Compact Llama chat](examples/nanochat.clj): small-corpus training, checkpointing, and token generation
- [Distributed CUDA training](examples/distributed_training.clj): NCCL workers, DDP, AMP, accumulation, sampling, and checkpoints

## Project status

The cross-language comparison suite currently passes 40 of 40 numerical scenarios. The tracked feature catalog predates the PyTorch 2.10 distributed-training milestone and needs a version-pinned recount before Clorch publishes another breadth percentage.

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

Run the GPU release suite on a Linux NVIDIA host:

```bash
clojure -Sthreads 1 -M -m clorch.release-check --mode gpu
```

Run all shipped examples:

```bash
scripts/run-examples.sh
```

The nREPL listens on `127.0.0.1:7891` by default. Set `CLORCH_NREPL_PORT` or `CLORCH_NREPL_BIND` to override it.

## License

Clorch is available under the [MIT License](LICENSE).
