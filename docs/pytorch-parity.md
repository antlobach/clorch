# PyTorch Parity

Clorch follows PyTorch's tensor, autograd, module, optimizer, data, and device concepts while presenting them as idiomatic Clojure APIs. Parity has two separate measurements: behavior of implemented operations and breadth of the API surface.

## Current parity

| Measurement | Current result | Meaning |
|---|---:|---|
| Cross-language numerical parity suite | **100% (40/40 scenarios)** | Every currently paired Clorch/Python scenario passes within its configured numerical tolerance. |
| Tracked feature-catalog coverage | **Pending recount after the PyTorch 2.10 upgrade** | `PORTING_STATUS.md` records the implemented surfaces, but the prior 268/328 estimate predates AMP and distributed training. |
| Entire upstream PyTorch public API | **Not yet enumerated** | A whole-project percentage requires a version-pinned inventory of every upstream public symbol and behavior. |

The 100% figure applies only to the 40 cross-language scenarios exercised by `tests_comparison/compare_torch.py`. Clorch will publish another breadth percentage after generating a version-pinned PyTorch 2.10 inventory.

The release suite verifies CPU behavior and a single-GPU CUDA path, including CUDA discovery, NCCL world size one, DDP backward, AMP overflow handling, fused scaled-dot-product attention, checkpoints, worker failures, and process cleanup. Multi-rank validation requires a host with at least two visible NVIDIA GPUs.

## Capability comparison

| PyTorch area | Clorch today | Major work remaining for full parity |
|---|---|---|
| Tensor creation, dtypes, devices | Dense tensors; common numeric, boolean, half, bfloat16, and complex dtypes; CPU/CUDA placement | Complete upstream dtype/device overload matrix, meta tensors, nested tensors, and every layout option |
| Core tensor math | Approximately 170 tracked operations covering arithmetic, transcendental functions, reductions, broadcasting, shape manipulation, and sampling | Remaining long-tail operators, overloads, out variants, named tensors, and exact edge-case/error parity |
| Indexing and slicing | Python-style `ix`, negative indices, ellipsis, stepped slices, tensor indices, masks, gather, select, and scatter-reduce | Full advanced-indexing mutation parity, sparse/nested indexing, and every upstream indexing edge case |
| Linear algebra | Matrix products, decompositions, solving, inverse, determinant, eigen, SVD, QR, Cholesky, and norms | Remaining `torch.linalg` routines, batched edge cases, driver options, and complete complex-number coverage |
| Autograd | `requires-grad`, `backward`, gradient access, detach, and `no-grad` | Custom autograd functions, `gradcheck`, forward-mode AD, inference mode, hooks, anomaly detection, and full graph-control APIs |
| Neural-network modules | Linear, convolution, transposed convolution, normalization, pooling, padding, recurrent layers, embeddings, activations, dropout, and custom `defmodel` modules | Native transformer modules, complete container modules, hooks, parametrizations, pruning, lazy modules, and remaining specialized layers |
| Functional API and losses | Core linear/convolutional functions, activations, normalization, pooling, interpolation, padding, and common classification/regression losses | Remaining functional operators and full loss/reduction/weighting option parity |
| Optimizers | SGD, Adam, AdamW, RMSprop, Adagrad, zeroing, stepping, and native checkpoint restore | Remaining optimizers, learning-rate schedulers, public optimizer state dictionaries, parameter-group mutation, and hooks |
| Data loading | Dataset protocol, tensor datasets, batching, shuffling, thread/process workers, and deterministic distributed samplers | Iterable datasets, DataPipes, collator breadth, pin-memory semantics, and complete worker lifecycle parity |
| Serialization and JIT | Tensor/model save-load, state dictionaries, weight save-load, JIT load-save-forward, and atomic rank-zero training checkpoints | `torch.export`, compilation, tracing breadth, package APIs, ONNX export, CUDA RNG checkpointing, and complete archive compatibility |
| CUDA | Automatic backend selection, availability, device count and selection, synchronization, seeding, explicit placement, autocast, dynamic gradient scaling, and fused scaled-dot-product attention | Streams, events, graphs, memory/allocator controls, peer access, and the remaining multi-GPU APIs |
| Sparse and quantized computation | Basic quantized and complex dtype exposure | Sparse layouts and operators, complete quantization workflows, observers, prepared/converted modules, and quantized kernels |
| Distributed training | Direct NCCL collectives, local rank launcher, distributed samplers, synchronous DDP, gradient accumulation, failure propagation, and rank-zero checkpoints | Gloo, RPC, FSDP, tensor parallelism, elastic membership, unused-parameter discovery, multi-node launcher support, and broader multi-GPU validation |
| Compiler and performance stack | Native LibTorch execution and explicit native-memory scopes | `torch.compile`-equivalent graph capture, Dynamo/Inductor-style compilation, profiler breadth, and benchmark tooling |
| Domain libraries | Core deep-learning library only | TorchVision, TorchAudio, TorchText, TorchData, and ecosystem-specific model/data APIs |

## LLM-relevant architecture

| Capability | Status | Clorch surface or example |
|---|---|---|
| Token embeddings | Implemented | `nn/embedding`, `nn/embedding-from-pretrained` |
| RMSNorm | Implemented | `nn/rmsnorm` |
| Rotary position embeddings | Implemented | `torch/precompute-rope-freqs`, `torch/apply-rope` |
| Grouped-query attention | Implemented | `nn/GroupedQueryAttention` |
| SwiGLU feed-forward block | Implemented | `nn/SwiGLU` |
| Causal attention masks | Implemented | Tensor masking operations and Llama/GPT examples |
| KV cache | Implemented in Llama-style model flow | `examples/modern_llama.clj`, `examples/nanochat.clj` |
| Autoregressive generation | Implemented | `nn/generate` and the NanoChat generation loop |
| GPT-style transformer construction | Implemented with custom modules | `examples/llms_from_scratch.clj` |
| Llama-style blocks | Implemented with custom modules | `examples/modern_llama.clj`, `examples/nanochat.clj` |
| Flash/scaled-dot-product attention kernels | Implemented | `F/scaled-dot-product-attention` uses LibTorch's fused dispatcher |
| Mixed-precision training | Implemented for CUDA/CPU autocast and dynamic gradient scaling | `clorch.amp` |
| Quantized LLM inference | Not complete | Add end-to-end weight/activation quantization and quantized kernels |
| Distributed LLM training/inference | Partial | NCCL DDP training exists; add FSDP, tensor parallelism, RPC, and multi-node orchestration |

## Roadmap to 100%

1. **Freeze the target.** Select an exact upstream PyTorch release and generate a machine-readable inventory of its supported public Python and C++ APIs.
2. **Turn inventory into contracts.** Map every upstream symbol to implemented, partial, intentionally different, or missing; attach behavior tests to every implemented mapping.
3. **Close core tensor gaps.** Finish dense overloads and edge cases, then sparse, nested, meta, quantized, and named-tensor behavior.
4. **Complete autograd and modules.** Add custom functions, grad checking, hooks, inference controls, native transformer/container modules, and remaining functional/loss APIs.
5. **Complete optimization and data.** Add public optimizer state, schedulers, remaining algorithms, iterable data loading, DataPipes, and worker/pinning parity.
6. **Complete accelerator support.** Add CUDA streams, events, graphs, allocator controls, Gloo, FSDP, tensor parallelism, elastic execution, and broader multi-GPU conformance tests.
7. **Complete export and ecosystem surfaces.** Add compile/export/ONNX/package behavior and explicitly scope domain libraries.
8. **Prove the result.** Run generated cross-language conformance tests across Linux, macOS, Windows, CPU, CUDA, supported dtypes, and error/edge-case behavior.

Until that inventory exists, report the measured result without inventing breadth precision: **100% of 40 tested numerical scenarios; tracked catalog percentage pending recount**.
