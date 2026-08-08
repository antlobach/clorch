# PyTorch Parity

Clorch follows PyTorch's tensor, autograd, module, optimizer, data, and device concepts while presenting them as idiomatic Clojure APIs. Parity has two separate measurements: behavior of implemented operations and breadth of the API surface.

## Current parity

| Measurement | Current result | Meaning |
|---|---:|---|
| Cross-language numerical parity suite | **100% (40/40 scenarios)** | Every currently paired Clorch/Python scenario passes within its configured numerical tolerance. |
| Tracked feature-catalog coverage | **approximately 81.7% (268 of approximately 328 capabilities)** | Coverage of the explicit capability catalog maintained in this repository. |
| Entire upstream PyTorch public API | **Not yet enumerated** | A defensible whole-project percentage requires a version-pinned inventory of every upstream public symbol and behavior. |

The 81.7% figure is the best current breadth estimate: $268 / (268 + 60) \times 100 \approx 81.7\%$. It is not a claim that 81.7% of every symbol in every PyTorch package exists. The 100% figure applies only to the 40 cross-language scenarios currently exercised by `tests_comparison/compare_torch.py`.

The release suite also verifies CUDA discovery, GPU tensor allocation, device reporting, CUDA computation, synchronization, and transfer back to CPU on NVIDIA hardware.

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
| Optimizers | SGD, Adam, AdamW, RMSprop, Adagrad, zeroing, and stepping | Remaining optimizers, learning-rate schedulers, optimizer state dictionaries, parameter-group mutation, and hooks |
| Data loading | Dataset protocol, tensor datasets, batching, shuffling, thread workers, and process workers | Iterable datasets, distributed samplers, DataPipes, collator breadth, pin-memory semantics, and complete worker lifecycle parity |
| Serialization and JIT | Tensor/model save-load, state dictionaries, weight save-load, and JIT load-save-forward | `torch.export`, compilation, tracing breadth, package APIs, ONNX export, and complete archive compatibility |
| CUDA | Automatic native-backend selection, availability, device count, synchronization, seeding, and explicit CUDA tensor/model placement | Streams, events, graphs, AMP/autocast, memory statistics, allocator controls, peer access, and complete multi-GPU APIs |
| Sparse and quantized computation | Basic quantized and complex dtype exposure | Sparse layouts and operators, complete quantization workflows, observers, prepared/converted modules, and quantized kernels |
| Distributed training | Not implemented | DistributedDataParallel, RPC, collectives, NCCL/Gloo integration, FSDP, tensor parallelism, and distributed checkpointing |
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
| Flash/scaled-dot-product attention kernels | Not implemented as dedicated fused APIs | Add fused attention bindings and kernel selection |
| Mixed-precision training | Partial dtype support; no complete AMP workflow | Add autocast, gradient scaling, and AMP state management |
| Quantized LLM inference | Not complete | Add end-to-end weight/activation quantization and quantized kernels |
| Distributed LLM training/inference | Not implemented | Add collectives, DDP/FSDP, tensor parallelism, and distributed checkpoints |

## Roadmap to a defensible 100%

1. **Freeze the target.** Select an exact upstream PyTorch release and generate a machine-readable inventory of its supported public Python and C++ APIs.
2. **Turn inventory into contracts.** Map every upstream symbol to implemented, partial, intentionally different, or missing; attach behavior tests to every implemented mapping.
3. **Close core tensor gaps.** Finish dense overloads and edge cases, then sparse, nested, meta, quantized, and named-tensor behavior.
4. **Complete autograd and modules.** Add custom functions, grad checking, hooks, inference controls, native transformer/container modules, and remaining functional/loss APIs.
5. **Complete optimization and data.** Add optimizer state, schedulers, remaining algorithms, iterable/distributed data loading, and worker/pinning parity.
6. **Complete accelerator support.** Add CUDA streams/events/graphs, AMP, memory controls, multi-GPU execution, and distributed collectives.
7. **Complete export and ecosystem surfaces.** Add compile/export/ONNX/package behavior and explicitly scope domain libraries.
8. **Prove the result.** Run generated cross-language conformance tests across Linux, macOS, Windows, CPU, CUDA, supported dtypes, and error/edge-case behavior.

Until that version-pinned inventory exists, report both current numbers together: **100% of 40 tested numerical scenarios and approximately 81.7% of the repository's tracked feature catalog**.
