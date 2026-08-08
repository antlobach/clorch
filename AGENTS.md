# Clorch Agent Guide

## Project

Clorch is a Clojure deep-learning library backed by LibTorch through Bytedeco JavaCPP presets. It follows PyTorch concepts and naming where the binding exists, but it is not a symbol-for-symbol PyTorch port.

Before translating PyTorch code, verify the target var in `src/clorch/`, the public documentation, or an existing example. Do not invent wrapper APIs from Python names.

## API Map

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

## Runtime Invariants

- Resolve one application-level device, then place models and input tensors explicitly.
- Keep a model, optimizer, persistent masks, and other long-lived tensors outside repeated `with-torch` scopes.
- Use one `with-torch` scope per allocating training batch or generation step.
- End a scope with a JVM scalar or `nil` unless a tensor must escape.
- Use `retain!` only when a tensor must outlive its scope; release retained tensors when their owner closes.
- Clorch currently supports single-device training. Do not assume distributed training, AMP, Flash Attention, or `torch.compile` exists.

Read `docs/memory.md` before changing repeated tensor workloads and `docs/pytorch-parity.md` before claiming PyTorch coverage.

## REPL Workflow

Start the repository nREPL:

```bash
CLOJURE_DISABLE_RLWRAP=1 clojure -M:dev
```

Connect on `127.0.0.1:7891`, then evaluate changed definitions in their namespace before persisting or validating them:

```clojure
(require '[clorch.torch :as t] :reload)
(t/with-torch
  (t/item-float (t/sum (t/tensor [1.0 2.0 3.0]))))
```

Prefer small, reproducible REPL evaluations. Reload changed namespaces with `:reload`.

## Existing Patterns

- Tensor creation and operations: `src/clorch/torch.clj`
- Modules and custom `defmodel` models: `src/clorch/nn.clj`
- Stateless neural-network operations: `src/clorch/nn/functional.clj`
- Training loops: `examples/synthetic.clj` and `examples/custom_model_dataset.clj`
- LLM components: `examples/modern_llama.clj` and `examples/nanochat.clj`
- Native-memory lifecycle: `docs/memory.md`
- Current coverage and gaps: `docs/pytorch-parity.md`

Reuse these patterns instead of introducing a second convention.

## Clojure Delimiters and Formatting

Do not brute-force unbalanced delimiters. If a `.clj`, `.cljc`, `.cljs`, or `.edn` file has a delimiter error, run:

```bash
clj-paren-repair path/to/file.clj
```

If repair fails, report it and stop trying to hand-fix the delimiters. The repair tool formats repaired files.

After changing Clojure or EDN files, format them with:

```bash
clj -Sdeps '{:deps {dev.weavejester/cljfmt {:mvn/version "0.13.0"}}}' -M -m cljfmt.main fix path/to/file.clj
```

Run file-scoped clj-kondo through the installed MCP server. Fix errors and warnings in changed code. If the MCP server is unavailable, use the `clj-kondo` CLI when installed or perform a compile/reload check.

## Verification

Use the smallest check that exercises the changed contract, then run the relevant repository verification:

```bash
# CPU release suite
clojure -Sthreads 1 -M -m clorch.release-check --mode cpu

# Every shipped example
scripts/run-examples.sh

# Cross-language numerical comparison
run_comparison.sh
```

For CUDA changes, use `scripts/release-check.sh --mode gpu` on NVIDIA hardware. For documentation changes, build MkDocs with strict validation.
