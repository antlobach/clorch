# Memory Management

Clorch tensors are JavaCPP wrappers around LibTorch objects allocated outside the JVM heap. Tensor storage may live in native CPU RAM or CUDA VRAM.

## JVM GC and native memory

The JVM garbage collector can eventually reclaim an unreachable Clorch tensor: JavaCPP attaches a native deallocator to the JVM wrapper. The JVM cannot, however, see the size or pressure of the underlying LibTorch allocation. A small wrapper may own gigabytes of native RAM, and CUDA VRAM pressure does not directly trigger JVM collection.

This makes ordinary GC suitable for small scripts, occasional operations, and long-lived objects. Repeated tensor-producing loops need deterministic scopes so native memory does not grow until a delayed collection or native out-of-memory error.

## When to use `with-torch`

| Workload | Recommendation |
|---|---|
| Small REPL expression or short script | GC is usually sufficient |
| Long-lived model, optimizer, or dataset | Keep it outside iteration scopes |
| Training or inference batch loop | Use one `with-torch` per iteration |
| Autoregressive generation or MCMC loop | Use one `with-torch` per step |
| Large CPU tensors | Use `with-torch` around temporary computation |
| CUDA workloads | Strongly prefer deterministic scopes |
| Interactive session with many expressions | Use `start-session!` and `stop-session!` |

Users do not need `with-torch` around every operation. Put it around the smallest repeated unit that creates temporary tensors.

## Canonical training loop

Create model and optimizer once. Scope only batch-local outputs, losses, and intermediates. Finish the scope with a JVM scalar or `nil` when no tensor should escape:

```clojure
(require '[clorch.torch :as t]
         '[clorch.nn :as nn]
         '[clorch.nn.functional :as F]
         '[clorch.autograd :as autograd]
         '[clorch.optim :as optim])

(let [model (create-model)
      optimizer (optim/adam (nn/parameters model))]
  (doseq [{:keys [data target]} dataloader]
    (let [loss-value
          (t/with-torch
            (optim/zero-grad optimizer)
            (let [prediction (nn/forward model data)
                  loss (F/cross-entropy prediction target)]
              (autograd/backward loss)
              (optim/step optimizer)
              (t/item-float loss)))]
      (println "Loss:" loss-value))))
```

See the complete runnable [`pytorch_basics_tutorial.clj`](https://github.com/antlobach/clorch/blob/main/examples/pytorch_basics_tutorial.clj) and [`synthetic.clj`](https://github.com/antlobach/clorch/blob/main/examples/synthetic.clj) examples for this pattern.

## How `with-torch` works

```clojure
(t/with-torch
  (let [a (t/randn [1000 1000])
        b (t/randn [1000 1000])]
    (t/matmul a b)))
```

`with-torch` opens a JavaCPP `PointerScope`. Pointers created in the block attach to that scope. Before closing it, Clorch recursively retains pointers found in the final result. Unreturned temporaries are released when the scope closes; returned tensors remain valid and become GC-managed.

Returned maps, records, and collections are traversed through their values:

```clojure
(t/with-torch
  {:prediction prediction
   :attention attention})
```

### Avoid accidentally retaining an ignored tensor

This returns and therefore retains one tensor per iteration before `doseq` discards it:

```clojure
(doseq [batch dataloader]
  (t/with-torch
    (compute-loss batch)))
```

Return a JVM value or `nil` instead:

```clojure
(doseq [batch dataloader]
  (t/with-torch
    (let [loss (compute-loss batch)]
      (println (t/item-float loss))
      nil)))
```

## Explicit retention

`retain!` keeps a pointer alive after its current scope closes. Use it when a tensor escapes through state that `with-torch` cannot discover, such as an atom, closure, cache, or arbitrary Java object:

```clojure
(def saved (atom nil))

(t/with-torch
  (let [x (t/randn [5])]
    (t/retain! x)
    (reset! saved x)
    nil))
```

Returning the tensor directly is simpler when possible:

```clojure
(reset! saved
        (t/with-torch
          (t/randn [5])))
```

`rescue-pointers!` is an alias for `retain!`.

## Manual release

`release!` immediately deallocates a pointer or recursively releases pointers in map values and collections:

```clojure
(def x (t/randn [100 100]))
(t/release! x)
```

After release, the wrapper is invalid and must not be used. Aliases to the same wrapper are invalid too. Prefer lexical `with-torch` scopes unless ownership is unambiguous.

## Interactive session scopes

`start-session!` opens a long-lived scope for the current thread. `stop-session!` closes it and releases pointers created in that session:

```clojure
(t/start-session!)

(try
  (do-repl-experiments)
  (finally
    (t/stop-session!)))
```

Starting a new session closes any existing session on that thread. Session scopes are thread-local; worker threads need their own scopes.

## Forcing a GC pass

`gc!` calls `System/gc` and `System/runFinalization`:

```clojure
(t/gc!)
```

Use it for interactive diagnostics or recovery, not as normal loop memory management. It cannot release reachable tensors, and JVM configurations may delay or ignore explicit GC requests.

## Lifecycle API

| Function | Effect |
|---|---|
| `with-torch` | Releases block-local pointers except pointers reachable from final result |
| `retain!` | Keeps a pointer or pointers in a collection alive across scope closure |
| `rescue-pointers!` | Alias for `retain!` |
| `release!` | Immediately invalidates and deallocates owned pointers |
| `start-session!` | Opens a thread-local interactive pointer scope |
| `stop-session!` | Closes current thread's interactive scope |
| `gc!` | Requests JVM GC and finalization |

## Rules of thumb

1. Keep model, optimizer, and intentionally long-lived tensors outside iteration scopes.
2. Use one `with-torch` per allocating batch, generation step, or sampler step.
3. End a scope with a scalar or `nil` unless a tensor must escape.
4. Use `retain!` for pointers hidden inside side effects.
5. Use `release!` only with clear ownership.
6. Do not depend on `gc!` for steady-state memory bounds.
