# Performance & Profiling

Clorch is designed for high-performance deep learning. Because it is built on LibTorch, it executes heavy tensor compute in C++, bypassing the overhead of the JVM for math operations.

## Performance Tips

### 1. Vectorized Operations
Always prefer vectorized tensor operations over manual loops in Clojure.

```clojure
;; FAST: Executed in C++
(t/add tensor 1.0)

;; SLOW: Crosses the JNI boundary for every element
(mapv #(+ % 1.0) (t/tseq tensor))
```

### 2. Contiguous Tensors
Certain operations like `view` require tensors to be contiguous in memory. If you encounter errors after transposing or slicing, call `contiguous`.

```clojure
(-> tensor (t/transpose 0 1) (.contiguous))
```

### 3. Using `no-grad`
During inference, always use `autograd/no-grad`. This prevents the creation of the autograd graph, significantly reducing memory usage and increasing speed.

```clojure
(autograd/no-grad
  (nn/forward model x))
```

## Memory Profiling

If you suspect a native memory leak, you can use the built-in profiling tool or JVM tools like `jcmd`.

### Using `test/profiler.clj`
Clorch includes a profiling script that monitors JVM Heap and Native RSS (Resident Set Size).

```bash
clojure -M test/profiler.clj
```

### Manual Monitoring
You can monitor the Resident Set Size (RSS) of your process to see native memory usage:

```bash
ps -o rss -p <PID>
```

If the RSS grows indefinitely while the JVM Heap remains stable, you likely have a missing `with-torch` scope or a leaked native pointer.

## JNI Overhead
Clorch uses JavaCPP to minimize JNI overhead. However, calling a native method still has a small fixed cost. For maximum performance, group your operations into larger tensor calls rather than many small ones.
