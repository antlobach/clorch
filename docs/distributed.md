# Distributed CUDA Training

Clorch runs one JVM per CUDA device and coordinates ranks with NCCL. The public API covers process-group lifecycle, collectives, local worker launch, distributed sampling, synchronous data parallelism, AMP, and rank-zero checkpoints.

## Requirements

- Linux with NVIDIA GPUs and a working driver
- Java 25 or newer for the managed launcher
- One distinct CUDA device per local rank
- GPU-enabled JavaCPP PyTorch and CUDA native libraries on the classpath

Clorch currently supports the `:nccl` backend. It does not provide Gloo, RPC, FSDP, tensor parallelism, or elastic membership.

## Run the training example

The shipped example launches one worker JVM for each device:

```clojure
(require '[distributed-training :as training])

(training/run-local!
 [0 1]
 {:epochs 4
  :sample-count 1024
  :batch-size 32
  :accumulation 2
  :precision :bfloat16
  :checkpoint-path "/tmp/clorch-ddp.pt"})
```

`run-local!` waits for every rank and returns job status plus file-backed stdout and stderr. Use `:float16` to enable dynamic loss scaling. Bfloat16 uses autocast without a scaler.

## Worker entrypoints

`dist/launch!` accepts a namespace-qualified worker function and EDN arguments:

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

Clorch invokes the worker with this map:

```clojure
{:rank 0
 :local-rank 0
 :world-size 2
 :backend :nccl
 :process-group context
 :args {:epochs 10}}
```

The launcher sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, and `CUDA_VISIBLE_DEVICES` for each child. If one rank exits with an error, the launcher terminates the remaining ranks and preserves each rank's logs on disk. Call `dist/stop-job!` to cancel a running job.

## Process groups and collectives

Workers launched by Clorch receive an initialized process group. For a custom launcher, initialize from the standard environment variables:

```clojure
(require '[clorch.distributed :as dist])

(dist/with-process-group {:backend :nccl}
  (dist/all-reduce! tensor {:op :sum})
  (dist/barrier!))
```

Available collective operations:

- `all-reduce!`, `broadcast!`, and `reduce!`
- `all-gather-into!` and `reduce-scatter-into!`
- `all-to-all-single!`
- `send` and `receive!`
- `barrier!`

Collectives mutate CUDA tensors in place. Every rank must call collectives in the same order with compatible shapes, dtypes, and split sizes. Pass `:async? true` to receive a work handle, then call `dist/await!` before reading the result.

## DistributedDataParallel

Create the model on the rank-local CUDA device before wrapping it:

```clojure
(require '[clorch.nn :as nn]
         '[clorch.nn.parallel :as ddp]
         '[clorch.optim :as optim])

(def model (nn/to (nn/linear 128 32) :cuda))
(def optimizer (optim/adamw (nn/parameters model) :lr 3e-4))

(with-open [parallel-model
            (ddp/distributed-data-parallel
             model {:bucket-cap-mb 25.0
                    :broadcast-buffers? true})]
  ;; forward, backward, then:
  (ddp/optimizer-step! parallel-model optimizer))
```

The constructor verifies parameter signatures across ranks and broadcasts parameters from rank zero. Backward hooks copy gradients into reusable buckets and start asynchronous all-reduce operations. `ddp/optimizer-step!` waits for pending reductions before stepping the optimizer.

Each synchronized backward pass must produce a gradient for every trainable parameter. `:find-unused-parameters? true` and `:gradient-as-bucket-view? true` are unsupported and fail during construction.

### Gradient accumulation

Use `ddp/no-sync` around every micro-batch except the last one:

```clojure
(doseq [[index batch] (map-indexed vector micro-batches)]
  (let [train! #(train-micro-batch! parallel-model batch)]
    (if (= index (dec (count micro-batches)))
      (train!)
      (ddp/no-sync (train!)))))

(ddp/optimizer-step! parallel-model optimizer)
```

## Distributed sampling

Each rank needs a disjoint, deterministic sample stream:

```clojure
(require '[clorch.data :as data])

(def sampler
  (data/distributed-sampler
   dataset-size
   {:num-replicas (dist/world-size)
    :rank (dist/rank)
    :seed 1337
    :shuffle? true
    :drop-last? false}))

(data/set-epoch! sampler epoch)
(doseq [indices (partition-all batch-size (data/sample-indices sampler))]
  (train-batch! indices))
```

Call `set-epoch!` before each epoch. Otherwise each epoch reuses the same permutation. Sampler state includes epoch, seed, rank, replica count, dataset size, shuffle mode, and drop-last mode.

## Mixed precision

```clojure
(require '[clorch.amp :as amp])

(def scaler (amp/grad-scaler {:initial-scale 65536.0}))

(let [loss (amp/autocast {:device :cuda :dtype :float16}
             (compute-loss parallel-model batch))]
  (amp/backward! scaler loss)
  (ddp/optimizer-step! parallel-model optimizer {:scaler scaler}))
```

`amp/step!` unscales gradients, checks finite values across all initialized ranks, skips the optimizer step on overflow, and updates the scale. Autocast restores the thread's previous dtype, enabled flag, and cache setting after the body exits.

## Checkpoints

```clojure
(dist/save-checkpoint!
 "/checkpoints/model.pt"
 {:model model
  :optimizer optimizer
  :sampler sampler
  :scaler scaler
  :state {:epoch epoch :global-step step}})

(def training-state
  (dist/load-checkpoint!
   "/checkpoints/model.pt"
   {:model model
    :optimizer optimizer
    :sampler sampler
    :scaler scaler}))
```

Rank zero writes the tensor archive and EDN metadata through temporary files, then commits them with atomic moves. Every rank restores model weights, optimizer state, CPU RNG state, sampler state, scaler state, and the supplied EDN training state. Clorch does not capture CUDA generator state; store and reapply your CUDA seed when exact CUDA random-stream replay matters.

## Current verification scope

The release suite covers CPU behavior and a single-GPU CUDA path, including NCCL world size one, DDP backward, AMP overflow handling, fused scaled-dot-product attention, checkpoints, worker failures, and process cleanup. Two-rank execution requires a machine with at least two visible NVIDIA GPUs and belongs in release validation for changes to collectives or DDP.
