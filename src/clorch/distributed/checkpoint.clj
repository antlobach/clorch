(ns clorch.distributed.checkpoint
  "Atomic rank-zero checkpoints for distributed training."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clorch.distributed :as dist]
            [clorch.torch :as t])
  (:import [java.nio.file AtomicMoveNotSupportedException CopyOption Files
            StandardCopyOption]
           [java.util UUID]
           [org.bytedeco.pytorch Generator]
           [org.bytedeco.pytorch.global torch]))

(def ^:private checkpoint-version 1)

(defn- metadata-path [path]
  (str path ".edn"))

(defn- ensure-edn! [value]
  (let [encoded (pr-str value)]
    (when-not (= value (edn/read-string encoded))
      (throw (ex-info "Checkpoint training state must round-trip through EDN"
                      {:state value})))
    encoded))

(defn- atomic-move! [source destination]
  (let [source-path (.toPath (io/file source))
        destination-path (.toPath (io/file destination))]
    (try
      (Files/move source-path destination-path
                  (into-array CopyOption
                              [StandardCopyOption/ATOMIC_MOVE
                               StandardCopyOption/REPLACE_EXISTING]))
      (catch AtomicMoveNotSupportedException _
        (Files/move source-path destination-path
                    (into-array CopyOption
                                [StandardCopyOption/REPLACE_EXISTING]))))))

(defn- sampler-state [sampler]
  (when sampler
    ((requiring-resolve 'clorch.data/sampler-state) sampler)))

(defn- archive-values [model optimizer cpu-rng-state]
  (cond-> {:model model
           :cpu-rng-state cpu-rng-state}
    optimizer (assoc :optimizer optimizer)))

(defn save!
  "Writes model, optimizer, CPU RNG, sampler, scaler, and EDN training state on rank zero.

  The tensor archive is `path`; metadata is committed last to `path.edn`, so a
  missing metadata file always denotes an incomplete checkpoint."
  [context path {:keys [model optimizer sampler scaler state]}]
  (when-not model
    (throw (ex-info "Checkpoint requires :model" {})))
  (let [context (or context (dist/current-process-group))
        metadata {:version checkpoint-version
                  :world-size (:world-size context)
                  :sampler (sampler-state sampler)
                  :scaler (when scaler
                            ((requiring-resolve 'clorch.amp/scaler-state) scaler))
                  :state state}
        metadata-content (ensure-edn! metadata)]
    (dist/rank-zero-call!
     context
     (fn []
       (let [suffix (str ".tmp." (UUID/randomUUID))
             archive-temp (str path suffix)
             metadata-temp (str (metadata-path path) suffix)
             ^Generator cpu-generator (torch/getDefaultCPUGenerator)
             cpu-rng-state (.get_state cpu-generator)]
         (try
           (t/save (archive-values model optimizer cpu-rng-state)
                   archive-temp)
           (spit metadata-temp metadata-content)
           (atomic-move! archive-temp path)
           (atomic-move! metadata-temp (metadata-path path))
           (finally
             (Files/deleteIfExists (.toPath (io/file archive-temp)))
             (Files/deleteIfExists (.toPath (io/file metadata-temp))))))))
    (dist/barrier! context)
    path))

(defn load!
  "Restores model, optimizer, CPU RNG, sampler, scaler, and training state on every rank."
  [context path {:keys [model optimizer sampler scaler]}]
  (when-not model
    (throw (ex-info "Checkpoint requires :model" {})))
  (let [context (or context (dist/current-process-group))
        metadata-file (io/file (metadata-path path))]
    (when-not (.isFile metadata-file)
      (throw (ex-info "Checkpoint metadata is missing or incomplete"
                      {:path (metadata-path path)})))
    (let [metadata (edn/read-string (slurp metadata-file))]
      (when-not (= checkpoint-version (:version metadata))
        (throw (ex-info "Unsupported checkpoint version"
                        {:expected checkpoint-version
                         :actual (:version metadata)})))
      (let [^Generator cpu-generator (torch/getDefaultCPUGenerator)
            cpu-rng-state (.get_state cpu-generator)]
        (t/load (archive-values model optimizer cpu-rng-state) path)
        (.set_state cpu-generator cpu-rng-state))
      (when (and sampler (:sampler metadata))
        ((requiring-resolve 'clorch.data/load-sampler-state!)
         sampler
         (assoc (:sampler metadata)
                :rank (:rank context)
                :replicas (:world-size context))))
      (when (and scaler (:scaler metadata))
        ((requiring-resolve 'clorch.amp/load-scaler-state!)
         scaler (:scaler metadata)))
      (dist/barrier! context)
      (:state metadata))))
