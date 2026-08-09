(ns distributed-training
  "Runnable local multi-GPU DDP training example."
  (:require [clorch.amp :as amp]
            [clorch.data :as data]
            [clorch.distributed :as dist]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.nn.parallel :as ddp]
            [clorch.optim :as optim]
            [clorch.torch :as t])
  (:import [org.bytedeco.pytorch Scalar Tensor]))

(defn- training-example [sample-index sample-count]
  (let [x (/ (double sample-index) (double (max 1 (dec sample-count))))
        features [x (* x x) (Math/sin (* Math/PI x)) (Math/cos (* Math/PI x))]
        target [(+ (* 0.3 x) (* -0.2 x x) (* 0.5 (Math/sin (* Math/PI x))))]]
    [features target]))

(defn- batch-tensors [indices sample-count]
  (let [examples (mapv #(training-example % sample-count) indices)]
    {:input (t/to (t/tensor (mapv first examples) {:dtype :float32}) :cuda)
     :target (t/to (t/tensor (mapv second examples) {:dtype :float32}) :cuda)}))

(defn- scaled-loss [loss divisor]
  (with-open [value (Scalar. (double divisor))]
    (.div ^Tensor loss value)))

(defn train-worker
  "Worker entrypoint used by `run-local!`; one invocation per GPU."
  [{:keys [rank world-size process-group args]}]
  (let [{:keys [epochs sample-count batch-size accumulation precision checkpoint-path]
         :or {epochs 2
              sample-count 256
              batch-size 16
              accumulation 1
              precision :bfloat16}} args
        _ (t/manual-seed 1337)
        sampler (data/distributed-sampler
                 sample-count {:num-replicas world-size :rank rank :seed 1337})
        model (nn/to (nn/linear 4 1) :cuda)
        optimizer (optim/adamw (nn/parameters model) :lr 0.01)
        scaler (amp/grad-scaler {:enabled? (= precision :float16)})
        global-step (atom 0)
        last-loss (atom nil)]
    (with-open [parallel-model
                (ddp/distributed-data-parallel
                 model {:process-group process-group :bucket-cap-mb 25.0})]
      (dotimes [epoch epochs]
        (data/set-epoch! sampler epoch)
        (let [batches (partition-all batch-size (data/sample-indices sampler))]
          (doseq [micro-batches (partition-all accumulation batches)]
            (optim/zero-grad optimizer)
            (doseq [[micro-index indices] (map-indexed vector micro-batches)]
              (let [final-micro? (= micro-index (dec (count micro-batches)))
                    train-micro!
                    (fn []
                      (t/with-torch
                        (let [{:keys [input target]}
                              (batch-tensors indices sample-count)
                              loss (amp/autocast
                                    {:device :cuda :dtype precision}
                                    (F/mse-loss (nn/forward parallel-model input)
                                                target))]
                          (reset! last-loss (t/item-float loss))
                          (amp/backward! scaler
                                         (scaled-loss loss (count micro-batches))))))]
                (if final-micro?
                  (train-micro!)
                  (ddp/no-sync (train-micro!)))))
            (ddp/optimizer-step! parallel-model optimizer {:scaler scaler})
            (swap! global-step inc))))
      (when checkpoint-path
        (dist/save-checkpoint!
         process-group checkpoint-path
         {:model model
          :optimizer optimizer
          :sampler sampler
          :scaler scaler
          :state {:epoch epochs :global-step @global-step}}))
      (let [model-parameters (t/->vector (nn/parameters model))
            rank-zero-parameters (mapv #(.clone ^Tensor %) model-parameters)]
        (dist/broadcast! process-group rank-zero-parameters {:root-rank 0})
        {:rank rank
         :steps @global-step
         :loss @last-loss
         :precision precision
         :parameters-synchronized?
         (every? true?
                 (map #(.allclose ^Tensor %1 ^Tensor %2)
                      model-parameters rank-zero-parameters))}))))

(defn run-local!
  "Launches one training worker JVM per CUDA device and waits for completion."
  [devices & [options]]
  (let [job (dist/launch!
             {:nproc-per-node (count devices)
              :devices devices
              :main 'distributed-training/train-worker
              :args (or options {})})]
    (dist/await-job! job)
    {:status (dist/job-status job)
     :logs (into {}
                 (map (fn [rank] [rank (dist/job-logs job rank)]))
                 (range (count devices)))}))
