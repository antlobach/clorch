(ns clorch.cuda-smoke-test
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clorch.amp :as amp]
            [clorch.autograd :as autograd]
            [clorch.cuda :as cuda]
            [clorch.data :as data]
            [clorch.distributed :as dist]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.nn.parallel :as ddp]
            [clorch.optim :as optim]
            [clorch.torch :as torch])
  (:import [java.net ServerSocket]
           [org.bytedeco.javacpp Loader]
           [org.bytedeco.pytorch Tensor]))

(defn preload! []
  (doseq [class-name ["org.bytedeco.openblas.global.openblas_nolapack"
                      "org.bytedeco.cuda.global.nvrtc"
                      "org.bytedeco.pytorch.global.torch"]]
    (Loader/load (Class/forName class-name))))

(defn- check! [condition message data]
  (when-not condition
    (throw (ex-info message data))))

(defn- available-port []
  (with-open [socket (ServerSocket. 0)]
    (.getLocalPort socket)))

(defn- parameters [model]
  (let [values (nn/parameters model)]
    (mapv #(.get values (long %)) (range (.size values)))))

(defn- verify-launcher-failure! []
  (let [job (dist/launch! {:nproc-per-node 1
                           :devices [0]
                           :main 'clorch.cuda-smoke-test/missing-worker
                           :args {}})
        failure (try
                  (dist/await-job! job 120000)
                  nil
                  (catch clojure.lang.ExceptionInfo exception
                    (ex-data exception)))
        status (dist/job-status job)
        logs (dist/job-logs job 0)]
    (check! (= :failed (:state failure))
            "Launcher did not propagate worker failure"
            {:failure failure})
    (check! (every? (complement :alive?) (:workers status))
            "Launcher left a failed worker alive"
            {:status status})
    (check! (str/includes? (:stderr logs) "does not exist")
            "Worker logs omitted the failure diagnostic"
            {:logs logs})))

(defn run-test []
  (preload!)
  (if-not (cuda/available?)
    (do
      (println "SKIPPED: CUDA hardware is unavailable.")
      :skipped)
    (do
      (cuda/set-device! 0)
      (let [x (torch/ones [2 4] {:device :cuda})]
        (check! (.is_cuda ^Tensor (torch/add x 2.0))
                "CUDA scalar arithmetic returned a CPU tensor" {})
        (check! (= :float16
                   (amp/autocast {:device :cuda :dtype :float16}
                                 (torch/dtype
                                  (torch/matmul x (torch/transpose x 0 1)))))
                "CUDA autocast did not select float16" {}))
      (let [q (torch/randn [1 2 8 16] {:device :cuda :dtype :float16})
            output (F/scaled-dot-product-attention q q q :causal? true)]
        (check! (= [1 2 8 16] (torch/size output))
                "Fused attention returned the wrong shape"
                {:shape (torch/size output)}))
      (dist/with-process-group
        {:backend :nccl
         :rank 0
         :local-rank 0
         :world-size 1
         :master-port (available-port)
         :timeout-ms 30000}
        (let [model (nn/to (nn/linear 4 2) :cuda)
              optimizer (optim/adam (nn/parameters model))
              scaler (amp/grad-scaler {:initial-scale 128.0})
              sampler (data/distributed-sampler 16 {:num-replicas 1 :rank 0})
              checkpoint (str (System/getProperty "java.io.tmpdir")
                              "/clorch-cuda-smoke.pt")]
          (with-open [parallel-model
                      (ddp/distributed-data-parallel
                       model {:process-group (dist/current-process-group)})]
            ;; DDP must own independent tensor handles after constructor locals collect.
            (System/gc)
            (optim/zero-grad optimizer)
            (let [input (torch/randn [8 4] {:device :cuda})
                  target (torch/zeros [8 2] {:device :cuda})
                  loss (amp/autocast {:device :cuda :dtype :float16}
                                     (F/mse-loss (nn/forward parallel-model input) target))]
              (amp/backward! scaler loss)
              (check! (ddp/optimizer-step!
                       parallel-model optimizer {:scaler scaler})
                      "Finite AMP step was skipped" {})))
          (data/set-epoch! sampler 3)
          (let [before (mapv #(.clone ^Tensor %) (parameters model))]
            (dist/save-checkpoint!
             checkpoint {:model model
                         :optimizer optimizer
                         :sampler sampler
                         :scaler scaler
                         :state {:step 1}})
            (autograd/no-grad
             (doseq [^Tensor parameter (parameters model)]
               (.zero_ parameter)))
            (data/set-epoch! sampler 0)
            (check! (= {:step 1}
                       (dist/load-checkpoint!
                        checkpoint {:model model
                                    :optimizer optimizer
                                    :sampler sampler
                                    :scaler scaler}))
                    "Checkpoint training state did not round-trip" {})
            (check! (= 3 (:epoch sampler))
                    "Sampler epoch did not restore" {:epoch (:epoch sampler)})
            (check! (every? true?
                            (map #(.equal ^Tensor %1 ^Tensor %2)
                                 before (parameters model)))
                    "Model weights did not restore" {})
            (io/delete-file checkpoint true)
            (io/delete-file (str checkpoint ".edn") true))))
      (verify-launcher-failure!)
      (println "CUDA, NCCL, DDP, AMP, fused attention, and checkpoint smoke passed.")
      true)))

(defn -main [& _]
  (run-test))

(when (= (System/getProperty "clojure.main.filename") *file*)
  (run-test))
