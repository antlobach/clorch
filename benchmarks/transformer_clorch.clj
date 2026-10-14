(ns transformer-clorch
  (:require [clorch.autograd :as autograd]
            [clorch.nn :as nn]
            [clorch.torch :as t])
  (:import [org.bytedeco.pytorch.global torch]))

(set! *warn-on-reflection* true)

(def batch-size 4)
(def sequence-length 256)
(def dimension 512)
(def heads 8)
(def warmup 5)
(def iterations 20)

(defn- median [samples]
  (let [ordered (vec (sort samples))]
    (nth ordered (quot (count ordered) 2))))

(defn benchmark []
  (torch/set_num_threads 1)
  (t/manual-seed 23)
  (let [model (nn/multihead-attention dimension heads :dropout 0.0)
        x (t/randn [batch-size sequence-length dimension])]
    (nn/train model false)
    (autograd/no-grad
     (dotimes [_ warmup]
       (t/with-torch
         (nn/forward model x)
         nil))
     (let [samples
           (repeatedly
            iterations
            (fn []
              (let [started (System/nanoTime)]
                (t/with-torch
                  (nn/forward model x)
                  nil)
                (/ (double (- (System/nanoTime) started)) 1000000.0))))]
       {:runtime "clorch"
        :median-ms (median samples)}))))
