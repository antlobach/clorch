(ns synthetic
  (:require [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.autograd :as autograd]
            [clorch.data :as data]
            [clorch.optim :as optim]))

(defn make-data-map [d n-classes]
  (torch/with-torch
    (let [means (torch/mul (torch/randn [n-classes d]) 3.0)
          gen-sample (fn [n]
                       (let [y (torch/rand-int 0 n-classes [n])
                             x-m (torch/ix means y :_)
                             noise (torch/mul 0.5 (torch/randn [n d]))]
                         [(torch/add x-m noise) y]))
          [x-tr y-tr] (gen-sample 100)
          [x-te y-te] (gen-sample 20)]
      {:x-tr x-tr :y-tr y-tr :x-te x-te :y-te y-te})))

(torch/manual-seed 123)

(def d 10)
(def n-classes 3)
(def batch-size 2)
(def num-epochs 2)

(def data-map (make-data-map d n-classes))
(keys data-map)
(torch/size (:x-tr data-map))
(torch/size (:y-tr data-map))

(def model
  (nn/sequential
   (nn/linear d 4)
   (nn/relu)
   (nn/linear 4 n-classes)))

(def train-ds (data/tensor-dataset (:x-tr data-map) (:y-tr data-map)))
(def train-loader (data/dataloader train-ds :batch-size batch-size :shuffle true))
(def optimizer (optim/sgd (nn/parameters model) :lr 0.01))

(nn/train model true)
(doseq [epoch (range num-epochs)]
  (let [losses (atom [])]
    (doseq [{:keys [data target]} train-loader]
      (torch/with-torch
        (let [logits (nn/forward model data)
              loss (F/cross-entropy logits target)]
          (optim/zero-grad optimizer)
          (autograd/backward loss)
          (optim/step optimizer)
          (swap! losses conj (torch/item-float loss)))))
    (printf "Epoch %d | Avg Loss: %.4f\n"
            (inc epoch)
            (/ (apply + @losses) (count @losses)))))

(data/cleanup-data! data-map)
(data/cleanup-data! train-loader)
(when (instance? java.io.Closeable model) (.close model))
(when (instance? java.io.Closeable optimizer) (.close optimizer))
