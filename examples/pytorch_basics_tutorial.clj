(ns pytorch-basics-tutorial
  "Clorch port of the 'PyTorch Learn the Basics' tutorial.
   Source: https://docs.pytorch.org/tutorials/beginner/basics/intro.html"
  (:require [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.optim :as optim]
            [clorch.autograd :as autograd]
            [clorch.data :as data]
            [clojure.java.io]))

(defrecord NeuralNetwork [flatten-layer stack])

(extend-type NeuralNetwork
  nn/IModule
  (-forward [this x]
    (let [x (nn/forward (:flatten-layer this) x)]
      (nn/forward (:stack this) x)))
  (-train [this bool]
    (nn/train (:flatten-layer this) bool)
    (nn/train (:stack this) bool))
  (-to [this dtype-or-device]
    (-> this
        (assoc :flatten-layer (nn/to (:flatten-layer this) dtype-or-device))
        (assoc :stack (nn/to (:stack this) dtype-or-device)))))

(defn make-model []
  (->NeuralNetwork
   (nn/flatten)
   (nn/sequential
    (nn/linear (* 28 28) 512)
    (nn/relu)
    (nn/linear 512 512)
    (nn/relu)
    (nn/linear 512 10))))

(defn make-synthetic-fashion-dataloader []
  (let [n-samples 64
        dataset (data/dataset
                 :size (fn [] n-samples)
                 :get-item (fn [_idx]
                             {:data (torch/randn [1 28 28])
                              :target (torch/tensor (rand-int 10) {:dtype :int64})}))]
    (data/dataloader dataset :batch-size 16 :shuffle true)))

(defn train-epoch! [dataloader model loss-fn optimizer]
  (doseq [[batch-idx {:keys [data target]}] (map-indexed vector dataloader)]
    (torch/with-torch
      (let [target (torch/reshape target [(torch/size data 0)])
            pred (nn/forward model data)
            loss (loss-fn pred target)]

        ;; Backpropagation
        (optim/zero-grad optimizer)
        (autograd/backward loss)
        (optim/step optimizer)

        (when (= 0 (mod (inc batch-idx) 2))
          (printf "  Loss: %.6f  [%d/%d]\n"
                  (torch/item-float loss)
                  (* (inc batch-idx) (torch/size data 0))
                  (data/get-size (:dataset dataloader))))))))

(defn save-and-reload! [model]
  (let [path "model_basics.pt"]
    (torch/save model path)
    (let [new-model (make-model)]
      (torch/load new-model path)
      (println "Reloaded model from" path)
      (println new-model))
    (clojure.java.io/delete-file path)))

(torch/manual-seed 42)

(def tensor-data [[1 2] [3 4]])
(def x-data (torch/tensor tensor-data))
(def x-ones (torch/ones [2 3]))
(def x-rand (torch/randn [2 3]))
x-data
x-ones
x-rand
(torch/size x-rand)

(def dataloader (make-synthetic-fashion-dataloader))
(def first-batch (first dataloader))
(data/get-size (:dataset dataloader))
(torch/size (:data first-batch))
(torch/size (:target first-batch))

(def model (make-model))
model
(def sample-image (torch/randn [1 1 28 28]))
(def sample-logits (nn/forward model sample-image))
(torch/size sample-logits)

(def loss-fn F/cross-entropy)
(def optimizer (optim/sgd (nn/parameters model) :lr 1e-3))
(dotimes [epoch 3]
  (printf "Epoch %d\n-------------------------------\n" (inc epoch))
  (train-epoch! dataloader model loss-fn optimizer))

(save-and-reload! model)
