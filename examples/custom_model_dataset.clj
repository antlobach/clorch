(ns custom-model-dataset
  (:require [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.optim :as optim]
            [clorch.autograd :as autograd]
            [clorch.data :as data]))

;; --- 1. Custom Dataset ---
;; PyTorch Semantic: Anything with length and indexer.
(defn create-synthetic-dataset [n-samples dim]
  (data/dataset
   :size (fn [] n-samples)
   :get-item (fn [_idx]
                ;; Logic: label is 1 if sum of features is positive
               (let [x (torch/randn [dim])
                     y (if (> (torch/item-float (torch/sum x)) 0) 1 0)]
                 {:data x :target (torch/tensor y {:dtype :int64})}))))

;; --- 2. Custom Module (Zero-Overhead Protocol Design) ---
;; PyTorch Semantic: Container of state + forward logic.

;; Define the state structure using defrecord
(defrecord ResBlock [l1 l2])

;; Implement the high-performance IModule protocol
(extend-type ResBlock
  nn/IModule
  (-forward [this x]
    (let [residual x
          out (-> x
                  (as-> $ (nn/forward (:l1 this) $))
                  (F/relu)
                  (as-> $ (nn/forward (:l2 this) $)))]
      ;; The "Skip Connection"
      (torch/add out residual)))
  (-train [this bool]
    (nn/train (:l1 this) bool)
    (nn/train (:l2 this) bool))
  (-to [this dtype-or-device]
    (-> this
        (assoc :l1 (nn/to (:l1 this) dtype-or-device))
        (assoc :l2 (nn/to (:l2 this) dtype-or-device)))))

;; Constructor for the ResBlock
(defn res-block [dim]
  (->ResBlock (nn/linear dim dim) (nn/linear dim dim)))

;; Define a more complex model containing nested modules
(defrecord MyComplexModel [backbone head])

(extend-type MyComplexModel
  nn/IModule
  (-forward [this x]
    (let [features (nn/forward (:backbone this) x)]
      (nn/forward (:head this) features)))
  (-train [this bool]
    (nn/train (:backbone this) bool)
    (nn/train (:head this) bool))
  (-to [this dtype-or-device]
    (-> this
        (assoc :backbone (nn/to (:backbone this) dtype-or-device))
        (assoc :head (nn/to (:head this) dtype-or-device)))))

(defn my-complex-model [in-dim out-dim]
  (->MyComplexModel
   (nn/sequential
    (nn/linear in-dim 32)
    (nn/relu)
    (res-block 32) ;; Nested Custom Module!
    (res-block 32))
   (nn/linear 32 out-dim)))

(def dim 10)
(def n-classes 2)

(def dataset (create-synthetic-dataset 100 dim))
(def dl (data/dataloader dataset :batch-size 10 :shuffle true))
(data/get-size dataset)

(def batch0 (first dl))
(torch/size (:data batch0))
(torch/size (:target batch0))

(def model (my-complex-model dim n-classes))
model
(def sample-logits (nn/forward model (:data batch0)))
(torch/size sample-logits)

(def optimizer (optim/adam (nn/parameters model) :lr 1e-3))

;; Keep model and optimizer long-lived; scope only each batch. See docs/memory.md.

(dotimes [epoch 5]
  (let [total-loss (atom 0.0)
        n-batches (atom 0)]
    (doseq [{:keys [data target]} dl]
      (let [loss-value
            (torch/with-torch
              (optim/zero-grad optimizer)
              (let [logits (nn/forward model data)
                    target (torch/reshape target [(torch/size data 0)])
                    loss (F/cross-entropy logits target)]
                (autograd/backward loss)
                (optim/step optimizer)
                (torch/item-float loss)))]
        (swap! total-loss + loss-value)
        (swap! n-batches inc)))
    (printf "Epoch %d | Avg Loss: %.4f\n"
            (inc epoch)
            (/ @total-loss @n-batches))))
