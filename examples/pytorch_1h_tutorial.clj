(ns pytorch-1h-tutorial
  "Port of Sebastian Raschka's 'PyTorch in One Hour: From Tensors to Training Neural Networks'
   Source: https://sebastianraschka.com/teaching/pytorch-1h/
   Date: Jul 1, 2025"
  (:require [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.autograd :as autograd]
            [clorch.cuda :as cuda]
            [clorch.optim :as optim]
            [clorch.data :as data]))

(println "==========================================================")
(println "Port of: PyTorch in One Hour by Sebastian Raschka")
(println "==========================================================\n")

;; ==========================================================
;; 2. Understanding tensors
;; ==========================================================

(println "--- 2. Understanding tensors ---")
(def tensor-0d (torch/tensor 1))
(println "Scalar:" tensor-0d)

(def tensor-1d (torch/tensor [1 2 3]))
(println "Vector:" tensor-1d)

(def tensor-2d (torch/tensor [[1 2] [3 4]]))
(println "Matrix:\n" tensor-2d)

(def tensor-3d (torch/tensor [[[1 2] [3 4]] [[5 6] [7 8]]]))
(println "3D Tensor:\n" tensor-3d)

(println "tensor-1d dtype:" (.toString (.scalar_type tensor-1d)))
(def float-vec (torch/tensor [1.0 2.0 3.0]))
(println "float-vec dtype:" (.toString (.scalar_type float-vec)))

(def tensor-2d-ex (torch/tensor [[1 2 3] [4 5 6]]))
(println "Shape:" (torch/size tensor-2d-ex))
(def reshaped (torch/reshape tensor-2d-ex [3 2]))
(def transposed (torch/transpose tensor-2d-ex 0 1))
(def matmul-res (torch/matmul tensor-2d-ex transposed))
(println "Matmul result size:" (torch/size matmul-res))

;; ==========================================================
;; 3. Seeing models as computation graphs
;; ==========================================================

(println "\n--- 3. Computation Graphs ---")
(def y (torch/tensor [1.0]))
(def x1 (torch/tensor [1.1]))
(def w1 (torch/tensor [2.2]))
(def b (torch/tensor [0.0]))
(def z (torch/add (torch/mul x1 w1) b))
(println "Forward pass result:" (torch/item-float z))

;; ==========================================================
;; 4. Automatic differentiation made easy
;; ==========================================================

(println "\n--- 4. Automatic Differentiation ---")
(def w1-grad (torch/tensor [2.2] {:requires-grad true}))
(def b-grad (torch/tensor [0.0] {:requires-grad true}))

(torch/with-torch
  (let [z-grad (torch/add (torch/mul x1 w1-grad) b-grad)
        a (.sigmoid z-grad)
        loss (F/mse-loss a y)]
    (autograd/backward loss)
    (println "w1 gradient:" (torch/item-float (autograd/grad w1-grad)))
    (println "b gradient:" (torch/item-float (autograd/grad b-grad)))))

;; ==========================================================
;; 5. Implementing multilayer neural networks
;; ==========================================================

(println "\n--- 5. Multilayer Neural Networks ---")
(def model (nn/sequential
            (nn/linear 50 30)
            (nn/relu)
            (nn/linear 30 20)
            (nn/relu)
            (nn/linear 20 3)))

(println "Model structure summary:")
(println model)

;; Check trainable parameters
(def params (nn/parameters model))
(println "Total parameter chunks:" (.size params))

;; Forward pass
(def x-rand (torch/randn [1 50]))
(def out-logits (nn/forward model x-rand))
(println "Logits size:" (torch/size out-logits))

;; ==========================================================
;; 6. Setting up efficient data loaders
;; ==========================================================

(println "\n--- 6. Data Loaders ---")
(def x-train (torch/tensor [[-1.2 3.1]
                            [-0.9 2.9]
                            [-0.5 2.6]
                            [2.3 -1.1]
                            [2.7 -1.5]]))
(def y-train (torch/tensor [0 0 0 1 1] {:dtype :int64}))

(def train-dataset (data/tensor-dataset x-train y-train))
(def train-loader (data/dataloader train-dataset 2 {:shuffle true}))

;; ==========================================================
;; 7. A typical training loop
;; ==========================================================

(println "\n--- 7. Training Loop ---")
(def model-train (nn/sequential
                  (nn/linear 2 30)
                  (nn/relu)
                  (nn/linear 30 2)))
(def sgd-optimizer (optim/sgd (nn/parameters model-train) :lr 0.5))

;; Canonical scoped training pattern: see docs/memory.md.

(dotimes [epoch 10]
  (doseq [{:keys [data target]} train-loader]
    (torch/with-torch
      (let [logits (nn/forward model-train data)
            loss (F/cross-entropy logits target)]
        (optim/zero-grad sgd-optimizer)
        (autograd/backward loss)
        (optim/step sgd-optimizer)
        (println "Epoch:" (inc epoch) "Loss:" (torch/item-float loss))))))

;; Calculate Accuracy
(defn compute-accuracy [model-to-evaluate loader]
  (let [results (doall
                 (for [{:keys [data target]} loader]
                   (torch/with-torch
                     (let [logits (nn/forward model-to-evaluate data)
                           preds  (torch/argmax logits 1)
                           correct (torch/eq preds target)]
                       (torch/item-float (torch/mean (torch/to-float correct)))))))]
    (/ (apply clojure.core/+ results) (count results))))

(println "Final Training Accuracy:"
         (compute-accuracy model-train train-loader))

;; ==========================================================
;; 9. Optimizing training performance with GPUs
;; ==========================================================

(println "\n--- 9. GPU Performance ---")
(println "CUDA available?" (cuda/available?))

(println "\n--- Sebastian Raschka Tutorial Port Complete ---")
