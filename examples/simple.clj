(ns simple
  (:require [clorch.torch :as t]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.optim :as optim]
            [clorch.autograd :as autograd]))

(nn/defmodel MyModel [in out]
  [l1 (nn/linear in 16)
   l2 (nn/linear 16 out)]
  (forward [x]
           (let [h (nn/forward l1 x)
                 h (F/relu h)]
             (nn/forward l2 h))))

(t/with-torch
  ;; Build model: input dim 10, output dim 3
  (let [model (MyModel 10 3)

        ;; Fake batch: 4 samples, 10 features each
        x (t/randn [4 10])

        ;; Fake regression targets: 4 samples, 3 outputs each
        y (t/randn [4 3])

        ;; Optimizer over model parameters
        opt (optim/adam (nn/parameters model) :lr 1e-3)]

    ;; Forward pass
    (println "Model:")
    (println model)

    (println "\nInput shape:")
    (println (t/size x))

    (let [pred (nn/forward model x)]
      (println "\nOutput shape:")
      (println (t/size pred)))

    ;; One training step
    (optim/zero-grad opt)
    (let [pred (nn/forward model x)
          loss (F/mse-loss pred y)]
      (println "\nLoss before step:")
      (println (t/item-float loss))

      (autograd/backward loss)
      (optim/step opt))

    ;; Inspect parameters
    (let [params (nn/parameters model)]
      (println "\nParameter count:")
      (println (.size params)))

    model))



