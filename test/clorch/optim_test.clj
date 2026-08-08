(ns clorch.optim-test
  (:require [clorch.torch :as t]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.optim :as optim]
            [clorch.autograd :as autograd]
            [clojure.test :refer [deftest is testing]]))

(deftest optimizer-smoke-test
  (t/with-torch
    (let [model (nn/linear 10 1)
          params (nn/parameters model)
          x (t/randn [1 10])
          y (t/tensor [[1.0]])]
      (testing "AdamW"
        (let [opt (optim/adamw params :lr 0.01)]
          (optim/zero-grad opt)
          (let [loss (F/mse-loss (nn/forward model x) y)]
            (autograd/backward loss)
            (optim/step opt)
            (is (some? opt)))))

      (testing "RMSprop"
        (let [opt (optim/rmsprop params :lr 0.01)]
          (optim/zero-grad opt)
          (let [loss (F/mse-loss (nn/forward model x) y)]
            (autograd/backward loss)
            (optim/step opt)
            (is (some? opt)))))

      (testing "Adagrad"
        (let [opt (optim/adagrad params :lr 0.01)]
          (optim/zero-grad opt)
          (let [loss (F/mse-loss (nn/forward model x) y)]
            (autograd/backward loss)
            (optim/step opt)
            (is (some? opt))))))))

(defn- step-quadratic! [optimizer weight]
  (dotimes [_ 5]
    (optim/zero-grad optimizer)
    (autograd/backward (t/mul weight weight))
    (optim/step optimizer))
  (.item_double (t/->tensor weight)))

(deftest adam-beta-order-test
  (testing "Adam and AdamW preserve semantic beta order across native ABIs"
    (t/with-torch
      (let [adam-weight (t/tensor [10.0] {:dtype :float64 :requires-grad true})
            adamw-weight (t/tensor [10.0] {:dtype :float64 :requires-grad true})
            adam (optim/adam [adam-weight] :lr 0.1 :betas [0.8 0.95])
            adamw (optim/adamw [adamw-weight] :lr 0.1 :betas [0.8 0.95] :weight-decay 0.0)]
        (is (< (Math/abs (- 9.500734416507376
                            (step-quadratic! adam adam-weight)))
               1e-10))
        (is (< (Math/abs (- 9.500734416507376
                            (step-quadratic! adamw adamw-weight)))
               1e-10))))))
