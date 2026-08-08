(ns clorch.autograd-test
  (:require [clorch.torch :as t]
            [clorch.nn.functional :as F]
            [clorch.autograd :as autograd]
            [clojure.test :refer [deftest is testing]]))

(defn near? [a b]
  (let [eps 1e-3]
    (< (Math/abs (- (float a) (float b))) eps)))

(deftest activation-gradients-test
  (t/with-torch
    (testing "ReLU gradient"
      (let [x (t/tensor [-1.0 1.0] {:requires-grad true})
            y (t/sum (F/relu x))]
        (autograd/backward y)
        (is (= [0.0 1.0] (mapv t/item-float (t/tseq (autograd/grad x)))))))

    (testing "Sigmoid gradient"
      (let [x (t/tensor [0.0] {:requires-grad true})
            y (F/sigmoid x)]
        (autograd/backward y)
        ;; d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        ;; sigmoid(0) = 0.5, so 0.5 * 0.5 = 0.25
        (is (near? (t/item-float (t/ix (autograd/grad x) 0)) 0.25))))

    (testing "Tanh gradient"
      (let [x (t/tensor [0.0] {:requires-grad true})
            y (F/tanh x)]
        (autograd/backward y)
        ;; d/dx tanh(x) = 1 - tanh^2(x)
        ;; tanh(0) = 0, so 1 - 0 = 1
        (is (near? (t/item-float (t/ix (autograd/grad x) 0)) 1.0))))

    (testing "Mish gradient"
      (let [x (t/tensor [1.0] {:requires-grad true})
            y (F/mish x)]
        (autograd/backward y)
        (is (pos? (t/item-float (t/ix (autograd/grad x) 0))))))

    (testing "Softplus gradient"
      (let [x (t/tensor [0.0] {:requires-grad true})
            y (F/softplus x)]
        (autograd/backward y)
        ;; d/dx ln(1 + e^x) = 1 / (1 + e^-x) = sigmoid(x)
        ;; sigmoid(0) = 0.5
        (is (near? (t/item-float (t/ix (autograd/grad x) 0)) 0.5))))))

(deftest linear-gradient-test
  (t/with-torch
    (let [x (t/randn [1 10] {:requires-grad true})
          w (t/randn [5 10] {:requires-grad true})
          b (t/zeros [5] {:requires-grad true})
          y (t/sum (F/linear x w b))]
      (autograd/backward y)
      (is (some? (autograd/grad x)))
      (is (some? (autograd/grad w)))
      (is (some? (autograd/grad b))))))
