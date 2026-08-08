(ns clorch.activation-test
  (:require [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clojure.test :refer [deftest is testing]]))

(defn near? [a b]
  (let [eps 1e-4]
    (< (Math/abs (- (float a) (float b))) eps)))

(defn tensor-near? [t1 t2]
  (let [v1 (mapv torch/item-float (torch/tseq (torch/reshape t1 [-1])))
        v2 (mapv torch/item-float (torch/tseq (torch/reshape t2 [-1])))]
    (every? (fn [[a b]] (near? a b)) (map vector v1 v2))))

(deftest functional-activations-test
  (let [x (torch/tensor [-2.0 -1.0 0.0 1.0 2.0])]
    (testing "ReLU"
      (is (tensor-near? (F/relu x) (torch/tensor [0.0 0.0 0.0 1.0 2.0]))))

    (testing "ReLU6"
      (let [x6 (torch/tensor [-1.0 0.0 3.0 6.0 7.0])]
        (is (tensor-near? (F/relu6 x6) (torch/tensor [0.0 0.0 3.0 6.0 6.0])))))

    (testing "LeakyReLU"
      (is (tensor-near? (F/leaky-relu x 0.1) (torch/tensor [-0.2 -0.1 0.0 1.0 2.0]))))

    (testing "Tanh"
      (is (near? (torch/item-float (torch/ix (F/tanh (torch/tensor [1.0])) 0)) 0.76159)))

    (testing "GELU"
      (is (near? (torch/item-float (torch/ix (F/gelu (torch/tensor [1.0])) 0)) 0.8413447)))

    (testing "SiLU"
      (is (near? (torch/item-float (torch/ix (F/silu (torch/tensor [1.0])) 0)) 0.7310586)))

    (testing "Sigmoid"
      (is (near? (torch/item-float (torch/ix (F/sigmoid (torch/tensor [0.0])) 0)) 0.5)))

    (testing "LogSigmoid"
      (is (near? (torch/item-float (torch/ix (F/log-sigmoid (torch/tensor [0.0])) 0)) -0.693147)))

    (testing "ELU"
      (is (near? (torch/item-float (torch/ix (F/elu (torch/tensor [-1.0]) 1.0) 0)) -0.63212)))

    (testing "SELU"
      (is (near? (torch/item-float (torch/ix (F/selu (torch/tensor [1.0])) 0)) 1.0507)))

    (testing "CELU"
      (is (near? (torch/item-float (torch/ix (F/celu (torch/tensor [-1.0]) 1.0) 0)) -0.63212)))

    (testing "Softplus"
      (is (near? (torch/item-float (torch/ix (F/softplus (torch/tensor [0.0])) 0)) 0.693147)))

    (testing "Softsign"
      (is (tensor-near? (F/softsign x) (torch/tensor [-0.6666
                                                      -0.5
                                                      0.0
                                                      0.5
                                                      0.6666]))))

    (testing "Mish"
      (is (near? (torch/item-float (torch/ix (F/mish (torch/tensor [1.0])) 0)) 0.86505)))

    (testing "Hardswish"
      (is (near? (torch/item-float (torch/ix (F/hardswish (torch/tensor [1.0])) 0)) 0.66666)))

    (testing "Hardsigmoid"
      (is (near? (torch/item-float (torch/ix (F/hardsigmoid (torch/tensor [1.0])) 0)) 0.66666)))

    (testing "Hardtanh"
      (let [xt (torch/tensor [-2.0 -0.5 0.5 2.0])]
        (is (tensor-near? (F/hardtanh xt -1.0 1.0) (torch/tensor [-1.0 -0.5 0.5 1.0])))))

    (testing "LogSoftmax"
      (is (tensor-near? (F/log-softmax (torch/tensor [0.0 0.0]) 0) (torch/tensor [-0.6931 -0.6931]))))

    (testing "Softmin"
      (is (tensor-near? (F/softmin (torch/tensor [1.0 2.0]) 0) (torch/tensor [0.7310 0.2689]))))

    (testing "Shrink functions"
      (let [xt (torch/tensor [-2.0 -0.5 0.0 0.5 2.0])]
        (is (tensor-near? (F/hardshrink xt 0.5) (torch/tensor [-2.0 0.0 0.0 0.0 2.0])))
        (is (tensor-near? (F/softshrink xt 0.5) (torch/tensor [-1.5 0.0 0.0 0.0 1.5])))
        (is (tensor-near? (F/tanhshrink (torch/tensor [0.0])) (torch/tensor [0.0])))))

    (testing "Threshold"
      (is (tensor-near? (F/threshold x 0.5 10.0) (torch/tensor [10.0 10.0 10.0 1.0 2.0]))))

    (testing "GLU"
      (let [x-glu (torch/cat [x x] 0)]
        (is (tensor-near? (F/glu x-glu) (torch/tensor [-0.2384 -0.2689 0.0 0.7310 1.7615])))))

    (testing "RReLU"
      ;; Since it's randomized in training, we test eval mode (takes average of lower/upper)
      ;; Average of 1/8 and 1/3 is roughly 0.22916
      (let [expected-neg (* -2.0 (/ (+ 0.125 0.3333333333333333) 2.0))]
        (is (tensor-near? (F/rrelu (torch/tensor [-2.0 1.0]) :training false)
                          (torch/tensor [expected-neg 1.0])))))))

(deftest loss-functions-test
  (let [input (torch/tensor [0.5 0.5 0.5])
        target (torch/tensor [1.0 0.0 1.0])]
    (testing "MSE Loss"
      (is (near? (torch/item-float (F/mse-loss input target)) 0.25)))

    (testing "L1 Loss"
      (is (near? (torch/item-float (F/l1-loss input target)) 0.5)))

    (testing "Smooth L1 Loss"
      (is (near? (torch/item-float (F/smooth-l1-loss input target)) 0.125)))

    (testing "BCE Loss"
      (is (near? (torch/item-float (F/bce-loss input target)) 0.6931)))

    (testing "BCE with Logits Loss"
      (is (near? (torch/item-float (F/bce-with-logits-loss input target)) 0.6407)))))

(deftest module-activations-test
  (let [x (torch/tensor [-2.0 -1.0 0.0 1.0 2.0])]
    (testing "ReLU Module"
      (is (tensor-near? (nn/forward (nn/relu) x) (torch/tensor [0.0 0.0 0.0 1.0 2.0]))))

    (testing "GELU Module"
      (is (near? (torch/item-float (torch/ix (nn/forward (nn/gelu) (torch/tensor [1.0])) 0)) 0.8413447)))

    (testing "SiLU Module"
      (is (near? (torch/item-float (torch/ix (nn/forward (nn/silu) (torch/tensor [1.0])) 0)) 0.7310586)))

    (testing "LeakyReLU Module"
      (is (tensor-near? (nn/forward (nn/leaky-relu 0.1) x) (torch/tensor [-0.2 -0.1 0.0 1.0 2.0]))))

    (testing "PReLU Module"
      (let [m (nn/prelu {:num-parameters 1 :init 0.25})]
        (is (tensor-near? (nn/forward m x) (torch/tensor [-0.5 -0.25 0.0 1.0 2.0])))))

    (testing "Softplus Module"
      (let [m (nn/softplus 1.0 20.0)]
        (is (near? (torch/item-float (torch/ix (nn/forward m (torch/tensor [0.0])) 0)) 0.693147))))

    (testing "LogSoftmax Module"
      (is (tensor-near? (nn/forward (nn/log-softmax 0) (torch/tensor [0.0 0.0])) (torch/tensor [-0.6931 -0.6931]))))

    (testing "Shrink Modules"
      (let [xt (torch/tensor [-2.0 -0.5 0.0 0.5 2.0])]
        (is (tensor-near? (nn/forward (nn/hardshrink 0.5) xt) (torch/tensor [-2.0 0.0 0.0 0.0 2.0])))
        (is (tensor-near? (nn/forward (nn/softshrink 0.5) xt) (torch/tensor [-1.5 0.0 0.0 0.0 1.5])))))

    (testing "Threshold Module"
      (is (tensor-near? (nn/forward (nn/threshold 0.5 10.0) x) (torch/tensor [10.0 10.0 10.0 1.0 2.0]))))

    (testing "RReLU Module"
      (let [expected-neg (* -2.0 (/ (+ 0.125 0.3333333333333333) 2.0))
            m (nn/train (nn/rrelu) false)]
        (is (tensor-near? (nn/forward m (torch/tensor [-2.0 1.0]))
                          (torch/tensor [expected-neg 1.0])))))))

(deftest other-modules-test
  (testing "Identity"
    (let [x (torch/randn [2 2])]
      (is (tensor-near? (nn/forward (nn/identity) x) x))))

  (testing "Unflatten"
    (let [x (torch/reshape (torch/tensor (range 12)) [3 4])
          m (nn/unflatten 1 [2 2])
          out (nn/forward m x)]
      (is (= (torch/size out) [3 2 2]))))

  (testing "Pooling"
    (let [x (torch/randn [1 1 4 4])]
      (is (= (torch/size (nn/forward (nn/avg-pool2d 2) x)) [1 1 2 2]))
      (is (= (torch/size (nn/forward (nn/adaptive-avg-pool2d 2) x)) [1 1 2 2])))))
