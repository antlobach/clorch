(ns clorch.nn-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.autograd :as autograd]))

(deftest parameter-test
  (testing "Creation of parameters with default requires-grad=true"
    (torch/with-torch
      (let [p (nn/parameter (torch/randn [3 3]))]
        (is (instance? clorch.nn.Parameter p))
        (let [t (torch/->tensor p)]
          (is (instance? org.bytedeco.pytorch.Tensor t))
          (is (.requires_grad t))))))

  (testing "Creation of parameters with explicit requires-grad=false"
    (torch/with-torch
      (let [p (nn/parameter (torch/randn [3 3]) :requires-grad false)]
        (is (not (.requires_grad (torch/->tensor p))))))))

(defrecord MyModule [weight bias]
  nn/IModule
  (-forward [this x]
    (torch/add (torch/matmul x weight) bias))
  (-train [this bool] this))

(deftest module-parameters-discovery-test
  (testing "Parameters are discovered in custom records"
    (torch/with-torch
      (let [w (nn/parameter (torch/randn [10 5]))
            b (nn/parameter (torch/randn [5]))
            model (->MyModule w b)
            params (nn/parameters model)]
        (is (= 2 (.size params)))
        (is (some #(= [10 5] (torch/size %)) (map #(.get params (clojure.core/long %)) (range 2))))
        (is (some #(= [5] (torch/size %)) (map #(.get params (clojure.core/long %)) (range 2))))))))

(deftest trainable-parameters-discovery-test
  (testing "Only trainable parameters are discovered when trainable-only is true"
    (torch/with-torch
      (let [w (nn/parameter (torch/randn [10 5]))
            b (nn/parameter (torch/randn [5]) :requires-grad false)
            model (->MyModule w b)
            params (nn/parameters model :trainable-only true)]
        (is (= 1 (.size params)))
        (is (= [10 5] (torch/size (.get params 0))))))))

(deftest state-dict-parameter-test
  (testing "state-dict correctly extracts Tensors from Parameters"
    (torch/with-torch
      (let [w (nn/parameter (torch/ones [3]))
            model (->MyModule w (torch/zeros [3]))
            sd (nn/state-dict model)]
        (is (instance? org.bytedeco.pytorch.Tensor (:weight sd)))
        (is (instance? org.bytedeco.pytorch.Tensor (:bias sd)))
        (is (= [3] (torch/size (:weight sd))))))))

(deftest load-state-dict-parameter-test
  (testing "load-state-dict correctly updates Tensors in Parameters"
    (torch/with-torch
      (let [w (nn/parameter (torch/ones [3]))
            b (nn/parameter (torch/ones [3]))
            model (->MyModule w b)
            new-weights (torch/zeros [3])]
        (nn/load-state-dict model {:weight new-weights :bias (torch/zeros [3])})
        (is (= 0.0 (torch/item-float (torch/sum (torch/->tensor w)))))
        (is (= 0.0 (torch/item-float (torch/sum (torch/->tensor b)))))))))
