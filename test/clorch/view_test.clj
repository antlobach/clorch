(ns clorch.view-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.torch :as torch]))

(deftest view-vs-reshape-test
  (torch/with-torch
    (let [t (torch/randn [4 4])]

      (testing "view on contiguous tensor succeeds"
        (let [v (torch/view t [2 8])]
          (is (= [2 8] (torch/size v)))))

      (testing "view on non-contiguous tensor fails (PyTorch strictness)"
        (let [t-trans (torch/T t)]
          (is (thrown? Exception (torch/view t-trans [16])))))

      (testing "reshape on non-contiguous tensor succeeds (flexible)"
        (let [t-trans (torch/T t)
              r (torch/reshape t-trans [16])]
          (is (= [16] (torch/size r)))
          ;; Verify data is the same
          (is (= (torch/item-float (torch/ix t-trans 0 0))
                 (torch/item-float (torch/ix r 0)))))))))

(deftest reduction-ops-test
  (torch/with-torch
    (let [t (torch/tensor [[1.0 2.0] [3.0 4.0]])]
      (testing "mean with dim and keepdim"
        (let [m (torch/mean t 1 :keepdim true)]
          (is (= [2 1] (torch/size m)))
          (is (= 1.5 (torch/item-float (torch/ix m 0 0))))))

      (testing "sum with dim"
        (let [s (torch/sum t 0)]
          (is (= [2] (torch/size s)))
          (is (= 4.0 (torch/item-float (torch/ix s 0))))))

      (testing "var with dim"
        (let [v (torch/var t 1 :unbiased false)]
          (is (= [2] (torch/size v)))
          (is (= 0.25 (torch/item-float (torch/ix v 0)))))))))
