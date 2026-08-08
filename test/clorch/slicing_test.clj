(ns clorch.slicing-test
  (:require [clorch.torch :as torch]
            [clojure.test :refer [deftest is testing]]))

(defn tensor->vec [t]
  (mapv torch/item-float (torch/tseq t)))

(defn ix-ellipsis [t dim]
  (if (nil? dim)
    (torch/ix t (quote ...))
    (torch/ix t (quote ...) dim)))

(deftest basic-indexing-1d
  (testing "Basic 1D tensor indexing"
    (let [x-1d (torch/tensor [10 11 12 13 14])]
      (is (= 10.0 (torch/item-float (torch/ix x-1d 0))))
      (is (= 14.0 (torch/item-float (torch/ix x-1d -1)))))))

(deftest basic-indexing-2d
  (testing "Basic 2D tensor indexing"
    (let [x-2d (torch/tensor [[1 2 3] [4 5 6] [7 8 9]])]
      (is (= 2.0 (torch/item-float (torch/ix x-2d 0 1))))
      (is (= [1.0 2.0 3.0] (tensor->vec (torch/ix x-2d 0)))))))

(deftest slicing-1d
  (testing "1D tensor slicing"
    (let [y-1d (torch/tensor (range 10))]
      (is (= [2.0 3.0 4.0] (tensor->vec (torch/ix y-1d [2 5]))))
      (is (= [0.0 1.0 2.0 3.0] (tensor->vec (torch/ix y-1d [0 4]))))
      (is (= [6.0 7.0 8.0 9.0] (tensor->vec (torch/ix y-1d [6 10]))))
      (is (= [0.0 2.0 4.0 6.0 8.0] (tensor->vec (torch/ix y-1d [0 10 2])))))))

(deftest negative-indexing
  (testing "Negative indexing"
    (let [x (torch/tensor (range 10))]
      (is (= 9.0 (torch/item-float (torch/ix x -1))))
      (is (= 8.0 (torch/item-float (torch/ix x -2))))
      (is (= [7.0 8.0 9.0] (tensor->vec (torch/ix x [-3 10])))))))

(deftest ellipsis-indexing
  (testing "Ellipsis indexing"
    (let [t (torch/reshape (torch/tensor (range 24)) [2 3 4])]
      (is (= [3 4] (torch/size (torch/ix t 0 (quote ...)))))
      (is (= [2 3] (torch/size (torch/ix t (quote ...) 1)))))))

(deftest identity-slicing
  (testing "Select all with :_ or :all"
    (let [t (torch/tensor [[1 2 3] [4 5 6]])]
      (is (= [2 3] (torch/size (torch/ix t :_ :_))))
      (is (= [2 3] (torch/size (torch/ix t :all :all)))))))

(deftest slice-edge-cases
  (testing "Slice edge cases"
    (let [t (torch/tensor (range 10))]
      (is (= [0.0 1.0 2.0] (tensor->vec (torch/ix t [0 3]))))
      (is (= [7.0 8.0 9.0] (tensor->vec (torch/ix t [7 10]))))
      (is (= [8.0 9.0] (tensor->vec (torch/ix t [-2 10])))))))
