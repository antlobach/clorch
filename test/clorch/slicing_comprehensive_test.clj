(ns clorch.slicing-comprehensive-test
  (:require [clorch.torch :as torch]
            [clojure.test :refer [deftest is testing]]))

(defn tensor->vec [t]
  (mapv torch/item-float (torch/tseq t)))

(defn tensor->vecs [t]
  (mapv #(mapv torch/item-float (torch/tseq %)) (torch/tseq t)))

(defn near? [a b]
  (let [diff (Math/abs (- (float a) (float b)))]
    (< diff 1e-5)))

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

(deftest negative-step-slicing
  (testing "Negative step (reverse) slicing - PyTorch parity"
    (let [t (torch/tensor (range 10))]
      (testing "Full reverse [::-1]"
        (is (= [9.0 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0]
               (tensor->vec (torch/ix t [nil nil -1])))))
      (testing "Reverse from end [5::-1]"
        (is (= [5.0 4.0 3.0 2.0 1.0 0.0]
               (tensor->vec (torch/ix t [5 nil -1])))))
      (testing "Reverse subset [5:2:-1]"
        (is (= [5.0 4.0 3.0]
               (tensor->vec (torch/ix t [5 2 -1])))))
      (testing "Reverse with step -2"
        (is (= [9.0 7.0 5.0 3.0 1.0]
               (tensor->vec (torch/ix t [nil nil -2]))))))))

(deftest negative-step-2d
  (testing "Negative step on 2D tensors"
    (let [t (torch/tensor [[1 2 3 4] [5 6 7 8] [9 10 11 12]])]
      (testing "Reverse rows [::-1]"
        (is (= [[9.0 10.0 11.0 12.0] [5.0 6.0 7.0 8.0] [1.0 2.0 3.0 4.0]]
               (tensor->vecs (torch/ix t [nil nil -1] :_)))))
      (testing "Reverse columns [:, ::-1]"
        (is (= [[4.0 3.0 2.0 1.0] [8.0 7.0 6.0 5.0] [12.0 11.0 10.0 9.0]]
               (tensor->vecs (torch/ix t :_ [nil nil -1]))))))))

(deftest open-ended-slicing
  (testing "Open-ended slicing"
    (let [t (torch/tensor (range 10))]
      (is (= [5.0 6.0 7.0 8.0 9.0] (tensor->vec (torch/ix t [5 nil]))))
      (is (= [0.0 1.0 2.0 3.0 4.0] (tensor->vec (torch/ix t [nil 5]))))
      (is (= (mapv float (range 10)) (tensor->vec (torch/ix t [nil nil])))))))

(deftest integer-tensor-indexing
  (testing "Integer tensor indexing"
    (let [t (torch/tensor [[1 2 3] [4 5 6] [7 8 9] [10 11 12]])
          idx (torch/tensor [0 2 0 1] {:dtype :int64})]
      (is (= [4] (torch/size (torch/ix t idx idx))))
      (is (= [1.0 9.0 1.0 5.0] (tensor->vec (torch/ix t idx idx))))
      (is (= [4 3] (torch/size (torch/ix t idx :_))))
      (is (= [4 4] (torch/size (torch/ix t :_ idx)))))))

(deftest boolean-mask-indexing
  (testing "1D boolean mask indexing"
    (let [t (torch/tensor [1.0 2.0 3.0 4.0 5.0])
          mask (torch/tensor [false false true true true] {:dtype :bool})]
      (is (= [3.0 4.0 5.0] (tensor->vec (torch/ix t mask))))))
  (testing "2D boolean mask indexing"
    (let [t (torch/tensor [[1 2] [3 4]])
          mask (torch/tensor [[true false] [false true]] {:dtype :bool})
          res (torch/ix t mask)]
      (is (= [2] (torch/size res)))
      (is (= [1.0 4.0] (tensor->vec res))))))

(deftest multi-dimensional-slicing
  (testing "Multi-dimensional indexing"
    (let [t (torch/reshape (torch/tensor (range 120) {:dtype :float32}) [2 6 10])]
      (is (= 12.0 (torch/item-float (torch/ix t 0 1 2))))
      (is (= 119.0 (torch/item-float (torch/ix t 1 5 9))))
      (is (= [6 10] (torch/size (torch/ix t 0))))
      (is (= [10] (torch/size (torch/ix t 0 1))))
      (is (= [] (torch/size (torch/ix t 0 1 0))))
      (is (= [2 3 10] (torch/size (torch/ix t :_ [1 4] :_))))
      (is (= [2 6 3] (torch/size (torch/ix t :all :all [0 5 2])))))))

(deftest gradient-preservation
  (testing "Gradient tracking through slicing"
    (let [t (torch/tensor [1.0 2.0 3.0] {:requires-grad true})
          sliced (torch/ix t [0 2])]
      (is (.requires_grad t))
      (is (.requires_grad sliced)))))

(deftest empty-slice
  (testing "Empty slice results"
    (let [t (torch/tensor (range 10))]
      (is (= [0] (torch/size (torch/ix t [5 5]))))
      (is (= [] (tensor->vec (torch/ix t [5 5])))))))

(deftest tensor-4d-indexing
  (testing "4D tensor indexing with ellipsis"
    (let [t (torch/reshape (torch/tensor (range 24) {:dtype :float32}) [1 2 3 4])]
      (is (= [1 2 3 4] (torch/size t)))
      (is (= [3 4] (torch/size (torch/ix t 0 0 :_ :_))))
      (is (= [1 2] (torch/size (torch/ix t (quote ...) 0 0))))
      (is (= [2 3] (torch/size (torch/ix t 0 (quote ...) 0)))))))

(deftest combined-advanced-indexing
  (testing "Combined integer tensor with slice"
    (let [t (torch/tensor [[1 2 3 4] [5 6 7 8] [9 10 11 12] [13 14 15 16]])
          idx (torch/tensor [0 2] {:dtype :int64})]
      (is (= [2 4] (torch/size (torch/ix t idx :_))))
      (is (= [4 2] (torch/size (torch/ix t :_ idx)))))))
