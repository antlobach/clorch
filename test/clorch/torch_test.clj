(ns clorch.torch-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clorch.torch :as torch]))

;; --- Helpers ---

(defn- near? [a b]
  (let [diff (Math/abs (- (float a) (float b)))]
    (< diff 1e-3)))

;; --- Generators ---

(def gen-shape (gen/vector (gen/choose 2 8) 1 4))

;; --- ix (Ergonomic Indexing) Tests ---

(deftest ix-unit-examples-test
  (testing "Exhaustive indexing examples matching Python"
    (torch/with-torch
      (let [t (torch/reshape (torch/tensor (map float (range 120))) [2 6 10])]

        (testing "Selection (Rank Reduction)"
          (is (= [] (torch/size (torch/ix t 0 0 0))))
          (is (= [6 10] (torch/size (torch/ix t 1)))))

        (testing "Slicing"
          (is (= [2 3 10] (torch/size (torch/ix t :_ [1 4] :_))))
          (is (= [2 6 3] (torch/size (torch/ix t :all :all [0 5 2])))))

        (testing "Ellipsis"
          (is (= [2 6] (torch/size (torch/ix t '... 0))))
          (is (= [6 10] (torch/size (torch/ix t 0 '...)))))

        (testing "Open-ended & Steps"
          (is (= [2 4 10] (torch/size (torch/ix t :_ [2 nil]))))
          (is (= [2 6 5] (torch/size (torch/ix t :_ :_ [nil nil 2])))))

        (testing "Negative Indexing"
          ;; Fix: Use manual selection for stable comparison
          (let [expected (torch/item-float (torch/ix t 1 5 9))]
            (is (near? expected (torch/item-float (torch/ix t -1 -1 -1))))))))))

(deftest advanced-indexing-test
  (testing "Indexing with Tensors"
    (torch/with-torch
      (let [data (torch/randn [3 10])
            ;; Fix indices: Must be strictly less than 3
            indices (torch/tensor [0 2 0 1] {:dtype :int64})
            res (torch/ix data indices :_)]
        (is (= [4 10] (torch/size res)))))))

;; --- Property Based Slicing Tests ---

(defspec ix-rank-reduction-prop 50
  (prop/for-all [shape (gen/vector (gen/choose 2 5) 2 4)]
                (torch/with-torch
                  (let [t (torch/randn shape)
                        res (torch/ix t 0)]
                    (= (torch/size res) (vec (rest shape)))))))

(defspec ix-identity-prop 50
  (prop/for-all [shape gen-shape]
                (torch/with-torch
                  (let [t (torch/randn shape)
                        indexers (repeat (count shape) :_)
                        sliced (apply torch/ix t indexers)]
                    (= (torch/size sliced) shape)))))

(defspec ix-ellipsis-prop 50
  (prop/for-all [shape (gen/vector (gen/choose 2 5) 3 4)]
                (torch/with-torch
                  (let [t (torch/randn shape)
                        res (torch/ix t '... 0)]
                    (= (torch/size res) (vec (butlast shape)))))))

;; --- General Arithmetic Properties ---

(defspec add-commutative-prop 50
  (prop/for-all [shape gen-shape]
                (torch/with-torch
                  (let [a (torch/randn shape)
                        b (torch/randn shape)
                        res1 (torch/add a b)
                        res2 (torch/add b a)
                        diff (torch/sub res1 res2)
                        sq-err (torch/sum (torch/mul diff diff))]
                    (< (torch/item-float sq-err) 1e-3)))))

(deftest dtype-preservation-test
  (testing "Dtype preservation across creation"
    (torch/with-torch
      (doseq [[kw expected] [[:float32 "Float"] [:int64 "Long"] [:int32 "Int"]
                             [:float16 "Half"] [:bfloat16 "BFloat16"]
                             [:complex64 "ComplexFloat"] [:complex128 "ComplexDouble"]]]
        (let [t (torch/ones [2] {:dtype kw})]
          (is (= expected (.toString (.scalar_type t))))))
      (is (= :float16 (torch/dtype (torch/ones [2] {:dtype :float16}))))
      (is (= :bfloat16 (torch/dtype (torch/ones [2] {:dtype :bfloat16})))))))

(deftest float64-tensor-construction-test
  (testing "Float64 tensors preserve dtype and values"
    (torch/with-torch
      (let [t (torch/tensor [1.25 2.5] {:dtype :float64})]
        (is (= :float64 (torch/dtype t)))
        (is (= [2] (torch/size t)))
        (is (near? 3.75 (torch/item-float (torch/sum t))))))))

(deftest math-ops-test
  (testing "Basic arithmetic value check"
    (torch/with-torch
      (let [t1 (torch/tensor [1.0 2.0 3.0])
            t2 (torch/tensor [4.0 5.0 6.0])
            t3 (torch/add t1 t2)
            t4 (torch/mul t1 2.0)]
        (is (near? 5.0 (torch/item-float (torch/ix t3 0))))
        (is (near? 7.0 (torch/item-float (torch/ix t3 1))))
        (is (near? 9.0 (torch/item-float (torch/ix t3 2))))
        (is (near? 2.0 (torch/item-float (torch/ix t4 0))))
        (is (near? 4.0 (torch/item-float (torch/ix t4 1))))
        (is (near? 6.0 (torch/item-float (torch/ix t4 2))))))))
