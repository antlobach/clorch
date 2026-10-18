(ns clorch.torch-compatibility-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.nn.functional :as F]
            [clorch.torch :as t]))

(defn- values [tensor]
  (mapv t/item-float (t/tseq (t/reshape tensor [-1]))))

(defn- near? [expected actual]
  (< (Math/abs (- (double expected) (double actual))) 1.0e-6))

(deftest like-factories-preserve-input-properties-test
  (t/with-torch
    (let [source (t/tensor [[2.0 3.0]] {:dtype :float64})]
      (doseq [result [(t/empty-like source)
                      (t/zeros-like source)
                      (t/ones-like source)
                      (t/full-like source 4.0)
                      (t/rand-like source)
                      (t/randn-like source)]]
        (is (= [1 2] (t/size result)))
        (is (= :float64 (t/dtype result))))
      (is (= [0.0 0.0] (values (t/zeros-like source))))
      (is (= [1.0 1.0] (values (t/ones-like source))))
      (is (= [4.0 4.0] (values (t/full-like source 4.0)))))))

(deftest reduction-compatibility-test
  (t/with-torch
    (let [input (t/tensor [[1.0 2.0] [3.0 4.0]])]
      (is (near? 24.0 (t/item-float (t/prod input))))
      (is (= [2.0 12.0] (values (t/prod input 1))))
      (is (every? #(near? (Math/sqrt 0.5) %) (values (t/std input 1 :unbiased true))))
      (is (near? 2.0 (t/item-float (t/median input))))
      (is (= [2.0 3.0]
             (values (t/nanmean (t/tensor [[1.0 ##NaN] [3.0 3.0]]) 0)))))))

(deftest indexing-and-set-operations-test
  (t/with-torch
    (testing "cumulative maximum returns values and source indices"
      (let [[maxima indices] (t/cummax (t/tensor [1.0 3.0 2.0 4.0]) 0)]
        (is (= [1.0 3.0 3.0 4.0] (values maxima)))
        (is (= [0.0 1.0 1.0 3.0] (values indices)))))
    (testing "unique and isin support class-set validation"
      (let [classes (t/unique (t/tensor [2 1 2 3 1] {:dtype :int64}))]
        (is (= [1.0 2.0 3.0] (values classes)))
        (is (= [1.0 0.0]
               (values (t/to (t/isin (t/tensor [1 4] {:dtype :int64}) classes)
                             :float32))))))
    (testing "scatter leaves its input unchanged"
      (let [input (t/zeros [2 3])
            result (t/scatter input 1
                              (t/tensor [[0 2] [1 0]] {:dtype :int64})
                              (t/tensor [[5.0 6.0] [7.0 8.0]]))]
        (is (= [5.0 0.0 6.0 8.0 7.0 0.0] (values result)))
        (is (= [0.0 0.0 0.0 0.0 0.0 0.0] (values input)))))))

(deftest shape-and-one-hot-compatibility-test
  (t/with-torch
    (let [base (t/tensor [[1.0] [2.0]])
          expanded (t/expand-as base (t/zeros [2 3]))
          encoded (F/one-hot (t/tensor [2 0] {:dtype :int64}) 3)]
      (is (= 2 (count (t/size expanded))))
      (is (= 6 (count (values expanded))))
      (is (= [1.0 1.0 1.0 2.0 2.0 2.0]
             (values (t/contiguous expanded))))
      (is (= [0.0 0.0 1.0 1.0 0.0 0.0] (values encoded))))))
