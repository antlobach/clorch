(ns clorch.nn-init-test
  (:require [clorch.torch :as t]
            [clorch.nn.init :as init]
            [clojure.test :refer [deftest is testing]]))

(defn near? [a b]
  (let [eps 1e-4]
    (< (Math/abs (- (float a) (float b))) eps)))

(deftest init-smoke-test
  (t/with-torch
    (let [x (t/zeros [10 10])]
      (testing "ones!"
        (init/ones! x)
        (is (every? #(= 1.0 %) (mapv t/item-float (t/tseq (t/reshape x [-1]))))))
      
      (testing "zeros!"
        (init/zeros! x)
        (is (every? #(= 0.0 %) (mapv t/item-float (t/tseq (t/reshape x [-1]))))))
      
      (testing "constant!"
        (init/constant! x 3.14)
        (is (every? #(near? % 3.14) (mapv t/item-float (t/tseq (t/reshape x [-1]))))))
      
      (testing "uniform!"
        (init/uniform! x -1 1)
        (let [v (mapv t/item-float (t/tseq (t/reshape x [-1])))]
          (is (every? #(and (>= % -1) (<= % 1)) v))))
      
      (testing "normal!"
        (init/normal! x 0 1)
        (is (some? x)))
      
      (testing "xavier-uniform!"
        (init/xavier-uniform! x)
        (is (some? x)))
      
      (testing "kaiming-normal!"
        (init/kaiming-normal! x :non-linearity :relu)
        (is (some? x))))))
