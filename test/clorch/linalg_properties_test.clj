(ns clorch.linalg-properties-test
  (:require [clorch.torch :as t]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec linalg-inv-prop 20
  (prop/for-all [n (gen/choose 2 10)]
    (t/with-torch
      (let [A (t/add (t/randn [n n]) (t/mul (t/eye n) 0.1))
            A-inv (t/linalg-inv A)
            res (t/matmul A A-inv)
            id (t/eye n)]
        (t/allclose res id {:atol 1e-4})))))

(defspec linalg-det-identity-prop 20
  (prop/for-all [n (gen/choose 1 10)]
    (t/with-torch
      (let [id (t/eye n)
            d (t/linalg-det id)]
        (< (Math/abs (- (t/item-float d) 1.0)) 1e-5)))))

(defspec linalg-solve-prop 20
  (prop/for-all [n (gen/choose 2 8)]
    (t/with-torch
      (let [A (t/randn [n n])
            x (t/randn [n 1])
            b (t/matmul A x)
            x-solved (t/linalg-solve A b)]
        (t/allclose x x-solved {:atol 1e-4})))))
