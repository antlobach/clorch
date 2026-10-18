(ns clorch.linalg-properties-test
  (:require [clorch.torch :as t]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(def ^:private invertible-matrix-gen
  (gen/bind (gen/choose 2 10)
            (fn [n]
              (gen/fmap (fn [values] [n values])
                        (gen/vector (gen/choose -100 100) (* n n))))))

(defspec linalg-inv-prop 20
  (prop/for-all [[n values] invertible-matrix-gen]
                (t/with-torch
                  (let [matrix (t/reshape (t/to (t/tensor values) :float32) [n n])
                        matrix (t/div matrix 100.0)
                        diagonal (t/add (t/sum (t/abs matrix) 1) 1.0)
                        a (t/add matrix (t/diag diagonal))
                        actual (t/matmul a (t/linalg-inv a))]
                    (t/allclose actual (t/eye n) {:atol 1e-4})))))

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
