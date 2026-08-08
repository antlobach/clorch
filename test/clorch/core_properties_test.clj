(ns clorch.core-properties-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clorch.torch :as t]))

;; --- Helpers ---

(defn- near? [a b]
  (let [diff (Math/abs (- (float a) (float b)))]
    (< diff 1e-3)))

(defn- tensor-near? [a b]
  (t/item-float (t/all (t/isclose a b {:atol 1e-4 :rtol 1e-4}))))

(def gen-shape (gen/vector (gen/choose 2 8) 1 4))

;; --- Comparison & Logical Properties ---

(defspec comparison-negation-prop 20
  (prop/for-all [shape gen-shape]
    (t/with-torch
      (let [a (t/randn shape)
            b (t/randn shape)
            eq-res (t/eq a b)
            ne-res (t/ne a b)
            ;; eq AND ne should be false everywhere
            combined (t/logical-and eq-res ne-res)]
        (zero? (t/item-float (t/any combined)))))))

(defspec comparison-le-ge-prop 20
  (prop/for-all [shape gen-shape]
    (t/with-torch
      (let [a (t/randn shape)
            b (t/randn shape)
            le-res (t/le a b)
            ge-res (t/ge a b)
            ;; (le a b) OR (ge a b) should always be true (or at least one must be true if a == b)
            combined (t/logical-or le-res ge-res)]
        (t/item-float (t/all combined))))))

(defspec logical-de-morgan-prop 20
  (prop/for-all [shape gen-shape]
    (t/with-torch
      (let [a (t/to (t/bernoulli (t/full shape 0.5)) :bool)
            b (t/to (t/bernoulli (t/full shape 0.5)) :bool)
            ;; Not(A or B) == Not(A) and Not(B)
            lhs (t/logical-not (t/logical-or a b))
            rhs (t/logical-and (t/logical-not a) (t/logical-not b))]
        (t/item-float (t/all (t/eq lhs rhs)))))))

;; --- Math Properties ---

(defspec square-pow-prop 20
  (prop/for-all [shape gen-shape]
    (t/with-torch
      (let [a (t/randn shape)
            res1 (t/square a)
            res2 (t/pow a 2.0)]
        (tensor-near? res1 res2)))))

(defspec lerp-boundary-prop 20
  (prop/for-all [shape gen-shape]
    (t/with-torch
      (let [start (t/randn shape)
            end (t/randn shape)
            res-0 (t/lerp start end 0.0)
            res-1 (t/lerp start end 1.0)]
        (and (tensor-near? res-0 start)
             (tensor-near? res-1 end))))))

(defspec addcmul-identity-prop 20
  (prop/for-all [shape gen-shape]
    (t/with-torch
      (let [a (t/randn shape)
            b (t/randn shape)
            c (t/randn shape)
            res (t/addcmul a b c :value 0.0)]
        (tensor-near? res a)))))

;; --- Shape & Manipulation Properties ---

(defspec clone-identity-prop 20
  (prop/for-all [shape gen-shape]
    (t/with-torch
      (let [a (t/randn shape)
            b (t/clone a)]
        (and (not= (t/->tensor a) (t/->tensor b)) ;; Different native objects
             (tensor-near? a b))))))

(defspec permute-inverse-prop 20
  (prop/for-all [shape (gen/vector (gen/choose 2 5) 2 4)]
    (t/with-torch
      (let [a (t/randn shape)
            dims (vec (range (count shape)))
            rev-dims (vec (reverse dims))
            permuted (t/permute a rev-dims)
            restored (t/permute permuted rev-dims)]
        (and (= (t/size restored) shape)
             (tensor-near? a restored))))))

(defspec split-with-sizes-sum-prop 20
  (prop/for-all [n (gen/choose 4 20)]
    (t/with-torch
      (let [a (t/randn [n])
            s1 (quot n 2)
            s2 (- n s1)
            parts (t/split-with-sizes a [s1 s2] 0)
            restored (t/cat parts 0)]
        (tensor-near? a restored)))))

(defspec type-as-prop 20
  (prop/for-all [shape gen-shape]
    (t/with-torch
      (let [a (t/randn shape) ;; float32
            b (t/zeros shape {:dtype :int64})
            c (t/type-as a b)]
        (= "Long" (.toString (.scalar_type (t/->tensor c))))))))
