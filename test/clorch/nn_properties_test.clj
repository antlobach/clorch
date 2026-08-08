(ns clorch.nn-properties-test
  (:require [clorch.torch :as t]
            [clorch.nn :as nn]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defn- near? [a b eps]
  (< (Math/abs (- (double a) (double b))) eps))

(defn- conv2d-out-size [in-size kernel-size stride padding]
  (+ (quot (- (+ in-size (* 2 padding)) kernel-size) stride) 1))

(defspec layernorm-moments-prop 20
  (prop/for-all [batch-size (gen/choose 1 16)
                 features (gen/choose 32 64)]
    (t/with-torch
      (let [ln (nn/layernorm features)
            input (t/randn [batch-size features])
            output (nn/forward ln input)
            mean (t/mean output 1)
            var (t/var output 1 :unbiased false)]
        (and (every? #(near? % 0.0 2e-3) (mapv t/item-float (t/tseq mean)))
             (every? #(near? % 1.0 5e-3) (mapv t/item-float (t/tseq var))))))))

(defspec batchnorm-moments-prop 20
  (prop/for-all [batch-size (gen/choose 32 64)
                 features (gen/choose 1 64)]
    (t/with-torch
      (let [bn (nn/batchnorm1d features)
            _ (nn/train bn true)
            input (t/randn [batch-size features])
            output (nn/forward bn input)
            mean (t/mean output 0)
            var (t/var output 0 :unbiased false)]
        (and (every? #(near? % 0.0 2e-3) (mapv t/item-float (t/tseq mean)))
             (every? #(near? % 1.0 5e-3) (mapv t/item-float (t/tseq var))))))))

(defspec conv2d-shape-prop 20
  (prop/for-all [in-c (gen/choose 1 8)
                 out-c (gen/choose 1 8)
                 in-h (gen/choose 8 32)
                 in-w (gen/choose 8 32)
                 k-h (gen/choose 1 7)
                 k-w (gen/choose 1 7)
                 stride-h (gen/choose 1 2)
                 stride-w (gen/choose 1 2)
                 pad-h (gen/choose 0 3)
                 pad-w (gen/choose 0 3)]
    (t/with-torch
      (let [m (nn/conv2d in-c out-c [k-h k-w] :stride [stride-h stride-w] :padding [pad-h pad-w])
            input (t/randn [1 in-c in-h in-w])
            output (nn/forward m input)
            out-shape (t/size output)
            expected-h (conv2d-out-size in-h k-h stride-h pad-h)
            expected-w (conv2d-out-size in-w k-w stride-w pad-w)]
        (= out-shape [1 out-c expected-h expected-w])))))

(def finite-double-gen (gen/double* {:infinite? false :NaN? false}))

(defspec linear-linearity-prop 20
  (prop/for-all [in (gen/choose 1 16)
                 out (gen/choose 1 16)
                 a finite-double-gen
                 b finite-double-gen]
    (t/with-torch
      (let [m (nn/linear in out :bias false)
            x1 (t/randn [1 in])
            x2 (t/randn [1 in])
            y1 (nn/forward m x1)
            y2 (nn/forward m x2)
            x-combined (t/add (t/mul x1 a) (t/mul x2 b))
            y-combined (nn/forward m x-combined)
            expected (t/add (t/mul y1 a) (t/mul y2 b))]
        (t/allclose y-combined expected {:atol 1e-4})))))
