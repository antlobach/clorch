(ns clorch.distributions-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.distributions :as dist]
            [clorch.torch :as t]))

(defn near? [a b]
  (< (Math/abs (- (double a) (double b))) 1e-4))

(defn finite-number? [x]
  (and (not (Double/isNaN (double x)))
       (not (Double/isInfinite (double x)))))

(defn tensor-finite? [x]
  (every? finite-number? (map t/item-float (t/tseq (t/reshape x [-1])))))

(deftest distributions-sampling-smoke-test
  (t/with-torch
    (testing "sample shapes"
      (is (= [4 3] (t/size (dist/sample (dist/normal 0.0 1.0) [4 3]))))
      (is (= [4] (t/size (dist/sample (dist/log-normal 0.0 0.5) [4]))))
      (is (= [2] (t/size (dist/sample (dist/bernoulli (t/tensor [0.2 0.8]))))))
      (is (= [1] (t/size (dist/sample (dist/categorical (t/tensor [0.1 0.2 0.7]))))))
      (is (= [5] (t/size (dist/sample (dist/binomial 10.0 0.3) [5]))))
      (is (= [2 2] (t/size (dist/sample (dist/uniform -1.0 2.0) [2 2]))))
      (is (= [5] (t/size (dist/sample (dist/poisson 3.0) [5]))))
      (is (= [5] (t/size (dist/sample (dist/exponential 2.0) [5]))))
      (is (= [3] (t/size (dist/sample (dist/cauchy 0.0 1.0) [3]))))
      (is (= [3] (t/size (dist/sample (dist/geometric 0.4) [3]))))
      (is (= [3] (t/size (dist/sample (dist/gumbel 0.0 1.0) [3]))))
      (is (= [3] (t/size (dist/sample (dist/laplace 0.0 1.0) [3]))))
      (is (= [6] (t/size (dist/sample (dist/gamma 2.0 3.0) [6]))))
      (is (= [6] (t/size (dist/sample (dist/beta 2.0 5.0) [6]))))
      (is (= [4 3] (t/size (dist/sample (dist/dirichlet [1.0 2.0 3.0]) [4]))))
      (is (= [6] (t/size (dist/sample (dist/student-t 7.0 0.0 1.0) [6]))))
      (is (= [6] (t/size (dist/sample (dist/chi2 4.0) [6]))))
      (is (= [6] (t/size (dist/sample (dist/fisher-f 5.0 9.0) [6]))))
      (is (= [6] (t/size (dist/sample (dist/weibull 2.0 3.0) [6]))))
      (is (= [6] (t/size (dist/sample (dist/pareto 1.0 3.0) [6]))))
      (is (= [6] (t/size (dist/sample (dist/logistic 0.0 1.0) [6]))))
      (is (= [6] (t/size (dist/sample (dist/negative-binomial 7.0 0.4) [6]))))))

  (testing "dirichlet samples are on simplex"
    (t/with-torch
      (let [s (dist/sample (dist/dirichlet [0.7 1.1 2.3]) [8])
            row-sums (t/sum s 1)]
        (is (tensor-finite? s))
        (doseq [v (map t/item-float (t/tseq row-sums))]
          (is (near? 1.0 v)))))))

(deftest distributions-log-prob-test
  (t/with-torch
    (testing "known values"
      (is (near? -0.9189385 (t/item-float (dist/log-prob (dist/normal 0.0 1.0) 0.0))))
      (is (near? (Math/log 0.7) (t/item-float (dist/log-prob (dist/categorical (t/tensor [0.1 0.2 0.7])) 2))))
      (is (near? (Math/log 0.8)
                 (t/item-float (dist/log-prob (dist/bernoulli (t/tensor [0.8])) (t/tensor [1.0])))))
      (is (near? (- (Math/log 2.0)) (double (dist/log-prob (dist/uniform 0.0 2.0) 0.5))))
      (is (near? (- (* 2.0 (Math/log 3.0)) 3.0 (Math/log 2.0))
                 (t/item-float (dist/log-prob (dist/poisson 3.0) 2.0))))))

  (t/with-torch
    (testing "extended distributions produce finite log-prob"
      (let [lps [(dist/log-prob (dist/gamma 2.0 3.0) 0.7)
                 (dist/log-prob (dist/beta 2.0 5.0) 0.3)
                 (dist/log-prob (dist/student-t 7.0 0.0 1.0) 0.2)
                 (dist/log-prob (dist/chi2 4.0) 2.5)
                 (dist/log-prob (dist/fisher-f 5.0 9.0) 1.1)
                 (dist/log-prob (dist/weibull 2.0 3.0) 1.5)
                 (dist/log-prob (dist/pareto 1.0 3.0) 2.0)
                 (dist/log-prob (dist/logistic 0.0 1.0) 0.2)
                 (dist/log-prob (dist/negative-binomial 7.0 0.4) 3.0)]
            dirichlet-lp (dist/log-prob (dist/dirichlet [1.1 2.2 3.3]) [0.2 0.3 0.5])]
        (doseq [lp lps]
          (is (finite-number? (t/item-float lp))))
        (is (finite-number? (t/item-float dirichlet-lp)))))))

(deftest distributions-moments-test
  (testing "mean/variance"
    (is (near? 0.0 (double (dist/mean (dist/normal 0.0 1.0)))))
    (is (near? 1.0 (double (dist/variance (dist/normal 0.0 1.0)))))
    (is (near? 3.0 (double (dist/mean (dist/binomial 10.0 0.3)))))
    (is (near? 2.1 (double (dist/variance (dist/binomial 10.0 0.3)))))
    (is (near? 1.5 (double (dist/mean (dist/uniform 1.0 2.0)))))
    (is (near? (/ 1.0 12.0) (double (dist/variance (dist/uniform 1.0 2.0)))))
    (is (near? 0.5 (double (dist/mean (dist/exponential 2.0)))))
    (is (near? 0.25 (double (dist/variance (dist/exponential 2.0)))))
    (is (near? (/ 1.0 0.4) (double (dist/mean (dist/geometric 0.4)))))
    (is (near? (/ 0.6 (* 0.4 0.4)) (double (dist/variance (dist/geometric 0.4)))))
    (is (near? (/ 2.0 3.0) (double (dist/mean (dist/gamma 2.0 3.0)))))
    (is (near? (/ 2.0 9.0) (double (dist/variance (dist/gamma 2.0 3.0)))))
    (is (near? (/ 2.0 7.0) (double (dist/mean (dist/beta 2.0 5.0)))))
    (is (near? (/ 10.0 (* 49.0 8.0)) (double (dist/variance (dist/beta 2.0 5.0)))))
    (is (near? 4.0 (double (dist/mean (dist/chi2 4.0)))))
    (is (near? 8.0 (double (dist/variance (dist/chi2 4.0)))))
    (is (near? 1.8 (double (dist/mean (dist/negative-binomial 6.0 0.7692307692307693)))))
    (is (near? 2.34 (double (dist/variance (dist/negative-binomial 6.0 0.7692307692307693))))))

  (testing "vector-valued moments"
    (t/with-torch
      (let [m (dist/mean (dist/dirichlet [2.0 3.0 5.0]))
            v (dist/variance (dist/dirichlet [2.0 3.0 5.0]))]
        (is (= [3] (t/size m)))
        (is (= [3] (t/size v)))
        (is (near? 0.2 (t/item-float (t/ix m 0))))
        (is (near? 0.3 (t/item-float (t/ix m 1))))
        (is (near? 0.5 (t/item-float (t/ix m 2))))))))
