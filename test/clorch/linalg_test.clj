(ns clorch.linalg-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.linalg :as linalg]
            [clorch.torch :as t]))

(defn near? [a b]
  (< (Math/abs (- (double a) (double b))) 1e-5))

(deftest linalg-namespace-smoke-test
  (t/with-torch
    (testing "norm and determinant under clorch.linalg namespace"
      (let [x (t/tensor [3.0 4.0])
            m (t/tensor [[1.0 2.0] [3.0 4.0]])]
        (is (near? 5.0 (t/item-float (linalg/norm x))))
        (is (near? -2.0 (t/item-float (linalg/det m))))))

    (testing "solve and qr are routed through user-facing names"
      (let [a (t/tensor [[3.0 1.0] [1.0 2.0]])
            b (t/tensor [[9.0] [8.0]])
            x (linalg/solve a b)
            [q r] (linalg/qr a)]
        (is (= [2 1] (t/size x)))
        (is (= [2 2] (t/size q)))
        (is (= [2 2] (t/size r)))))))

(deftest linalg-decomposition-and-solver-test
  (t/with-torch
    (let [a (t/tensor [[3.0 1.0] [1.0 2.0]])
          b (t/tensor [[9.0] [8.0]])
          spd (t/tensor [[4.0 1.0] [1.0 3.0]])]
      (testing "SVD and eig families"
        (let [[u s vh] (linalg/svd a {:full-matrices false})
              [evals evecs] (linalg/eig a)
              [hevals hevecs] (linalg/eigh spd {:uplo :L})]
          (is (= [2 2] (t/size u)))
          (is (= [2] (t/size s)))
          (is (= [2 2] (t/size vh)))
          (is (= [2] (t/size evals)))
          (is (= [2 2] (t/size evecs)))
          (is (= [2] (t/size hevals)))
          (is (= [2 2] (t/size hevecs)))
          (is (= [2] (t/size (linalg/eigvals a))))
          (is (= [2] (t/size (linalg/eigvalsh spd))))
          (is (= [2] (t/size (linalg/svdvals a))))))

      (testing "LU/LDL and triangular/least-squares"
        (let [[_p l u] (linalg/lu a)
              [lu piv] (linalg/lu-factor a)
              [ldl piv2] (linalg/ldl-factor spd)
              x-lu (linalg/lu-solve lu piv b)
              x-ldl (linalg/ldl-solve ldl piv2 b)
              x-tri (linalg/solve-triangular (t/triu a) b true {:left true})
              ls (linalg/lstsq a b)
              ls2 (linalg/least-squares a b {:rcond 1e-8})]
          (is (= [2 2] (t/size l)))
          (is (= [2 2] (t/size u)))
          (is (= [2 1] (t/size x-lu)))
          (is (= [2 1] (t/size x-ldl)))
          (is (= [2 1] (t/size x-tri)))
          (is (= [:rank :residuals :singular-values :solution]
                 (sort (keys ls))))
          (is (= [2 1] (t/size (:solution ls))))
          (is (= [2 1] (t/size (:solution ls2))))))

      (testing "matrix-rank and condition number options"
        (let [r0 (linalg/matrix-rank a)
              r1 (linalg/matrix-rank a 1e-8 {:hermitian false})
              r2 (linalg/matrix-rank a {:atol 1e-8 :rtol 1e-5})
              c0 (linalg/cond a)
              c1 (linalg/cond a 2)
              c2 (linalg/cond a :fro)]
          (is (= 2.0 (t/item-float r0)))
          (is (= 2.0 (t/item-float r1)))
          (is (= 2.0 (t/item-float r2)))
          (is (> (t/item-float c0) 1.0))
          (is (> (t/item-float c1) 1.0))
          (is (> (t/item-float c2) 1.0)))))))

(deftest linalg-batched-smoke-test
  (t/with-torch
    (let [batched (t/tensor [[[3.0 1.0] [1.0 2.0]]
                             [[4.0 2.0] [2.0 5.0]]])
          [u s vh] (linalg/svd batched {:full-matrices false})
          ranks (linalg/matrix-rank batched)]
      (is (= [2 2 2] (t/size u)))
      (is (= [2 2] (t/size s)))
      (is (= [2 2 2] (t/size vh)))
      (is (= [2] (t/size ranks))))))

(deftest linalg-extra-ops-test
  (t/with-torch
    (let [a (t/tensor [[3.0 1.0] [1.0 2.0]])
          b (t/tensor [[9.0] [8.0]])
          spd (t/tensor [[4.0 1.0] [1.0 3.0]])
          [x info] (linalg/solve-ex a b)
          [inv inv-info] (linalg/inv-ex a)
          [chol chol-info] (linalg/cholesky-ex spd)
          [lu-data piv lu-info] (linalg/lu-factor-ex a)
          [ldl piv2 ldl-info] (linalg/ldl-factor-ex spd)
          [p l u] (linalg/lu-unpack lu-data piv)
          vander (linalg/vander (t/tensor [1.0 2.0 3.0]) 4)
          vecdot (linalg/vecdot (t/tensor [1.0 2.0 3.0]) (t/tensor [4.0 5.0 6.0]))
          tid (t/reshape (t/eye 4) [2 2 2 2])
          tensorinv (linalg/tensorinv tid 2)
          tensorsolve (linalg/tensorsolve tid (t/tensor [[1.0 2.0] [3.0 4.0]]))
          multi (linalg/multi-dot [(t/tensor [[1.0 2.0] [3.0 4.0]])
                                   (t/tensor [[5.0 6.0] [7.0 8.0]])])]
      (is (= [2 1] (t/size x)))
      (is (= [] (t/size info)))
      (is (= [2 2] (t/size inv)))
      (is (= [] (t/size inv-info)))
      (is (= [2 2] (t/size chol)))
      (is (= [] (t/size chol-info)))
      (is (= [2 2] (t/size lu-data)))
      (is (= [2] (t/size piv)))
      (is (= [] (t/size lu-info)))
      (is (= [2 2] (t/size ldl)))
      (is (= [2] (t/size piv2)))
      (is (= [] (t/size ldl-info)))
      (is (= [2 2] (t/size p)))
      (is (= [2 2] (t/size l)))
      (is (= [2 2] (t/size u)))
      (is (= [3 4] (t/size vander)))
      (is (= [] (t/size vecdot)))
      (is (= [2 2 2 2] (t/size tensorinv)))
      (is (= [2 2] (t/size tensorsolve)))
      (is (= [2 2] (t/size multi))))))

(deftest linalg-batch2-aliases-test
  (t/with-torch
    (let [spd (t/tensor [[4.0 1.0] [1.0 3.0]])
          chol (linalg/cholesky spd)
          inv (linalg/cholesky-inverse chol)
          b (t/tensor [[1.0] [2.0]])
          x (linalg/cholesky-solve b chol)
          recon (t/matmul spd x)
          diag (linalg/diagonal spd)
          mexp (linalg/matrix-exp (t/tensor [[0.0 0.0] [0.0 0.0]]))
          cross (linalg/cross (t/tensor [1.0 0.0 0.0])
                              (t/tensor [0.0 1.0 0.0]))]
      (is (= [2 2] (t/size inv)))
      (is (= [2 1] (t/size x)))
      (is (t/allclose b recon))
      (is (= [2] (t/size diag)))
      (is (t/allclose mexp (t/eye 2)))
      (is (t/allclose cross (t/tensor [0.0 0.0 1.0]))))))

(deftest linalg-batch4-triangular-solve-alias-test
  (t/with-torch
    (let [a (t/tensor [[3.0 1.0] [0.0 2.0]])
          b (t/tensor [[7.0] [4.0]])
          x (linalg/triangular-solve a b true {:left true})]
      (is (= [2 1] (t/size x)))
      (is (t/allclose (t/matmul a x) b)))))
