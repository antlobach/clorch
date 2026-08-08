(ns clorch.einsum-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.autograd :as autograd]
            [clorch.einsum :refer [defein ein]]
            [clorch.torch :as t]))

(declare i j k y)

(def ^:private A* nil)
(def ^:private x* nil)

(defn- near?
  [a b]
  (< (Math/abs (- (float a) (float b))) 1e-4))

(defn- tensor-vec
  [tensor]
  (mapv t/item-float (t/tseq tensor)))

(deftest ein-lowers-to-torch-test
  (t/with-torch
    (testing "matrix-vector contraction"
      (let [A (t/tensor [[1.0 2.0 3.0]
                         [4.0 5.0 6.0]])
            x (t/tensor [10.0 20.0 30.0])
            out (ein [i] := (* (A i j) (x j)))
            expected (t/matmul A x)]
        (is (= [140.0 320.0] (tensor-vec out)))
        (is (= (tensor-vec expected) (tensor-vec out)))))

    (testing "outer product"
      (let [x (t/tensor [1.0 2.0 3.0])
            yv (t/tensor [4.0 5.0])
            out (ein [i j] := (* (x i) (yv j)))
            expected (t/outer x yv)]
        (is (= [3 2] (t/size out)))
        (is (= (mapv t/item-float (t/tseq (t/reshape expected [-1])))
               (mapv t/item-float (t/tseq (t/reshape out [-1])))))))

    (testing "trace"
      (let [A (t/tensor [[1.0 2.0]
                         [3.0 4.0]])
            tr (ein [] := (A i i))]
        (is (near? 5.0 (t/item-float tr)))))

    (testing "scalar factors"
      (let [A (t/tensor [[1.0 2.0]
                         [3.0 4.0]])
            x (t/tensor [5.0 6.0])
            out (ein [i] := (* 2.0 (A i j) (x j)))
            expected (t/mul (t/matmul A x) 2.0)]
        (is (= (tensor-vec expected) (tensor-vec out)))))))

(deftest defein-defines-var-test
  (t/with-torch
    (with-redefs [A* (t/tensor [[1.0 2.0]
                                [3.0 4.0]])
                  x* (t/tensor [7.0 11.0])]
      (defein clorch.einsum-test/y [i] := (* (A* i j) (x* j)))
      (is (= [29.0 65.0]
             (tensor-vec (var-get #'clorch.einsum-test/y)))))))

(deftest ein-shape-validation-test
  (t/with-torch
    (let [A (t/randn [2 3])
          x (t/randn [4])]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Incompatible dimension for einsum index"
           (ein [i] := (* (A i j) (x j))))))))

(deftest ein-macro-validation-test
  (let [thrown (try
                 (eval '(clorch.einsum/ein [k] := (* (A i j) (x j))))
                 nil
                 (catch Throwable t t))]
    (is (instance? clojure.lang.Compiler$CompilerException thrown))
    (is (instance? clojure.lang.ExceptionInfo (ex-cause thrown)))
    (is (re-find #"Output index does not appear in RHS tensor terms"
                 (ex-message (ex-cause thrown))))))

(deftest ein-autograd-test
  (t/with-torch
    (let [A (t/tensor [[1.0 2.0]
                       [3.0 4.0]] {:requires-grad true})
          x (t/tensor [5.0 7.0] {:requires-grad true})
          out (ein [i] := (* (A i j) (x j)))
          loss (t/sum out)]
      (autograd/backward loss)
      (let [gA (autograd/grad A)
            gx (autograd/grad x)]
        (is (= [2 2] (t/size gA)))
        (is (= [2] (t/size gx)))
        (is (near? 5.0 (t/item-float (t/ix gA 0 0))))
        (is (near? 7.0 (t/item-float (t/ix gA 0 1))))
        (is (near? 5.0 (t/item-float (t/ix gA 1 0))))
        (is (near? 7.0 (t/item-float (t/ix gA 1 1))))
        (is (near? 4.0 (t/item-float (t/ix gx 0))))
        (is (near? 6.0 (t/item-float (t/ix gx 1))))))))
