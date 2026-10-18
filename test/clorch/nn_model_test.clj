(ns clorch.nn-model-test
  (:require [clorch.torch :as t]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clojure.test :refer [deftest is testing]]))

(deftest model-lifecycle-test
  (t/with-torch
    (let [model (nn/sequential
                 (nn/linear 10 20)
                 (nn/relu)
                 (nn/linear 20 5))]
      (testing "Parameters collection"
        (let [params (nn/parameters model)]
          (is (= 4 (.size params))))) ;; 2 layers * (weight + bias)

      (testing "State dict"
        (let [sd (nn/state-dict model)]
          (is (map? sd))
          (is (= 3 (count sd))))) ;; sequential has 3 children (Linear, ReLU, Linear)

      (testing "To float/double"
        (let [m-double (nn/to model :float64)
              params (t/->vector (nn/parameters m-double))]
          (is (every? #(= :float64 (t/dtype %)) params)))))))

(deftest custom-model-test
  (t/with-torch
    #_{:clj-kondo/ignore [:unresolved-symbol]}
    (nn/defmodel MyModel [in out]
      [l1 (nn/linear in 16)
       l2 (nn/linear 16 out)
       norm (nn/layernorm out)
       dropout (nn/dropout 0.5)]
      (forward [x]
               (let [x1 (nn/forward l1 x)
                     x2 (F/relu x1)]
                 (nn/forward dropout
                             (nn/forward norm (nn/forward l2 x2))))))

    (let [m (MyModel 10 5)
          x (t/randn [2 10])
          out (nn/forward m x)]
      (is (= [2 5] (t/size out)))
      (is (= 6 (.size (nn/parameters m))))
      (testing "Training mode reaches modules without treating parameters as modules"
        (nn/train m false)
        (is (t/allclose (nn/forward m x) (nn/forward m x))))
      (testing "Device and dtype conversion reaches defmodel fields"
        (let [m-double (nn/to m :float64)
              params (t/->vector (nn/parameters m-double))]
          (is (every? #(= :float64 (t/dtype %)) params))
          (is (every? #(.is_leaf ^org.bytedeco.pytorch.Tensor %) params)))))))

(deftest flat-state-dict-loading-test
  (t/with-torch
    (let [model (nn/sequential (nn/linear 3 2))
          names (nn/flat-parameter-names model)
          state {"[0].weight" (t/full [2 3] 2.0)
                 "[0].bias" (t/full [2] -1.0)}
          loaded (nn/load-flat-state-dict! model state)
          output (nn/forward model (t/ones [1 3]))]
      (is (= #{"[0].weight" "[0].bias"} names loaded))
      (is (= 10.0 (double (t/item-float (t/sum output))))))))

(deftest model-freeze-and-snapshot-test
  (t/with-torch
    (let [model (nn/sequential (nn/linear 2 1))]
      (testing "model sections can be frozen and unfrozen"
        (nn/set-trainable! model false)
        (is (zero? (.size (nn/parameters model :trainable-only true))))
        (nn/set-trainable! model true)
        (is (= 2 (.size (nn/parameters model :trainable-only true)))))
      (testing "a copied state restores parameters after later updates"
        (let [snapshot (nn/clone-flat-state-dict model)]
          (try
            (nn/load-flat-state-dict!
             model
             {"[0].weight" (t/full [1 2] 3.0)
              "[0].bias" (t/full [1] 4.0)})
            (nn/load-flat-state-dict! model snapshot)
            (is (= (set (keys snapshot)) (nn/flat-parameter-names model)))
            (is (every? (fn [[parameter-name value]]
                          (t/item-float
                           (t/all (t/eq value
                                        (get (nn/named-parameters model) parameter-name)))))
                        snapshot))
            (finally
              (t/release! snapshot))))))))

(deftest bias-free-layernorm-test
  (t/with-torch
    (let [layer (nn/layernorm 3 :bias? false)
          output (nn/forward layer (t/tensor [[1.0 2.0 3.0]]))]
      (is (= 1 (.size (nn/parameters layer))))
      (is (= [1 3] (t/size output)))
      (is (< (Math/abs (t/item-float (t/mean output))) 1.0e-6)))))

(deftest training-mode-ignores-configuration-values
  (let [dropout (nn/dropout 0.5)
        model {:layers [dropout 1.0] :config {:dropout 0.5}}]
    (is (identical? model (nn/train model false)))
    (is (identical? model (nn/train model true)))))
