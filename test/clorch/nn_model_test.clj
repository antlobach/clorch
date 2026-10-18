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
       l2 (nn/linear 16 out)]
      (forward [x]
               (let [x1 (nn/forward l1 x)
                     x2 (F/relu x1)]
                 (nn/forward l2 x2))))

    (let [m (MyModel 10 5)
          x (t/randn [2 10])
          out (nn/forward m x)]
      (is (= [2 5] (t/size out)))
      (is (= 4 (.size (nn/parameters m))))
      (testing "Device and dtype conversion reaches defmodel fields"
        (let [m-double (nn/to m :float64)]
          (is (every? #(= :float64 (t/dtype %))
                      (t/->vector (nn/parameters m-double)))))))))

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
