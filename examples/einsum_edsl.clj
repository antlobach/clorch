(ns examples.einsum-edsl
  (:require [clorch.einsum :refer [ein]]
            [clorch.torch :as t]))

(declare i j)

(def matrix-a (t/tensor [[1.0 2.0 3.0]
                         [4.0 5.0 6.0]]))
(def matrix-x (t/tensor [10.0 20.0 30.0]))
(def matrix-y (ein [i] := (* (matrix-a i j) (matrix-x j))))
(mapv t/item-float (t/tseq matrix-y))

(def outer-x (t/tensor [1.0 2.0 3.0]))
(def outer-y (t/tensor [4.0 5.0]))
(def outer-product (ein [i j] := (* (outer-x i) (outer-y j))))
outer-product

(def trace-a (t/tensor [[1.0 2.0]
                        [3.0 4.0]]))
(def trace-value (ein [] := (trace-a i i)))
(t/item-float trace-value)

(def scaled-a (t/tensor [[1.0 2.0]
                         [3.0 4.0]]))
(def scaled-x (t/tensor [5.0 6.0]))
(def scaled-y (ein [i] := (* 0.5 (scaled-a i j) (scaled-x j))))
scaled-y
