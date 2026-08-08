(ns slicing-examples
  (:require [clorch.torch :as torch]))

(defn tensor->vec [t]
  (mapv torch/item-float (torch/tseq t)))

(defn tensor->vecs [t]
  (mapv #(mapv torch/item-float (torch/tseq %)) (torch/tseq t)))

(def base-1d (torch/tensor [10 11 12 13 14]))
(torch/item-float (torch/ix base-1d 0))
(torch/item-float (torch/ix base-1d -1))
(tensor->vec (torch/ix base-1d [1 4]))
(tensor->vec (torch/ix base-1d [nil nil 2]))
(tensor->vec (torch/ix base-1d [nil nil -1]))
(tensor->vec (torch/ix base-1d [3 nil -1]))

(def base-2d (torch/tensor [[1 2 3]
                            [4 5 6]
                            [7 8 9]]))
(tensor->vecs base-2d)
(tensor->vec (torch/ix base-2d 0))
(torch/item-float (torch/ix base-2d 0 1))
(tensor->vec (torch/ix base-2d :_ 1))
(tensor->vecs (torch/ix base-2d [0 2] [1 3]))
(tensor->vecs (torch/ix base-2d [nil nil -1] :_))
(tensor->vecs (torch/ix base-2d :_ [nil nil -1]))

(def base-3d (torch/reshape (torch/tensor (range 24)) [2 3 4]))
(torch/size base-3d)
(torch/size (torch/ix base-3d 0))
(torch/item-float (torch/ix base-3d 1 2 3))
(torch/size (torch/ix base-3d (quote ...) 0))
(torch/size (torch/ix base-3d 0 (quote ...)))

(def fancy-matrix (torch/tensor [[1 2 3 4]
                                 [5 6 7 8]
                                 [9 10 11 12]
                                 [13 14 15 16]]))
(def fancy-idx (torch/tensor [0 2] {:dtype :int64}))
(tensor->vecs (torch/ix fancy-matrix fancy-idx :_))
(tensor->vecs (torch/ix fancy-matrix :_ fancy-idx))
(tensor->vec (torch/ix (torch/tensor [1 2 3 4 5])
                       (torch/tensor [false false true true true] {:dtype :bool})))

(def sequence-data (torch/tensor (range 20)))
(tensor->vec (torch/ix sequence-data [0 5]))
(tensor->vec (torch/ix sequence-data [3 8]))
(tensor->vec (torch/ix sequence-data [6 11]))

(def image (torch/reshape (torch/tensor (range 16)) [4 4]))
(tensor->vecs image)
(tensor->vecs (torch/ix image [0 2] [0 2]))
(tensor->vecs (torch/ix image [2 4] [2 4]))

(def train-test-data (torch/tensor (range 100)))
(def train-split (torch/ix train-test-data [0 80]))
(def test-split (torch/ix train-test-data [80 nil]))
(torch/size train-split)
(torch/size test-split)
