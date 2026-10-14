(ns clorch.nn-transformer-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.nn :as nn]
            [clorch.torch :as t]))

(set! *warn-on-reflection* true)

(defn- max-error [left right]
  (t/item-float (t/max (t/abs (t/sub left right)))))

(deftest interleaved-rope-rotates-adjacent-pairs
  (t/with-torch
    (let [x (t/tensor [[[[1.0 2.0 3.0 4.0]]]])
          cosines (t/zeros [1 4])
          sines (t/ones [1 4])
          half-split (t/apply-rope x cosines sines)
          interleaved (t/apply-rope x cosines sines :interleaved? true)]
      (is (< (max-error half-split
                        (t/tensor [[[[-3.0 -4.0 1.0 2.0]]]]))
             1.0e-6))
      (is (< (max-error interleaved
                        (t/tensor [[[[-2.0 1.0 -4.0 3.0]]]]))
             1.0e-6)))))

(deftest cached-cross-attention-matches-full-attention
  (t/manual-seed 11)
  (let [attention (nn/multihead-attention 8 2 :dropout 0.0)
        query (t/randn [2 3 8])
        context (t/randn [2 5 8])
        full (nn/forward attention {:query query
                                    :key context
                                    :value context
                                    :return-kv? true})]
    (t/with-torch
      (let [cached (nn/forward attention
                               {:query query
                                :cached-key (:key full)
                                :cached-value (:value full)})]
        (is (= [2 3 8] (t/size cached)))
        (is (< (max-error (:output full) cached) 1.0e-6))))))

(deftest key-padding-mask-excludes-padded-values
  (t/manual-seed 13)
  (let [attention (nn/multihead-attention 8 2 :dropout 0.0)
        query (t/randn [2 3 8])
        context (t/randn [2 5 8])
        changed (t/clone context)
        _ (t/copy- (t/ix changed 1 [3 5] :_) (t/full [2 8] 1000.0))
        padding-mask (t/tensor [[false false false false false]
                                [false false false true true]]
                               {:dtype :bool})]
    (t/with-torch
      (let [baseline (nn/forward attention {:query query
                                            :key context
                                            :value context
                                            :key-padding-mask padding-mask})
            modified (nn/forward attention {:query query
                                            :key changed
                                            :value changed
                                            :key-padding-mask padding-mask})]
        (testing "unmasked first row remains ordinary cross-attention"
          (is (= [2 3 8] (t/size baseline))))
        (testing "masked values cannot affect the second row"
          (is (< (max-error (t/ix baseline 1 :_ :_)
                            (t/ix modified 1 :_ :_))
                 1.0e-5)))))))

(deftest transformer-encoder-runs-pre-and-post-norm
  (t/manual-seed 17)
  (t/with-torch
    (let [x (t/randn [2 4 8])
          pre-layer (nn/transformer-encoder-layer
                     8 2 :feedforward-dim 16 :dropout 0.0
                     :activation :gelu :norm-first? true)
          post-layer (nn/transformer-encoder-layer
                      8 2 :feedforward-dim 16 :dropout 0.0
                      :activation :relu :norm-first? false)
          encoder (nn/transformer-encoder [pre-layer post-layer])]
      (is (= [2 4 8] (t/size (nn/forward pre-layer x))))
      (is (= [2 4 8] (t/size (nn/forward post-layer x))))
      (is (= [2 4 8] (t/size (nn/forward encoder x)))))))

(deftest eval-mode-disables-attention-dropout
  (t/manual-seed 19)
  (let [attention (nn/multihead-attention 8 2 :dropout 0.75)
        x (t/randn [2 5 8])]
    (nn/train attention false)
    (t/with-torch
      (let [first-output (nn/forward attention x)
            second-output (nn/forward attention x)]
        (is (< (max-error first-output second-output) 1.0e-7))))))
