(ns compare-transformer
  (:require [clojure.data.json :as json]
            [clorch.nn :as nn]
            [clorch.torch :as t]))

(set! *warn-on-reflection* true)

(def specs
  {"attention.in-projection.weight" [0.002 -0.10]
   "attention.in-projection.bias" [0.003 -0.05]
   "attention.out-projection.weight" [0.004 -0.10]
   "attention.out-projection.bias" [0.005 0.02]
   "feedforward[0].weight" [0.001 -0.08]
   "feedforward[0].bias" [0.002 0.01]
   "feedforward[3].weight" [0.0015 -0.06]
   "feedforward[3].bias" [0.003 -0.02]
   "norm1.gamma" [0.001 1.0]
   "norm1.beta" [0.002 -0.01]
   "norm2.gamma" [0.0015 0.9]
   "norm2.beta" [0.001 0.02]})

(def attention-specs
  {"in-projection.weight" [0.002 -0.10]
   "in-projection.bias" [0.003 -0.05]
   "out-projection.weight" [0.004 -0.10]
   "out-projection.bias" [0.005 0.02]})

(defn- initialize! [model parameter-specs]
  (doseq [[parameter-name tensor] (nn/named-parameters model)]
    (let [[scale offset] (get parameter-specs parameter-name)
          shape (t/size tensor)
          values (t/to (t/add (t/mul (t/arange (reduce * 1 shape)) scale)
                              offset)
                       :float32)]
      (when-not scale
        (throw (ex-info "Missing deterministic parameter specification"
                        {:parameter parameter-name})))
      (t/copy- tensor (t/reshape values shape))))
  model)

(defn- max-error [actual expected]
  (t/item-float (t/max (t/abs (t/sub actual (t/tensor expected))))))

(defn -main [reference-path]
  (let [reference (json/read-str (slurp reference-path))
        attention (initialize!
                   (nn/multihead-attention 8 2 :dropout 0.0)
                   attention-specs)
        layer (initialize!
               (nn/transformer-encoder-layer
                8 2 :feedforward-dim 16 :dropout 0.0
                :activation :gelu :norm-first? true)
               specs)]
    (nn/train attention false)
    (nn/train layer false)
    (t/with-torch
      (let [query (t/sub (t/mul (t/reshape (t/arange 48) [2 3 8]) 0.01) 0.2)
            context (t/sub (t/mul (t/reshape (t/arange 80) [2 5 8]) 0.008) 0.15)
            padding-mask (t/tensor [[false false false false false]
                                    [false false false true true]]
                                   {:dtype :bool})
            rope-input (t/tensor [[[[1.0 2.0 3.0 4.0]]
                                   [[-1.0 0.5 2.0 -3.0]]]])
            angles (t/tensor [[0.2 0.2 0.6 0.6]
                              [0.4 0.4 0.8 0.8]])
            results
            {"self_attention" (nn/forward attention query)
             "cross_attention" (nn/forward attention
                                           {:query query
                                            :key context
                                            :value context
                                            :key-padding-mask padding-mask})
             "transformer" (nn/forward layer query)
             "interleaved_rope" (t/apply-rope
                                 rope-input
                                 (t/cos angles)
                                 (t/sin angles)
                                 :interleaved? true)}
            errors (into {}
                         (map (fn [[result-name actual]]
                                [result-name (max-error actual (get reference result-name))]))
                         results)
            tolerance 2.0e-5]
        (println (pr-str errors))
        (when (some #(>= (double %) tolerance) (vals errors))
          (throw (ex-info "Clorch transformer parity check failed"
                          {:errors errors :tolerance tolerance})))))))
