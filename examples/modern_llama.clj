(ns modern-llama
  (:require [clorch.torch :as torch]
            [clorch.nn :as nn]))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel ModernLlamaBlock [dim n-heads n-kv-heads context-len drop-rate]
  [sa #_{:clj-kondo/ignore [:unresolved-var]} (nn/GroupedQueryAttention dim n-heads n-kv-heads context-len drop-rate)
   ffwd #_{:clj-kondo/ignore [:unresolved-var]} (nn/SwiGLU dim (* 4 dim))
   ln1 (nn/rmsnorm dim)
   ln2 (nn/rmsnorm dim)]
  (forward [input]
           (let [{:keys [x mask freqs kv-cache]} (if (map? input) input {:x input})
                 x (torch/add x (nn/forward sa {:x (nn/forward ln1 x) :mask mask :freqs freqs :kv-cache kv-cache}))
                 x (torch/add x (nn/forward ffwd (nn/forward ln2 x)))]
             x)))

(def dim 128)
(def n-heads 4)
(def n-kv-heads 2)
(def ctx-len 64)

(def block (ModernLlamaBlock dim n-heads n-kv-heads ctx-len 0.1))
(def x (torch/randn [2 16 dim]))
(def freqs (torch/precompute-rope-freqs (quot dim n-heads) 16))
(def mask (torch/eq (torch/tril (torch/ones [16 16])) 0))
(def out (nn/forward block {:x x :mask mask :freqs freqs}))
(nn/summary block {:x x :mask mask :freqs freqs})
(torch/size x)
(torch/size out)

(def cache-block (ModernLlamaBlock dim n-heads n-kv-heads ctx-len 0.0))
(def cache (atom {}))
(def x1 (torch/randn [1 10 dim]))
(def freqs1 (torch/precompute-rope-freqs (quot dim n-heads) 10))
(def out1 (nn/forward cache-block {:x x1 :freqs freqs1 :kv-cache cache}))
(def x2 (torch/randn [1 1 dim]))
(def all-freqs (torch/precompute-rope-freqs (quot dim n-heads) 11))
(def freqs2 [(torch/ix (first all-freqs) [10 11])
             (torch/ix (second all-freqs) [10 11])])
(def out2 (nn/forward cache-block {:x x2 :freqs freqs2 :kv-cache cache}))
(torch/size out1)
(torch/size out2)
(torch/size (:k-prev @cache))
(torch/size (:v-prev @cache))
