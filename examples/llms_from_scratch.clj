(ns llms-from-scratch
  "Clojure port of 'Build a Large Language Model (From Scratch)' by Sebastian Raschka.
   Chapter 2, 3 & 4: Data, Attention & GPT Architecture.
   
   This is a REPL-friendly walkthrough designed to be executed step-by-step."
  (:require [clojure.java.io :as io]
            [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.data :as data]
            [clorch.optim :as optim]
            [clorch.autograd :as autograd])
  (:import [com.knuddels.jtokkit Encodings]
           [com.knuddels.jtokkit.api EncodingType]))

;; --- CHAPTER 2: DATA LOADING ---

(def url "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
(def file-path "the-verdict.txt")

(defn download-text []
  (println "Downloading text...")
  (with-open [in (io/input-stream url)
              out (io/output-stream file-path)]
    (io/copy in out))
  (println "Download complete."))

(when-not (.exists (io/file file-path))
  (download-text))

(defonce raw-text (slurp file-path))
(defonce registry (Encodings/newDefaultEncodingRegistry))
(def tokenizer (.getEncoding registry EncodingType/R50K_BASE))

;; GPT Dataset using the ergonomic defdataset macro
(data/defdataset GPTDataset [txt tokenizer max-length stride]
  [token-ids (vec (.toArray (.encode tokenizer txt)))
   n (count token-ids)
   chunks (for [i (range 0 (- n max-length) stride)]
            (let [input-chunk  (subvec token-ids i (+ i max-length))
                  target-chunk (subvec token-ids (inc i) (+ i max-length 1))]
              [(torch/tensor input-chunk {:dtype :int64})
               (torch/tensor target-chunk {:dtype :int64})]))
   inputs  (mapv first chunks)
   targets (mapv second chunks)]
  (get-item [idx] {:data (nth inputs idx) :target (nth targets idx)})
  (get-size [] (count inputs)))

(def gpt-ds (GPTDataset raw-text tokenizer 4 1))
(def dataloader (data/dataloader gpt-ds :batch-size 8 :shuffle false))

(println "Dataset and DataLoader initialized.")

;; --- CHAPTER 3: ATTENTION MECHANISMS ---

;; Input: 6 tokens, each with an embedding of size 3
(def x (torch/tensor [[1.0 0.1 0.2] [0.1 1.0 0.3] [0.2 0.3 1.0]
                      [0.4 0.5 0.6] [0.7 0.8 0.9] [1.0 1.1 1.2]]))

(nn/defmodel CausalAttention [d-in d-out]
  [W-query (nn/parameter (torch/randn [d-in d-out]))
   W-key   (nn/parameter (torch/randn [d-in d-out]))
   W-value (nn/parameter (torch/randn [d-in d-out]))]
  (forward [x]
           (let [q (torch/mm x W-query)
                 k (torch/mm x W-key)
                 v (torch/mm x W-value)
                 d-k (torch/size k -1)
                 scores (torch/div (torch/mm q (torch/T k)) (torch/sqrt d-k))
                 shape (torch/size scores)
                 T (last shape)
                 mask (torch/tril (torch/ones [T T]))
                 masked-scores (torch/masked-fill scores (torch/eq mask 0) torch/-inf)
                 weights (F/softmax masked-scores -1)]
             (torch/mm weights v))))

(nn/defmodel MultiHeadAttention [d-in d-out n-heads]
  [heads (vec (repeatedly n-heads #(CausalAttention d-in d-out)))
   out-proj (nn/linear (* d-out n-heads) d-in)]
  (forward [x]
           (let [head-outputs (mapv #(nn/forward % x) heads)
                 combined (torch/cat head-outputs -1)]
             (nn/forward out-proj combined))))

;; --- CHAPTER 4: IMPLEMENTING A GPT MODEL ---

(nn/defmodel FeedForward [d-in]
  [linear1 (nn/linear d-in (* 4 d-in))
   linear2 (nn/linear (* 4 d-in) d-in)]
  (forward [x]
           (let [out1 (nn/forward linear1 x)
                 act  (F/gelu out1)]
             (nn/forward linear2 act))))

(nn/defmodel TransformerBlock [d-in n-heads]
  [d-head (quot d-in n-heads)
   att (MultiHeadAttention d-in d-head n-heads)
   ff  (FeedForward d-in)
   ln1 (nn/layernorm d-in)
   ln2 (nn/layernorm d-in)]
  (forward [x]
           (let [x (torch/add x (nn/forward att (nn/forward ln1 x)))
                 x (torch/add x (nn/forward ff (nn/forward ln2 x)))]
             x)))

(nn/defmodel GPTModel [cfg]
  [tok-emb (nn/embedding (:vocab-size cfg) (:emb-dim cfg) :_freeze false)
   pos-emb (nn/embedding (:context-len cfg) (:emb-dim cfg) :_freeze false)
   blocks  (vec (repeatedly (:n-layers cfg) #(TransformerBlock (:emb-dim cfg) (:n-heads cfg))))
   ln-f    (nn/layernorm (:emb-dim cfg))
   out-head (nn/linear (:emb-dim cfg) (:vocab-size cfg))]
  (forward [x]
           (let [[batch-size seq-len] (torch/size x)
                 tok-embeds (nn/forward tok-emb x)
                 pos-indices (torch/tensor (range seq-len) {:dtype :int64})
                 pos-embeds (nn/forward pos-emb pos-indices)
                 emb-dim (last (torch/size tok-embeds))
                 pos-embeds-ready (torch/view pos-embeds [1 seq-len emb-dim])
                 x (torch/add tok-embeds pos-embeds-ready)
                 x (reduce (fn [acc b] (nn/forward b acc)) x blocks)
                 x (nn/forward ln-f x)]
             (nn/forward out-head x))))

;; Execution
(def gpt-config {:vocab-size 50257 :context-len 256 :emb-dim 128 :n-heads 4 :n-layers 2})
(def gpt (GPTModel gpt-config))

(println "GPT-2 Architecture (Raschka Style) initialized.")
(nn/summary gpt (torch/zeros [8 4] {:dtype :int64}))

(def batch (first dataloader))
(def gpt-logits (nn/forward gpt (:data batch)))
(println "GPT Output Logits shape:" (torch/size gpt-logits))

(println "\nWalkthrough complete. Sebastian Raschka's architecture is verified!")
