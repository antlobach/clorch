(ns nanochat
  "Modern Llama-3 style implementation of Karpathy's nanochat architecture in Clojure.
   Uses RMSNorm, RoPE, SwiGLU, and Grouped Query Attention (GQA)."
  (:require [clorch.torch :as torch]
            [clorch.cuda :as cuda]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.data :as data]
            [clorch.optim :as optim]
            [clorch.autograd :as autograd])
  (:import [com.knuddels.jtokkit Encodings]
           [com.knuddels.jtokkit.api EncodingType]))

(defn- env-int [k default]
  (if-let [v (System/getenv k)]
    (Integer/parseInt v)
    default))

;; --- Configuration (Llama-3-Tiny) ---
(def config
  {:vocab-size 50257
   :context-len 64
   :emb-dim     128
   :n-heads     4
   :n-kv-heads  1  ;; GQA: fewer heads for K/V
   :n-layers    2
   :drop-rate   0.1
   :batch-size  4
   :train-batches (env-int "CLORCH_NANOCHAT_TRAIN_BATCHES" 12)
   :sample-tokens (env-int "CLORCH_NANOCHAT_SAMPLE_TOKENS" 20)
   :chat-max-new-tokens (env-int "CLORCH_NANOCHAT_CHAT_MAX_NEW_TOKENS" 60)
   :device      (if (cuda/available?) :cuda :cpu)})

;; --- Data Loading ---
(defonce registry (Encodings/newDefaultEncodingRegistry))
(def tokenizer (.getEncoding registry EncodingType/R50K_BASE))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(data/defdataset TinyDataset [txt tokenizer max-length]
  [token-ids (vec (.toArray (.encode tokenizer txt)))
   n (count token-ids)
   inputs (mapv #(torch/tensor (subvec token-ids % (+ % max-length)) {:dtype :int64})
                (range 0 (- n max-length)))
   ml max-length]
  (get-item [idx] {:data (nth inputs idx)
                   :target (torch/tensor (subvec token-ids (inc idx) (+ idx ml 1)) {:dtype :int64})})
  (get-size [] (count inputs)))

;; --- Architecture ---

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel LlamaBlock [dim n-heads n-kv-heads context-len drop-rate]
  [sa #_{:clj-kondo/ignore [:unresolved-var]} (nn/GroupedQueryAttention dim n-heads n-kv-heads context-len drop-rate)
   ffwd #_{:clj-kondo/ignore [:unresolved-var]} (nn/SwiGLU dim (* 4 dim))
   ln1 (nn/rmsnorm dim)
   ln2 (nn/rmsnorm dim)]
  (forward [input]
           (let [{:keys [x mask freqs kv-cache]} (if (map? input) input {:x input})
                 sa-out (nn/forward sa {:x (nn/forward ln1 x) :mask mask :freqs freqs :kv-cache kv-cache})
                 x (torch/add x sa-out)
                 x (torch/add x (nn/forward ffwd (nn/forward ln2 x)))]
             x)))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel Llama [vocab-size emb-dim context-len n-layers n-heads n-kv-heads drop-rate]
  [tok-emb (nn/embedding vocab-size emb-dim :_freeze false)
   blocks (vec (repeatedly n-layers #(LlamaBlock emb-dim n-heads n-kv-heads context-len drop-rate)))
   ln-f (nn/rmsnorm emb-dim)
   output (nn/linear emb-dim vocab-size)
   head-dim (quot emb-dim n-heads)]
  (forward [input]
           (let [{:keys [idx mask freqs caches]} (if (map? input) input {:idx input})
                 [_ _T] (torch/size idx)
                 x (nn/forward tok-emb idx)
                 ;; Forward through blocks
                 x (loop [i 0 curr-x x]
                     (if (= i (count blocks))
                       curr-x
                       (recur (inc i)
                              (nn/forward (nth blocks i)
                                          {:x curr-x
                                           :mask mask
                                           :freqs freqs
                                           :kv-cache (when caches (nth caches i))}))))
                 x (nn/forward ln-f x)]
             (nn/forward output x))))

;; --- Utilities ---

(defn decode [ids]
  (let [int-list (com.knuddels.jtokkit.api.IntArrayList.)]
    (doseq [id ids] (.add int-list (int id)))
    (.decode tokenizer int-list)))

(defn encode [text]
  (let [token-ids (.encode tokenizer text)]
    (torch/tensor [(vec (.toArray token-ids))] {:dtype :int64})))

(defn fit-context-window [idx context-size]
  (let [[_ seq-len] (torch/size idx)
        seq-len (long seq-len)
        context-size (long context-size)]
    (if (<= seq-len context-size)
      idx
      (torch/ix idx :_ [(- seq-len context-size) seq-len]))))

(defn generate-stream [model idx max-new-tokens context-size & {:keys [use-cache?] :or {use-cache? true}}]
  (let [idx (fit-context-window idx context-size)
        head-dim (:head-dim model)
        all-freqs (torch/precompute-rope-freqs head-dim context-size)
        caches (when use-cache? (vec (repeatedly (count (:blocks model)) #(atom {}))))]
    (try
      (autograd/no-grad
       (loop [i 0
              curr-idx idx]
         (if (>= i max-new-tokens)
           curr-idx
           (let [old-caches (when caches (mapv deref caches))
                 next-idx
                 (torch/with-torch
                   (let [window-idx (fit-context-window curr-idx context-size)
                         [_ sz] (torch/size window-idx)
                         ;; In cache mode, only pass the last token after prefill.
                         step-idx (if (and use-cache? (> i 0))
                                    (torch/ix curr-idx :_ [-1])
                                    window-idx)
                         [_ step-sz] (torch/size step-idx)
                         start-pos (if (and use-cache? (> i 0)) (dec (long (min sz context-size))) 0)
                         freqs-step [(torch/ix (first all-freqs) [start-pos (+ start-pos step-sz)])
                                     (torch/ix (second all-freqs) [start-pos (+ start-pos step-sz)])]
                         mask (when (> (long step-sz) 1)
                                (let [m (torch/tril (torch/ones [step-sz step-sz]))]
                                  (torch/eq m 0)))
                         logits (nn/forward model {:idx step-idx :mask mask :freqs freqs-step :caches caches})
                         last-logits (torch/ix logits :_ -1 :_)
                         probs (F/softmax last-logits -1)
                         next-token (torch/multinomial probs 1)]
                     ;; Cache tensors escape through atoms, so retain them explicitly.
                     (when caches
                       (run! #(torch/retain! @%) caches))
                     (print (decode [(long (torch/item-float next-token))]))
                     (flush)
                     (torch/cat [curr-idx next-token] 1)))]
             ;; New cache tensors own concatenated storage; old internal caches can close now.
             (when old-caches
               (run! torch/release! old-caches))
             ;; Never release the caller-owned input; later accumulators are internal.
             (when (pos? i)
               (torch/release! curr-idx))
             (if (>= (second (torch/size next-idx)) context-size)
               next-idx
               (recur (inc i) next-idx))))))
      (finally
        (when caches
          (run! #(torch/release! @%) caches))
        (torch/release! all-freqs)))))

;; --- Main Operations ---

(defn train []
  (println "Loading 'The Verdict'...")
  (let [text (slurp "the-verdict.txt")
        dataset (TinyDataset text tokenizer (:context-len config))
        loader (data/dataloader dataset :batch-size (:batch-size config) :shuffle true)
        model (nn/to (Llama (:vocab-size config) (:emb-dim config) (:context-len config)
                            (:n-layers config) (:n-heads config) (:n-kv-heads config) (:drop-rate config))
                     (:device config))
        optimizer (optim/adam (nn/parameters model) :lr 3e-4)
        head-dim (:head-dim model)
        freqs (nn/to (torch/precompute-rope-freqs head-dim (:context-len config)) (:device config))
        mask (nn/to (torch/eq (torch/tril (torch/ones [(:context-len config) (:context-len config)])) 0)
                    (:device config))]

    ;; Keep model, optimizer, frequencies, and mask outside the per-batch scope.

    (println "Starting Modern NanoChat Training Loop (" (:train-batches config) " batches for verification)...")
    (nn/train model true)
    (let [total-loss (atom 0.0)
          cnt (atom 0)]
      (doseq [{:keys [data target]} (take (:train-batches config) loader)]
        (let [loss-value
              (torch/with-torch
                (optim/zero-grad optimizer)
                (let [logits (nn/forward model {:idx (nn/to data (:device config)) :mask mask :freqs freqs})
                      [B T V] (torch/size logits)
                      loss (F/cross-entropy (torch/reshape logits [(* B T) V])
                                            (torch/reshape (nn/to target (:device config)) [(* B T)]))]
                  (autograd/backward loss)
                  (nn/clip-grad-norm! (nn/parameters model) 1.0)
                  (optim/step optimizer)
                  (torch/item-float loss)))]
          (swap! total-loss + loss-value)
          (swap! cnt inc)
          (when (zero? (mod @cnt 5))
            (printf "Batch %d | Loss: %.4f\n" @cnt (/ @total-loss @cnt))))))

    (println "\nSaving model to llama_chat.pt...")
    (torch/save model "llama_chat.pt")

    (println "\nSample generation (with KV-Cache):")
    (nn/train model false)
    (let [start-ids (nn/to (encode "\n") (:device config))]
      (generate-stream model start-ids (:sample-tokens config) (:context-len config))
      (println))))

(defn chat []
  (let [model (nn/to (Llama (:vocab-size config) (:emb-dim config) (:context-len config)
                            (:n-layers config) (:n-heads config) (:n-kv-heads config) (:drop-rate config))
                     (:device config))
        ;; Llama-3 style chat history
        history (atom "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant trained on Edith Wharton's 'The Verdict'.<|eot_id|>")]
    (try
      (torch/load model "llama_chat.pt")
      (println "Model loaded from llama_chat.pt")
      (catch Exception _
        (println "No trained model found. Using untrained weights.")))

    (nn/train model false)
    (println "Welcome to Modern NanoChat (Llama-3 style)! (type 'exit' to quit)")
    (loop []
      (print "\nUser: ") (flush)
      (let [input (read-line)]
        (when (and input (not= input "exit"))
          (swap! history #(str % "<|start_header_id|>user<|end_header_id|>\n\n" input "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"))
          (print "GPT: ") (flush)
          (let [idx (-> @history
                        encode
                        (fit-context-window (:context-len config))
                        (nn/to (:device config)))
                out-idx (generate-stream model idx
                                         (:chat-max-new-tokens config)
                                         (:context-len config)
                                         :use-cache? false)

                [_ full-sz] (torch/size out-idx)
                [_ in-sz] (torch/size idx)
                new-tokens (torch/ix out-idx :_ [(long in-sz) (long full-sz)])
                flat-ids (mapv #(long (torch/item-float %)) (torch/tseq (torch/view new-tokens [-1])))
                response (decode flat-ids)]
            (swap! history #(str % response "<|eot_id|>"))
            (println)
            (recur)))))))
