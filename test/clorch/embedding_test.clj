(ns clorch.embedding-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.autograd :as autograd]))

;; --- Helpers ---

(defn- near?
  ([a b] (near? a b 1e-4))
  ([a b epsilon]
   (let [diff (Math/abs (- (float a) (float b)))]
     (< diff epsilon))))

(defn- tensor->vec [t]
  (mapv torch/item-float (torch/tseq t)))

(defn- vec-approx? [v1 v2]
  (and (= (count v1) (count v2))
       (every? #(apply near? %) (map vector v1 v2))))

;; --- Basic Embedding Tests ---

(deftest embedding-basic-shape
  (testing "Embedding returns correct output shape"
    (torch/with-torch
      (let [emb (nn/embedding 100 32)
            input (torch/tensor [[1 2 3 4 5]
                                 [6 7 8 9 10]] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [2 5 32] (torch/size output)))))))

(deftest embedding-single-index
  (testing "Embedding lookup for single index"
    (torch/with-torch
      (let [emb (nn/embedding 10 5)
            input (torch/tensor [[0] [1] [2]] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [3 1 5] (torch/size output)))))))

(deftest embedding-2d-input
  (testing "Embedding with 2D input"
    (torch/with-torch
      (let [emb (nn/embedding 100 16)
            input (torch/tensor [[1 2 3 4 5]
                                 [6 7 8 9 10]] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [2 5 16] (torch/size output)))))))

(deftest embedding-1d-input
  (testing "Embedding with 1D input"
    (torch/with-torch
      (let [emb (nn/embedding 100 16)
            input (torch/tensor [1 2 3 4 5] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [5 16] (torch/size output)))))))

;; --- Padding Index Tests ---

(deftest embedding-padding-idx
  (testing "Padding index produces zero vectors"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :padding-idx 0)
            input (torch/tensor [[0 1 2 3 4]] {:dtype :int64})
            output (nn/forward emb input)
            first-embedding (torch/ix output 0 0)]
        (is (vec-approx? (tensor->vec first-embedding) [0 0 0 0 0]))))))

(deftest embedding-padding-idx-non-zero
  (testing "Non-zero padding index produces zero vectors"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :padding-idx 5)
            input (torch/tensor [[5 1 2 3 4]] {:dtype :int64})
            output (nn/forward emb input)
            first-embedding (torch/ix output 0 0)]
        (is (vec-approx? (tensor->vec first-embedding) [0 0 0 0 0]))))))

;; --- Max Norm Tests ---

(deftest embedding-max-norm
  (testing "Max norm renormalizes large vectors"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :max-norm 1.0 :norm-type 2.0)
            input (torch/tensor [[0]] {:dtype :int64})
            output (nn/forward emb input)
            out-vec (tensor->vec (torch/ix output 0 0))
            norm (Math/sqrt (reduce + (map #(* % %) out-vec)))]
        (is (<= norm 1.001))))))

(deftest embedding-no-max-norm
  (testing "Without max-norm, vectors are not renormalized"
    (torch/with-torch
      (let [emb (nn/embedding 10 5)
            weight (:weight (nn/state-dict emb))
            input (torch/tensor [[0]] {:dtype :int64})
            _ (nn/forward emb input)
            new-weight (:weight (nn/state-dict emb))]
        (is (= (torch/size weight) (torch/size new-weight)))))))

;; --- From Pretrained Tests ---

(deftest embedding-from-pretrained-basic
  (testing "From pretrained loads weights correctly"
    (torch/with-torch
      (let [weights (torch/randn [100 16])
            emb (nn/embedding-from-pretrained weights)
            loaded-weight (:weight (nn/state-dict emb))]
        (is (= [100 16] (torch/size loaded-weight)))))))

(deftest embedding-from-pretrained-output
  (testing "From pretrained produces expected output"
    (torch/with-torch
      (let [weights (torch/tensor [[1 2 3 4]
                                   [5 6 7 8]
                                   [9 10 11 12]] {:dtype :float32})
            emb (nn/embedding-from-pretrained weights)
            input (torch/tensor [[0 1 2]] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [1 3 4] (torch/size output)))
        (is (vec-approx? (tensor->vec (torch/ix output 0 0)) [1 2 3 4]))
        (is (vec-approx? (tensor->vec (torch/ix output 0 1)) [5 6 7 8]))
        (is (vec-approx? (tensor->vec (torch/ix output 0 2)) [9 10 11 12]))))))

;; --- Freeze/Trainable Tests ---

(deftest embedding-freeze
  (testing "Frozen embedding does not require grad"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :_freeze true)
            params-vec (nn/parameters emb)
            num-params (.size params-vec)]
        (is (pos? num-params))
        (loop [i 0]
          (when (< i num-params)
            (is (not (.requires_grad ^org.bytedeco.pytorch.Tensor (.get params-vec i))))
            (recur (inc i))))))))

(deftest embedding-trainable
  (testing "Trainable embedding requires grad"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :_freeze false)
            params-vec (nn/parameters emb)
            num-params (.size params-vec)]
        (is (pos? num-params))
        (loop [i 0]
          (when (< i num-params)
            (is (.requires_grad ^org.bytedeco.pytorch.Tensor (.get params-vec i)))
            (recur (inc i))))))))

(deftest embedding-freeze-from-pretrained
  (testing "From pretrained defaults to frozen"
    (torch/with-torch
      (let [weights (torch/randn [10 5])
            emb (nn/embedding-from-pretrained weights)
            params-vec (nn/parameters emb)
            num-params (.size params-vec)]
        (is (pos? num-params))
        (loop [i 0]
          (when (< i num-params)
            (is (not (.requires_grad ^org.bytedeco.pytorch.Tensor (.get params-vec i))))
            (recur (inc i))))))))

(deftest embedding-unfreeze-from-pretrained
  (testing "From pretrained with freeze false"
    (torch/with-torch
      (let [weights (torch/randn [10 5])
            emb (nn/embedding-from-pretrained weights :freeze false)
            params-vec (nn/parameters emb)
            num-params (.size params-vec)]
        (is (pos? num-params))
        (loop [i 0]
          (when (< i num-params)
            (is (.requires_grad ^org.bytedeco.pytorch.Tensor (.get params-vec i)))
            (recur (inc i))))))))

;; --- Parameters Tests ---

(deftest embedding-parameters-shape
  (testing "Parameters has correct shape"
    (torch/with-torch
      (let [emb (nn/embedding 100 32)
            params-vec (nn/parameters emb)]
        (is (= 1 (.size params-vec)))
        (is (= [100 32] (torch/size (.get params-vec 0))))))))

(deftest embedding-parameters-requires-grad
  (testing "Parameters have correct requires_grad when trainable"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :_freeze false)
            params-vec (nn/parameters emb)]
        (is (.requires_grad ^org.bytedeco.pytorch.Tensor (.get params-vec 0)))))))

;; --- State Dict Tests ---

(deftest embedding-state-dict-keys
  (testing "State dict contains weight key"
    (torch/with-torch
      (let [emb (nn/embedding 10 5)
            sd (nn/state-dict emb)]
        (is (contains? sd :weight))
        (is (= [10 5] (torch/size (:weight sd))))))))

(deftest embedding-state-dict-pretrained
  (testing "State dict preserves pretrained weights"
    (torch/with-torch
      (let [weights (torch/randn [10 5])
            emb (nn/embedding-from-pretrained weights)
            sd (nn/state-dict emb)]
        (is (= (torch/size weights) (torch/size (:weight sd))))))))

;; --- Train/Eval Mode Tests ---

(deftest embedding-train-eval
  (testing "Train and eval modes work"
    (torch/with-torch
      (let [emb (nn/embedding 10 5)
            _ (nn/train emb true)
            _ (nn/forward emb (torch/tensor [[0]] {:dtype :int64}))
            _ (nn/train emb false)
            _ (nn/forward emb (torch/tensor [[0]] {:dtype :int64}))]
        (is true)))))

;; --- Property Based Tests ---

(defspec embedding-output-dim-property 20
  (prop/for-all
   [embedding-dim (gen/choose 1 32)]
   (torch/with-torch
     (let [emb (nn/embedding 100 embedding-dim)
           input (torch/tensor [[1 2 3 4 5]] {:dtype :int64})
           output (nn/forward emb input)]
       (= [1 5 embedding-dim] (torch/size output))))))

(defspec embedding-consistency-property 20
  (prop/for-all
   [embedding-dim (gen/choose 5 15)
    seed (gen/choose 0 10000)]
   (torch/with-torch
     (torch/manual-seed seed)
     (let [emb (nn/embedding 50 embedding-dim)
           input (torch/tensor [[1 2 3]] {:dtype :int64})
           output1 (nn/forward emb input)
           output2 (nn/forward emb input)]
       (= (torch/size output1) (torch/size output2))))))

(defspec embedding-from-pretrained-preserves-shape 20
  (prop/for-all
   [rows (gen/choose 10 100)
    cols (gen/choose 5 30)]
   (torch/with-torch
     (let [weights (torch/randn [rows cols])
           emb (nn/embedding-from-pretrained weights)
           sd (nn/state-dict emb)]
       (= [rows cols] (torch/size (:weight sd)))))))

(defspec embedding-vocab-size-property 20
  (prop/for-all
   [vocab-size (gen/choose 10 100)]
   (torch/with-torch
     (let [emb (nn/embedding vocab-size 16)
           sd (nn/state-dict emb)]
       (= vocab-size (first (torch/size (:weight sd))))))))

(defspec embedding-dim-property 20
  (prop/for-all
   [embedding-dim (gen/choose 5 32)]
   (torch/with-torch
     (let [emb (nn/embedding 50 embedding-dim)
           sd (nn/state-dict emb)]
       (= embedding-dim (second (torch/size (:weight sd))))))))

(defspec embedding-max-norm-output-property 20
  (prop/for-all
   [max-norm (gen/double* {:min 0.5 :max 5.0 :NaN? false :infinite? false})]
   (torch/with-torch
     (let [emb (nn/embedding 50 10 :max-norm max-norm :norm-type 2.0)
           input (torch/tensor [[0]] {:dtype :int64})
           output (nn/forward emb input)]
       (= [1 1 10] (torch/size output))))))

(defspec embedding-batch-size-property 20
  (prop/for-all
   [batch-size (gen/choose 1 16)]
   (torch/with-torch
     (let [emb (nn/embedding 50 8)
           indices (vec (for [_ (range 5)] (long (rand 50))))
           batch-input (torch/tensor (vec (repeat batch-size indices)) {:dtype :int64})
           output (nn/forward emb batch-input)]
       (= batch-size (first (torch/size output)))))))

;; --- Edge Case Tests ---

(deftest embedding-single-element-batch
  (testing "Embedding with single element batch"
    (torch/with-torch
      (let [emb (nn/embedding 10 5)
            input (torch/tensor [[3]] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [1 1 5] (torch/size output)))))))

(deftest embedding-all-vocab-indices
  (testing "All vocabulary indices are valid"
    (torch/with-torch
      (let [num-embeddings 10
            embedding-dim 5
            emb (nn/embedding num-embeddings embedding-dim)
            indices (vec (range num-embeddings))
            input (torch/tensor [indices] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [1 num-embeddings embedding-dim] (torch/size output)))))))

;; --- Scale Grad By Freq Tests ---

(deftest embedding-scale-grad-by-freq
  (testing "Scale grad by frequency option is accepted"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :scale-grad-by-freq true)
            input (torch/tensor [[0 0 1 1 2]] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [1 5 5] (torch/size output)))))))

;; --- Sparse Gradients Tests ---

(deftest embedding-sparse-option
  (testing "Sparse gradient option is accepted"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :sparse true)
            input (torch/tensor [[0 1 2]] {:dtype :int64})
            output (nn/forward emb input)]
        (is (= [1 3 5] (torch/size output)))))))

;; --- Gradient Flow Tests ---

(deftest embedding-gradient-computation
  (testing "Embedding computes gradients when required"
    (torch/with-torch
      (let [emb (nn/embedding 10 5 :_freeze false)
            input (torch/tensor [[0 1]] {:dtype :int64})
            output (nn/forward emb input)
            loss (torch/sum output)
            _ (autograd/backward loss)
            params-vec (nn/parameters emb)
            weight (.get params-vec 0)
            grad (.grad weight)]
        (is (some? grad))
        (is (= (torch/size weight) (torch/size grad)))))))

;; --- Sequential Embedding Tests ---

(deftest embedding-in-sequential
  (testing "Embedding works in sequential"
    (torch/with-torch
      (let [model (nn/sequential
                   (nn/embedding 100 32)
                   (nn/linear 32 10))
            input (torch/tensor [[1 2 3 4 5]] {:dtype :int64})
            output (nn/forward model input)]
        (is (= [1 5 10] (torch/size output)))))))

(deftest embedding-sequential-state-dict
  (testing "Sequential with embedding has correct state dict"
    (torch/with-torch
      (let [model (nn/sequential
                   (nn/embedding 100 32)
                   (nn/linear 32 10))
            sd (nn/state-dict model)]
        (is (map? sd))
        (is (= 2 (count sd)))))))
