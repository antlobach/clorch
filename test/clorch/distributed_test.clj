(ns clorch.distributed-test
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is]]
            [clorch.data :as data]))

(defn process-dataset [n]
  (data/dataset
   :size (constantly n)
   :get-item (fn [index]
               {:data (long index)
                :target (long (* index index))})
   :process-spec {:factory 'clorch.distributed-test/process-dataset
                  :args [n]}))

(deftest distributed-sampler-test
  (let [rank-zero (data/distributed-sampler
                   10 {:num-replicas 2 :rank 0 :seed 41})
        rank-one (data/distributed-sampler
                  10 {:num-replicas 2 :rank 1 :seed 41})
        epoch-zero-a (data/sample-indices rank-zero)
        epoch-zero-b (data/sample-indices rank-one)]
    (is (= 5 (count epoch-zero-a)))
    (is (= 5 (count epoch-zero-b)))
    (is (= (set (range 10)) (set (concat epoch-zero-a epoch-zero-b))))
    (is (empty? (set/intersection
                 (set epoch-zero-a) (set epoch-zero-b))))
    (data/set-epoch! rank-zero 1)
    (data/set-epoch! rank-one 1)
    (is (= (set (range 10))
           (set (concat (data/sample-indices rank-zero)
                        (data/sample-indices rank-one)))))
    (is (not= epoch-zero-a (data/sample-indices rank-zero)))))

(deftest sampler-checkpoint-test
  (let [sampler (data/distributed-sampler
                 12 {:num-replicas 3 :rank 1 :seed 99})]
    (data/set-epoch! sampler 8)
    (let [state (data/sampler-state sampler)]
      (data/set-epoch! sampler 0)
      (data/load-sampler-state! sampler state)
      (is (= 8 (:epoch sampler)))
      (is (= state (data/sampler-state sampler))))))

(deftest sampler-and-shuffle-conflict-test
  (let [sampler (data/distributed-sampler
                 8 {:num-replicas 1 :rank 0})]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"cannot also shuffle"
         (data/dataloader (process-dataset 8)
                          {:sampler sampler :shuffle? true})))))

(deftest process-dataloader-preserves-order-test
  (let [loader (data/dataloader
                (process-dataset 10)
                {:batch-size 3
                 :shuffle? false
                 :num-workers 2
                 :prefetch-factor 2
                 :worker-backend :process
                 :timeout-ms 30000})
        batches (vec loader)]
    (is (= [[0 1 2] [3 4 5] [6 7 8] [9]]
           (mapv :data batches)))
    (is (= [[0 1 4] [9 16 25] [36 49 64] [81]]
           (mapv :target batches)))))
