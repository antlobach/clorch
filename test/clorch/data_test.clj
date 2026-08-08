(ns clorch.data-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clorch.data :as data]
            [clorch.data-process-test-support :as support]
            [clorch.torch :as t]))

(defn make-process-ds [n]
  (support/make-process-ds n))

(defn make-slow-process-ds [n sleep-ms]
  (support/make-slow-process-ds n sleep-ms))

(defn flatten-targets [batches]
  (vec (mapcat :target batches)))

(defn batch-n [batch]
  (t/size (:data batch) 0))

(deftest dataloader-batching-options-test
  (t/with-torch
    (let [x (t/reshape (t/arange 20) [10 2])
          y (t/arange 10)
          ds (data/tensor-dataset x y)]
      (testing "default partition-all behavior"
        (let [loader (data/dataloader ds :batch-size 3 :shuffle? false)
              batches (vec (seq loader))]
          (is (= 4 (count batches)))
          (is (= [3 3 3 1] (mapv batch-n batches)))))

      (testing "drop-last behavior"
        (let [loader (data/dataloader ds :batch-size 3 :shuffle? false :drop-last? true)
              batches (vec (seq loader))]
          (is (= 3 (count batches)))
          (is (= [3 3 3] (mapv batch-n batches)))))

      (testing "num-workers/prefetch-factor options are applied"
        (let [loader (data/dataloader ds :batch-size 3 :shuffle? false
                                      :num-workers 2 :prefetch-factor 2 :worker-backend :thread)
              batches (vec (seq loader))]
          (is (= 4 (count batches)))
          (is (= [3 3 3 1] (mapv batch-n batches)))
          (is (= 2 (:num-workers loader)))
          (is (= 2 (:prefetch-factor loader)))))

      (testing "custom collate-fn"
        (let [loader (data/dataloader ds :batch-size 4 :shuffle? false
                                      :collate-fn (fn [items] {:count (count items)}))
              batches (vec (seq loader))]
          (is (= [{:count 4} {:count 4} {:count 2}] batches)))))))

(deftest dataloader-process-workers-test
  (testing "num-workers defaults to process backend for process-capable dataset"
    (let [ds (make-process-ds 10)
          loader (data/dataloader ds :batch-size 4 :shuffle? false :num-workers 2)
          batches (vec (seq loader))]
      (is (= 3 (count batches)))
      (is (= [[0 0] [1 2] [2 4] [3 6]] (:data (first batches))))
      (is (= [0 1 2 3] (:target (first batches))))
      (is (= [8 9] (:target (last batches))))
      (is (= :process (:worker-backend loader)))))

  (testing "process workers reject non-process datasets"
    (t/with-torch
      (let [ds (data/tensor-dataset (t/reshape (t/arange 20) [10 2]) (t/arange 10))]
        (is (thrown-with-msg?
             IllegalArgumentException
             #"Process workers require dataset"
             (doall (seq (data/dataloader ds :batch-size 4 :shuffle? false :num-workers 2 :worker-backend :process))))))))

  (testing "process workers respect timeout-ms"
    (let [ds (make-slow-process-ds 8 75)
          loader (data/dataloader ds :batch-size 2 :shuffle? false :num-workers 2 :timeout-ms 20)]
      (is (thrown-with-msg?
           java.util.concurrent.TimeoutException
           #"timed out"
           (doall (seq loader)))))))

(deftest dataloader-worker-stress-test
  (testing "thread/process determinism and coverage across repeated epochs"
    (let [n 128
          ds (make-process-ds n)]
      (dotimes [_ 5]
        (let [thread-loader (data/dataloader ds :batch-size 16 :shuffle? false :num-workers 4 :worker-backend :thread)
              proc-loader (data/dataloader ds :batch-size 16 :shuffle? false :num-workers 4 :worker-backend :process)
              thread-batches (vec (seq thread-loader))
              proc-batches (vec (seq proc-loader))
              thread-targets (flatten-targets thread-batches)
              proc-targets (flatten-targets proc-batches)]
          (is (= 8 (count thread-batches)))
          (is (= 8 (count proc-batches)))
          (is (= (vec (range n)) thread-targets))
          (is (= (vec (range n)) proc-targets))
          (is (= thread-targets proc-targets))))))

  (testing "process backend shuffle still returns full set without duplicates"
    (let [n 200
          ds (make-process-ds n)
          loader (data/dataloader ds :batch-size 25 :shuffle? true :num-workers 4)
          batches (vec (seq loader))
          targets (flatten-targets batches)]
      (is (= n (count targets)))
      (is (= n (count (set targets))))
      (is (= (set (range n)) (set targets)))))

  (testing "process backend drop-last obeys truncation"
    (let [n 130
          ds (make-process-ds n)
          loader (data/dataloader ds :batch-size 16 :shuffle? false :drop-last? true :num-workers 3)
          batches (vec (seq loader))
          targets (flatten-targets batches)]
      (is (= 8 (count batches)))
      (is (= 128 (count targets)))
      (is (= (vec (range 128)) targets))))

  (testing "prefetch-factor does not alter data semantics"
    (let [n 96
          ds (make-process-ds n)
          l1 (data/dataloader ds :batch-size 12 :shuffle? false :num-workers 2 :prefetch-factor 1)
          l2 (data/dataloader ds :batch-size 12 :shuffle? false :num-workers 2 :prefetch-factor 8)]
      (is (= (flatten-targets (vec (seq l1)))
             (flatten-targets (vec (seq l2))))))))

(deftest process-parallel-batches-lazy-test
  (let [await-count (atom 0)
        stopped (atom 0)
        fake-ds (reify data/IProcessDataset
                  (process-spec [_] {:factory 'fake.ns/factory :args []}))]
    (with-redefs [data/write-edn-tempfile (fn [_ _] "/tmp/fake-spec.edn")
                  data/process-start-worker (fn [_ worker-id] {:id worker-id})
                  data/process-send! (fn [worker _idxs] worker)
                  data/process-await! (fn [worker _timeout-ms]
                                        (let [n (swap! await-count inc)]
                                          [worker [{:target n}]]))
                  data/process-stop-worker! (fn [worker]
                                              (swap! stopped inc)
                                              worker)]
      (let [batches (#'clorch.data/process-parallel-batches fake-ds
                                                            [[0] [1] [2]]
                                                            identity
                                                            1
                                                            1
                                                            nil
                                                            nil)]
        (is (= 0 @await-count))
        (is (= [{:target 1}] (first batches)))
        (is (= 1 @await-count))
        (is (= [[{:target 1}] [{:target 2}] [{:target 3}]] (doall batches)))
        (is (= 3 @await-count))
        (is (= 1 @stopped))))))

(deftest process-worker-failure-surfaces-immediately-test
  (let [stopped (atom 0)
        ds (data/dataset
            :size (fn [] 4)
            :get-item (fn [idx] {:data [idx] :target idx})
            :process-spec {:factory 'fake.ns/factory :args []})]
    (with-redefs [data/process-start-worker (fn [_ worker-id]
                                              (let [q (java.util.concurrent.LinkedBlockingQueue.)]
                                                (.put q {:ok false :worker-id worker-id :error "worker failed"})
                                                {:id worker-id :responses q :request-id 0}))
                  data/process-send! (fn [worker _idxs] worker)
                  data/process-stop-worker! (fn [worker]
                                              (swap! stopped inc)
                                              worker)
                  data/write-edn-tempfile (fn [_ _] "/tmp/fake-spec.edn")]
      (is (thrown? clojure.lang.ExceptionInfo
                   (doall (seq (data/dataloader ds :batch-size 2 :shuffle? false :num-workers 1 :worker-backend :process)))))
      (is (= 1 @stopped)))))

(deftest thread-worker-threads-timeout-after-partial-consumption-test
  (t/with-torch
    (let [x (t/reshape (t/arange 40) [20 2])
          y (t/arange 20)
          ds (data/tensor-dataset x y)
          loader (data/dataloader ds :batch-size 2 :shuffle? false :num-workers 2 :worker-backend :thread)]
      (is (some? (first (seq loader))))
      (Thread/sleep 2200)
      (let [workers (filter (fn [^Thread th]
                              (let [thread-name (.getName th)]
                                (and (.isAlive th)
                                     (str/starts-with? thread-name "clorch-data-thread-worker-"))))
                            (keys (Thread/getAllStackTraces)))]
        (is (empty? workers))))))

(deftest process-prefetch-factor-controls-inflight-capacity-test
  (let [send-count (atom 0)
        send-count-at-first-await (atom nil)
        item-id (atom 0)
        fake-ds (reify data/IProcessDataset
                  (process-spec [_] {:factory 'fake.ns/factory :args []}))]
    (with-redefs [data/write-edn-tempfile (fn [_ _] "/tmp/fake-spec.edn")
                  data/process-start-worker (fn [_ worker-id] {:id worker-id})
                  data/process-send! (fn [worker _idxs]
                                       (swap! send-count inc)
                                       worker)
                  data/process-await! (fn [worker _timeout-ms]
                                        (when (nil? @send-count-at-first-await)
                                          (reset! send-count-at-first-await @send-count))
                                        (let [n (swap! item-id inc)]
                                          [worker [{:target n}]]))
                  data/process-stop-worker! identity]
      (let [chunks (vec (repeat 10 [0]))]
        (doall (#'clorch.data/process-parallel-batches fake-ds chunks identity 2 3 nil nil))
        (is (= 6 @send-count-at-first-await))
        (is (= 10 @send-count))))))
