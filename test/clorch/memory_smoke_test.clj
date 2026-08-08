(ns clorch.memory-smoke-test
  (:require [clorch.torch :as t]
            [clojure.test :refer [deftest is testing run-tests]]
            [clojure.string :as str])
  (:import [org.bytedeco.javacpp Pointer]))

(defn tensor-alive? [x]
  (try
    (let [t-obj (t/get-tensor x)]
      (and (not (nil? t-obj))
           (not (.isNull ^Pointer t-obj))))
    (catch IllegalArgumentException _ false)))

(deftest deep-collection-release-test
  (testing "Manual release! on nested structures"
    (t/with-torch
      (let [data {:a (t/randn [2 2])
                  :b [(t/randn [2 2]) (t/randn [2 2])]
                  :c {:d (t/randn [2 2])}}
            tensors (filter #(try (instance? Pointer (t/get-tensor %)) (catch IllegalArgumentException _ false)) 
                            (tree-seq coll? identity data))]
        (is (every? tensor-alive? tensors))
        (t/release! data)
        (is (every? (complement tensor-alive?) tensors))))))

(deftest retain-cross-scope-test
  (testing "retain! prevents deallocation when scope exits"
    (let [x (t/with-torch
              (let [a (t/randn [2 2])]
                (t/retain! a)
                a))]
      (is (tensor-alive? x))
      (t/release! x)
      (is (not (tensor-alive? x))))))

(deftest session-idempotency-test
  (testing "Multiple start/stop calls don't crash"
    (is (nil? (t/stop-session!))) 
    (is (nil? (t/start-session!)))
    (is (nil? (t/start-session!)))
    (let [x (t/randn [2 2])]
      (is (tensor-alive? x))
      (t/stop-session!)
      (is (not (tensor-alive? x))))
    (is (nil? (t/stop-session!)))))

(deftest stress-gc-test
  (testing "Stress test: High frequency creation with gc!"
    (try
      (dotimes [i 1000]
        (t/randn [100 100])
        (when (zero? (mod i 250))
          (t/gc!)))
      (is true)
      (catch Throwable e
        (is false (str "Stress test failed: " (.getMessage e)))))))

(deftest thread-independence-test
  (testing "Sessions are thread-local"
    (t/start-session!)
    (let [x (t/randn [2 2])
          child-tensor-null? (atom nil)
          child-session-started? (promise)
          t (Thread. (fn []
                       (t/start-session!)
                       (let [y (t/randn [2 2])]
                         (deliver child-session-started? true)
                         (t/stop-session!)
                         (reset! child-tensor-null? (.isNull (t/get-tensor y))))))]
      (.start t)
      @child-session-started?
      (.join t)
      (is (true? @child-tensor-null?))
      (is (tensor-alive? x)) 
      (t/stop-session!)
      (is (not (tensor-alive? x))))))

(defn -main [& _args]
  (let [results (run-tests 'clorch.memory-smoke-test)]
    (System/exit (+ (:fail results) (:error results)))))
