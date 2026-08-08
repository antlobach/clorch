(ns clorch.jit-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.torch :as torch]))

(def ^:private jit-fixture-path "llama_chat.pt")

(defn- fixture-present? []
  (.exists (java.io.File. jit-fixture-path)))

(deftest jit-unsupported-apis
  (testing "unsupported APIs fail explicitly"
    (is (thrown-with-msg? UnsupportedOperationException #"jit-trace"
                          (torch/jit-trace nil)))
    (is (thrown-with-msg? UnsupportedOperationException #"jit-script"
                          (torch/jit-script nil)))
    (is (thrown-with-msg? UnsupportedOperationException #"onnx-export"
                          (torch/onnx-export nil)))))

(deftest jit-load-save-roundtrip
  (if-not (fixture-present?)
    (is true "Skipped: llama_chat.pt fixture not present")
    (testing "load and save a TorchScript module"
      (let [m (torch/jit-load jit-fixture-path)
            out "/tmp/clorch-jit-roundtrip.pt"]
        (is (instance? org.bytedeco.pytorch.JitModule m))
        (is (= out (torch/jit-save m out)))
        (is (.exists (java.io.File. out)))
        (is (instance? org.bytedeco.pytorch.JitModule (torch/jit-load out)))))))

(deftest jit-forward-failure-context
  (if-not (fixture-present?)
    (is true "Skipped: llama_chat.pt fixture not present")
    (testing "jit-forward returns useful context when invocation fails"
      (let [m (torch/jit-load jit-fixture-path)
            x (torch/tensor [[1 2 3 4]] {:dtype :int64})]
        (try
          (torch/jit-forward m [x])
          (is true "Invocation succeeded for this fixture")
          (catch clojure.lang.ExceptionInfo e
            (let [d (ex-data e)]
              (is (string? (:forward-error d)))
              (is (string? (:apply-error d))))))))))

(deftest jit-save-type-check
  (testing "jit-save rejects non-JitModule values"
    (is (thrown-with-msg? IllegalArgumentException #"jit-save expects"
                          (torch/jit-save (torch/tensor [1.0]) "/tmp/invalid-jit.pt")))))

