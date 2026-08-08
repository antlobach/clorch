(ns clorch.platform-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.platform :as platform]))

(deftest should-use-gpu-logic
  (testing "force CPU wins"
    (with-redefs [clorch.platform/gpu-runtime-present? (constantly true)
                  clorch.platform/nvidia-hardware-present? (constantly true)
                  clorch.platform/env (fn [k] (case k
                                                "CLORCH_FORCE_CPU" "1"
                                                "CLORCH_FORCE_GPU" nil
                                                nil))
                  clorch.platform/prop (fn [k] (case k
                                                 "os.name" "Linux"
                                                 nil))]
      (is (false? (platform/should-use-gpu?)))))
  (testing "force GPU wins"
    (with-redefs [clorch.platform/gpu-runtime-present? (constantly false)
                  clorch.platform/nvidia-hardware-present? (constantly false)
                  clorch.platform/env (fn [k] (case k
                                                "CLORCH_FORCE_CPU" nil
                                                "CLORCH_FORCE_GPU" "1"
                                                nil))
                  clorch.platform/prop (fn [k] (case k
                                                 "os.name" "Linux"
                                                 nil))]
      (is (true? (platform/should-use-gpu?)))))
  (testing "macOS auto mode uses CPU"
    (with-redefs [clorch.platform/gpu-runtime-present? (constantly true)
                  clorch.platform/nvidia-hardware-present? (constantly true)
                  clorch.platform/env (constantly nil)
                  clorch.platform/prop (fn [k] (case k
                                                 "os.name" "Mac OS X"
                                                 nil))]
      (is (false? (platform/should-use-gpu?)))))
  (testing "Linux requires both runtime and hardware"
    (with-redefs [clorch.platform/env (constantly nil)
                  clorch.platform/prop (fn [k] (case k
                                                 "os.name" "Linux"
                                                 nil))]
      (with-redefs [clorch.platform/gpu-runtime-present? (constantly true)
                    clorch.platform/nvidia-hardware-present? (constantly false)]
        (is (false? (platform/should-use-gpu?))))
      (with-redefs [clorch.platform/gpu-runtime-present? (constantly false)
                    clorch.platform/nvidia-hardware-present? (constantly true)]
        (is (false? (platform/should-use-gpu?))))
      (with-redefs [clorch.platform/gpu-runtime-present? (constantly true)
                    clorch.platform/nvidia-hardware-present? (constantly true)]
        (is (true? (platform/should-use-gpu?)))))))
