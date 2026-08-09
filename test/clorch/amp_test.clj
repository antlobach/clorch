(ns clorch.amp-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.amp :as amp]
            [clorch.torch :as t])
  (:import [org.bytedeco.pytorch Tensor]
           [org.bytedeco.pytorch.global torch torch$DeviceType]))

(deftest cpu-autocast-state-test
  (let [before (torch/is_autocast_enabled torch$DeviceType/CPU)
        output (amp/autocast {:device :cpu :dtype :bfloat16}
                             (t/matmul (t/randn [8 8]) (t/randn [8 8])))]
    (is (= "BFloat16" (.toString (.scalar_type ^Tensor output))))
    (is (= before (torch/is_autocast_enabled torch$DeviceType/CPU)))))

(deftest autocast-restores-state-after-failure-test
  (let [before (torch/is_autocast_enabled torch$DeviceType/CPU)]
    (is (thrown-with-msg? Exception #"body failed"
                          (amp/autocast {:device :cpu :dtype :bfloat16}
                                        (throw (Exception. "body failed")))))
    (is (= before (torch/is_autocast_enabled torch$DeviceType/CPU)))))

(deftest gradient-scaler-state-test
  (let [scaler (amp/grad-scaler {:initial-scale 1024.0})]
    (is (= 1024.0 (amp/current-scale scaler)))
    (amp/load-scaler-state! scaler {:scale 64.0 :growth-tracker 7})
    (is (= {:scale 64.0 :growth-tracker 7}
           (amp/scaler-state scaler)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (amp/load-scaler-state! scaler
                                         {:scale -1.0 :growth-tracker 0})))))

(deftest gradient-scaler-validation-test
  (testing "invalid dynamic scaling factors fail at construction"
    (is (thrown? clojure.lang.ExceptionInfo
                 (amp/grad-scaler {:initial-scale 0.0})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (amp/grad-scaler {:growth-factor 1.0})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (amp/grad-scaler {:backoff-factor 1.0})))))
