(ns clorch.dev-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.dev :as dev]))

(deftest default-nrepl-args
  (testing "defaults"
    (with-redefs [dev/env (constantly nil)]
      (is (= ["--bind" "127.0.0.1" "--port" "7891"]
             (#'dev/default-nrepl-args)))))
  (testing "env overrides"
    (with-redefs [dev/env (fn [k] (case k
                                    "CLORCH_NREPL_BIND" "0.0.0.0"
                                    "CLORCH_NREPL_PORT" "7999"
                                    nil))]
      (is (= ["--bind" "0.0.0.0" "--port" "7999"]
             (#'dev/default-nrepl-args))))))
