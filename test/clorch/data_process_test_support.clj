(ns clorch.data-process-test-support
  (:require [clorch.data :as data]))

(defn make-process-ds [n]
  (data/dataset
   :size (fn [] n)
   :get-item (fn [idx] {:data [idx (* 2 idx)] :target idx})
   :process-spec {:factory 'clorch.data-process-test-support/make-process-ds
                  :args [n]}))

(defn make-slow-process-ds [n sleep-ms]
  (data/dataset
   :size (fn [] n)
   :get-item (fn [idx]
               (Thread/sleep (long sleep-ms))
               {:data [idx] :target idx})
   :process-spec {:factory 'clorch.data-process-test-support/make-slow-process-ds
                  :args [n sleep-ms]}))
