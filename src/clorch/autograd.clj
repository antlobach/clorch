(ns clorch.autograd
  (:require [clorch.torch :as torch])
  (:import [org.bytedeco.pytorch NoGradGuard]))

(defn set-requires-grad [t bool]
  (.set_requires_grad (torch/->tensor t) (boolean bool))
  t)

(defn detach
  "Returns a new tensor detached from the current graph."
  [t]
  (.detach (torch/->tensor t)))

(defn backward [t]
  (.backward (torch/->tensor t))
  t)

(defn grad [t]
  (.grad (torch/->tensor t)))

(defmacro no-grad
  "Executes body with autograd disabled, mirroring torch.no_grad()."
  [& body]
  `(with-open [_# (NoGradGuard.)]
     ~@body))
