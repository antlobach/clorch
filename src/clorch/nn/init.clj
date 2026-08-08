(ns clorch.nn.init
  (:require [clorch.torch :as t])
  (:import [org.bytedeco.pytorch.global torch]
           [org.bytedeco.pytorch NoGradGuard Scalar kFanIn kFanOut kReLU kLeakyReLU kTanh kSigmoid kLinear FanModeType Nonlinearity]))

(defmacro ^:private no-grad [& body]
  `(with-open [_# (NoGradGuard.)]
     ~@body))

(defn- ->fan-mode [mode]
  (condp = mode
    :fan-in  (FanModeType. (kFanIn.))
    :fan-out (FanModeType. (kFanOut.))
    (FanModeType. (kFanIn.))))

(defn- ->non-linearity [non-linearity]
  (condp = non-linearity
    :relu       (Nonlinearity. (kReLU.))
    :leaky-relu (Nonlinearity. (kLeakyReLU.))
    :tanh       (Nonlinearity. (kTanh.))
    :sigmoid    (Nonlinearity. (kSigmoid.))
    :linear     (Nonlinearity. (kLinear.))
    (Nonlinearity. (kLeakyReLU.))))

(defn xavier-uniform! [tensor & [gain]]
  (no-grad
   (torch/xavier_uniform_ (t/->tensor tensor) (double (or gain 1.0)))))

(defn xavier-normal! [tensor & [gain]]
  (no-grad
   (torch/xavier_normal_ (t/->tensor tensor) (double (or gain 1.0)))))

(defn kaiming-uniform! [tensor & {:keys [a mode non-linearity]
                                 :or {a 0 mode :fan-in non-linearity :leaky-relu}}]
  (no-grad
   (torch/kaiming_uniform_ (t/->tensor tensor) (double a) (->fan-mode mode) (->non-linearity non-linearity))))

(defn kaiming-normal! [tensor & {:keys [a mode non-linearity]
                                :or {a 0 mode :fan-in non-linearity :leaky-relu}}]
  (no-grad
   (torch/kaiming_normal_ (t/->tensor tensor) (double a) (->fan-mode mode) (->non-linearity non-linearity))))

(defn normal! [tensor & [mean std]]
  (no-grad
   (torch/normal_ (t/->tensor tensor) (double (or mean 0.0)) (double (or std 1.0)))))

(defn uniform! [tensor & [from to]]
  (no-grad
   (torch/uniform_ (t/->tensor tensor) (double (or from 0.0)) (double (or to 1.0)))))

(defn constant! [tensor value]
  (no-grad
   (torch/constant_ (t/->tensor tensor) (Scalar. (double value)))))

(defn zeros! [tensor]
  (no-grad
   (torch/zeros_ (t/->tensor tensor))))

(defn ones! [tensor]
  (no-grad
   (torch/ones_ (t/->tensor tensor))))
