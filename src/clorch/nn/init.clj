(ns clorch.nn.init
  (:require [clorch.torch :as t])
  (:import [org.bytedeco.pytorch.global torch]
           [org.bytedeco.pytorch NoGradGuard Scalar kFanIn kFanOut kReLU kLeakyReLU kTanh kSigmoid kLinear FanModeType Nonlinearity]))

(set! *warn-on-reflection* true)

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

(defn- normal-cdf [value]
  (t/with-torch
    (* 0.5
       (+ 1.0
          (double
           (t/item-float
            (t/erf (t/tensor (/ (double value) (Math/sqrt 2.0))))))))))

(defn trunc-normal!
  "Fills a tensor from N(mean, std²), redrawn outside [a, b].

  Uses the inverse-CDF algorithm from torch.nn.init.trunc_normal_."
  [tensor & {:keys [mean std a b]
             :or {mean 0.0 std 1.0 a -2.0 b 2.0}}]
  (when-not (and (pos? std) (< a b))
    (throw (IllegalArgumentException.
            "trunc-normal! requires a positive std and a < b")))
  (let [lower (normal-cdf (/ (- a mean) std))
        upper (normal-cdf (/ (- b mean) std))
        native (t/->tensor tensor)]
    (no-grad
     (torch/uniform_ native (- (* 2.0 lower) 1.0) (- (* 2.0 upper) 1.0))
     (t/copy- native
              (-> native
                  t/erfinv
                  (t/mul (* std (Math/sqrt 2.0)))
                  (t/add mean)
                  (t/clamp a b)))))
  tensor)

(defn ones! [tensor]
  (no-grad
   (torch/ones_ (t/->tensor tensor))))
