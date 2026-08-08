(ns clorch.optim
  (:import [org.bytedeco.pytorch SGD SGDOptions Adam AdamOptions AdamW AdamWOptions RMSprop RMSpropOptions Adagrad AdagradOptions TensorVector]))

(defn- ->tensor-vector [params]
  (if (instance? TensorVector params)
    params
    (let [v (TensorVector.)]
      (doseq [p params]
        (.push_back v p))
      v)))

(defn- put-betas! [betas-pointer [beta1 beta2]]
  (let [first-default (.get betas-pointer 0)
        values (cond
                 (= first-default 0.9) [beta1 beta2]
                 (= first-default 0.999) [beta2 beta1]
                 :else (throw (IllegalStateException.
                               (str "Unexpected native Adam beta layout: " first-default))))]
    (.put betas-pointer (double-array values))))

(defn sgd
  "Implements the SGD algorithm using the LibTorch C++ frontend."
  [params & {:keys [lr momentum dampening weight-decay nesterov]
             :or {lr 0.01
                  momentum 0
                  dampening 0
                  weight-decay 0
                  nesterov false}}]
  (let [opts (SGDOptions. (double lr))]
    (.put (.momentum opts) (double momentum))
    (.put (.dampening opts) (double dampening))
    (.put (.weight_decay opts) (double weight-decay))
    (.put (.nesterov opts) (boolean nesterov))
    (SGD. (->tensor-vector params) opts)))

(defn adam
  "Implements the Adam algorithm using the LibTorch C++ frontend."
  [params & {:keys [lr betas eps weight-decay amsgrad]
             :or {lr 0.001
                  betas [0.9 0.999]
                  eps 1e-8
                  weight-decay 0
                  amsgrad false}}]
  (let [opts (AdamOptions.)]
    (.put (.lr opts) (double lr))
    (put-betas! (.betas opts) betas)
    (.put (.eps opts) (double eps))
    (.put (.weight_decay opts) (double weight-decay))
    (.put (.amsgrad opts) (boolean amsgrad))
    (Adam. (->tensor-vector params) opts)))

(defn adamw
  "Implements the AdamW algorithm using the LibTorch C++ frontend."
  [params & {:keys [lr betas eps weight-decay amsgrad]
             :or {lr 0.001
                  betas [0.9 0.999]
                  eps 1e-8
                  weight-decay 0.01
                  amsgrad false}}]
  (let [opts (AdamWOptions.)]
    (.put (.lr opts) (double lr))
    (put-betas! (.betas opts) betas)
    (.put (.eps opts) (double eps))
    (.put (.weight_decay opts) (double weight-decay))
    (.put (.amsgrad opts) (boolean amsgrad))
    (AdamW. (->tensor-vector params) opts)))

(defn rmsprop
  "Implements the RMSprop algorithm using the LibTorch C++ frontend."
  [params & {:keys [lr alpha eps weight-decay momentum centered]
             :or {lr 0.01
                  alpha 0.99
                  eps 1e-8
                  weight-decay 0
                  momentum 0
                  centered false}}]
  (let [opts (RMSpropOptions.)]
    (.put (.lr opts) (double lr))
    (.put (.alpha opts) (double alpha))
    (.put (.eps opts) (double eps))
    (.put (.weight_decay opts) (double weight-decay))
    (.put (.momentum opts) (double momentum))
    (.put (.centered opts) (boolean centered))
    (RMSprop. (->tensor-vector params) opts)))

(defn adagrad
  "Implements the Adagrad algorithm using the LibTorch C++ frontend."
  [params & {:keys [lr lr-decay weight-decay initial-accumulator-value eps]
             :or {lr 0.01
                  lr-decay 0
                  weight-decay 0
                  initial-accumulator-value 0
                  eps 1e-10}}]
  (let [opts (AdagradOptions.)]
    (.put (.lr opts) (double lr))
    (.put (.lr_decay opts) (double lr-decay))
    (.put (.weight_decay opts) (double weight-decay))
    (.put (.initial_accumulator_value opts) (double initial-accumulator-value))
    (.put (.eps opts) (double eps))
    (Adagrad. (->tensor-vector params) opts)))

(defn zero-grad [opt]
  (.zero_grad opt))

(defn step [opt]
  (.step opt))
