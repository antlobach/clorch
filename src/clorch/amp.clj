(ns clorch.amp
  "CUDA/CPU autocast and dynamic gradient scaling."
  (:require [clorch.torch :as t])
  (:import [org.bytedeco.pytorch Optimizer Scalar Tensor TensorVector]
           [org.bytedeco.pytorch.global torch torch$DeviceType torch$ScalarType]))

(defn- device-type [device]
  (case device
    :cuda torch$DeviceType/CUDA
    :cpu torch$DeviceType/CPU
    (throw (ex-info "Autocast device must be :cuda or :cpu" {:device device}))))

(defn- scalar-type [dtype]
  (case dtype
    :float16 torch$ScalarType/Half
    :bfloat16 torch$ScalarType/BFloat16
    (throw (ex-info "Autocast dtype must be :float16 or :bfloat16" {:dtype dtype}))))

(defn call-with-autocast
  "Runs `f` with thread-local autocast state, restoring prior state afterward."
  [{:keys [device dtype enabled? cache?]
    :or {device :cuda dtype :bfloat16 enabled? true cache? true}}
   f]
  (let [native-device-type (device-type device)
        dtype (scalar-type dtype)]
    (when (and enabled? (not (torch/is_autocast_available native-device-type)))
      (throw (ex-info "Autocast is unavailable for this device" {:device device})))
    (let [old-enabled (torch/is_autocast_enabled native-device-type)
          old-dtype (torch/get_autocast_dtype native-device-type)
          old-cache (torch/is_autocast_cache_enabled)]
      (try
        (torch/set_autocast_dtype native-device-type dtype)
        (torch/set_autocast_cache_enabled (boolean cache?))
        (torch/set_autocast_enabled native-device-type (boolean enabled?))
        (torch/increment_nesting)
        (f)
        (finally
          (when (zero? (torch/decrement_nesting))
            (torch/clear_cache))
          (torch/set_autocast_enabled native-device-type old-enabled)
          (torch/set_autocast_dtype native-device-type old-dtype)
          (torch/set_autocast_cache_enabled old-cache))))))

(defmacro autocast
  "Runs body under CUDA/CPU autocast. Defaults to CUDA bfloat16."
  [options & body]
  `(call-with-autocast ~options (fn [] ~@body)))

(defrecord GradScaler
           [scale growth-factor backoff-factor growth-interval growth-tracker enabled?])

(defn grad-scaler
  "Creates a dynamic FP16 loss scaler. Disable it for bfloat16 training."
  ([] (grad-scaler {}))
  ([{:keys [initial-scale growth-factor backoff-factor growth-interval enabled?]
     :or {initial-scale 65536.0
          growth-factor 2.0
          backoff-factor 0.5
          growth-interval 2000
          enabled? true}}]
   (when-not (pos? initial-scale)
     (throw (ex-info ":initial-scale must be positive" {:initial-scale initial-scale})))
   (when-not (> growth-factor 1.0)
     (throw (ex-info ":growth-factor must exceed 1" {:growth-factor growth-factor})))
   (when-not (< 0.0 backoff-factor 1.0)
     (throw (ex-info ":backoff-factor must be between 0 and 1"
                     {:backoff-factor backoff-factor})))
   (when-not (pos? growth-interval)
     (throw (ex-info ":growth-interval must be positive"
                     {:growth-interval growth-interval})))
   (->GradScaler (atom (double initial-scale))
                 (double growth-factor)
                 (double backoff-factor)
                 (long growth-interval)
                 (atom 0)
                 (boolean enabled?))))

(defn current-scale [scaler]
  @(:scale scaler))

(defn scaler-state [scaler]
  {:scale @(:scale scaler)
   :growth-tracker @(:growth-tracker scaler)})

(defn load-scaler-state! [scaler {:keys [scale growth-tracker]}]
  (when-not (and (number? scale) (pos? scale)
                 (integer? growth-tracker) (not (neg? growth-tracker)))
    (throw (ex-info "Invalid gradient scaler checkpoint state"
                    {:scale scale :growth-tracker growth-tracker})))
  (reset! (:scale scaler) (double scale))
  (reset! (:growth-tracker scaler) (long growth-tracker))
  scaler)

(defn scale-loss
  "Multiplies loss by current scale when scaling is enabled."
  [scaler loss]
  (if (:enabled? scaler)
    (with-open [factor (Scalar. (double @(:scale scaler)))]
      (.mul ^Tensor loss factor))
    loss))

(defn backward!
  "Scales loss and runs backward."
  [scaler loss]
  (.backward ^Tensor (scale-loss scaler loss))
  nil)

(defn- optimizer-gradients [^Optimizer optimizer]
  (let [^TensorVector parameters (.parameters optimizer)]
    (keep (fn [index]
            (let [parameter (.get parameters (long index))
                  gradient (.grad ^Tensor parameter)]
              (when (.defined gradient) gradient)))
          (range (.size parameters)))))

(defn- gradients-finite? [gradients]
  (every? (fn [^Tensor gradient]
            (t/with-torch
              (.item_bool (.all (.isfinite gradient)))))
          gradients))

(defn- globally-finite? [finite?]
  (if ((requiring-resolve 'clorch.distributed/initialized?))
    (t/with-torch
      (let [flag (t/tensor [(if finite? 0 1)] {:dtype :int32 :device :cuda})]
        ((requiring-resolve 'clorch.distributed/all-reduce!) flag {:op :max})
        (zero? (.item_int ^Tensor flag))))
    finite?))

(defn- unscale-gradients! [gradients scale]
  (with-open [divisor (Scalar. (double scale))]
    (doseq [^Tensor gradient gradients]
      (.dividePut gradient divisor))))

(defn- update-scale! [scaler finite?]
  (if finite?
    (let [steps (swap! (:growth-tracker scaler) inc)]
      (when (>= steps (:growth-interval scaler))
        (swap! (:scale scaler) * (:growth-factor scaler))
        (reset! (:growth-tracker scaler) 0)))
    (do
      (swap! (:scale scaler) * (:backoff-factor scaler))
      (reset! (:growth-tracker scaler) 0))))

(defn step!
  "Unscales gradients, synchronizes overflow state, conditionally steps optimizer,
  and updates the dynamic scale. Returns true when the optimizer stepped."
  [scaler ^Optimizer optimizer]
  (if-not (:enabled? scaler)
    (do (.step optimizer) true)
    (let [gradients (vec (optimizer-gradients optimizer))
          finite? (globally-finite? (gradients-finite? gradients))]
      (when finite?
        (unscale-gradients! gradients @(:scale scaler))
        (.step optimizer))
      (update-scale! scaler finite?)
      finite?)))
