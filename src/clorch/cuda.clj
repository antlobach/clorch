(ns clorch.cuda
  (:require [clorch.platform :as platform])
  (:import [org.bytedeco.pytorch.global torch]
           [org.bytedeco.javacpp Loader]))

;; Force loading with GPU extension if possible
(let [old-ext (System/getProperty "org.bytedeco.javacpp.platform.extension")]
  (try
    (platform/configure-platform-extension!)
    (System/setProperty "org.bytedeco.javacpp.platform.extension" "-gpu")
    (doseq [cls ["org.bytedeco.cuda.global.cudart"
                 "org.bytedeco.cuda.global.cublas"
                 "org.bytedeco.cuda.global.nvrtc"
                 "org.bytedeco.pytorch.global.torch"]]
      (try
        (Loader/load (Class/forName cls))
        (catch Throwable _)))
    (finally
      (if old-ext
        (System/setProperty "org.bytedeco.javacpp.platform.extension" old-ext)
        (System/clearProperty "org.bytedeco.javacpp.platform.extension")))))

(defn available?
  "Returns a boolean indicating if CUDA is currently available."
  []
  (torch/cuda_is_available))

(defn device-count
  "Returns the number of GPUs available."
  []
  (torch/cuda_device_count))

(defn synchronize
  "Waits for all kernels in all streams on a CUDA device to complete."
  ([] (torch/cuda_synchronize))
  ([device-index] (torch/cuda_synchronize (clojure.core/long device-index))))

(defn manual-seed
  "Sets the seed for generating random numbers on the current GPU."
  [seed]
  (torch/cuda_manual_seed (clojure.core/long seed)))

(defn manual-seed-all
  "Sets the seed for generating random numbers on all GPUs."
  [seed]
  (torch/cuda_manual_seed_all (clojure.core/long seed)))
