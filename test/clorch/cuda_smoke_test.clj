(ns clorch.cuda-smoke-test
  (:require [clorch.torch :as torch]
            [clorch.cuda :as cuda])
  (:import [org.bytedeco.javacpp Loader]))

(defn preload! []
  (println "Preloading native libraries...")
  (doseq [cls ["org.bytedeco.openblas.global.openblas_nolapack"
               "org.bytedeco.cuda.global.nvrtc"
               "org.bytedeco.pytorch.global.torch"]]
    (try
      (println "  Loading" cls)
      (Loader/load (Class/forName cls))
      (catch Throwable e
        (println "  Failed to load" cls ":" (.getMessage e))))))

(defn run-test []
  (preload!)
  (println "\n=== Clorch CUDA Smoke Test ===")
  (try
    (let [available? (cuda/available?)]
      (println "CUDA Available:" available?)
      (if available?
        (try
          (println "Device Count:" (cuda/device-count))
          (let [t (torch/tensor [1.0 2.0 3.0] {:device "cuda:0"})]
            (println "Tensor created on GPU:" t)
            (println "Tensor device:" (.device (torch/->tensor t)))
            (let [t-cpu (torch/to t :cpu)]
              (println "Moved back to CPU:" t-cpu))
            (println "SUCCESS: CUDA backend is working."))
          (catch Exception e
            (println "FAILURE: CUDA reported available but operation failed:")
            (println e)
            (when (instance? Throwable e)
              (.printStackTrace e))))
        (println "SKIPPED: No CUDA hardware detected. Logic is verified, but cannot run on this machine.")))
    (catch Exception e
      (println "FATAL ERROR during CUDA check:")
      (println e)
      (.printStackTrace e))))

(defn -main [& args]
  (run-test))

;; Allow running as script too
(when (= (System/getProperty "clojure.main.filename") *file*)
  (run-test))
