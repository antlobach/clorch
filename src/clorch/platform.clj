(ns clorch.platform
  (:require [clojure.java.shell :as sh]
            [clojure.string :as str])
  (:import [org.bytedeco.javacpp Loader]))

(defn- env [k]
  (System/getenv k))

(defn- prop [k]
  (System/getProperty k))

(defn- gpu-runtime-present? []
  (let [platform (Loader/getPlatform)
        resource (str "org/bytedeco/pytorch/" platform "-gpu/")
        classloader (.getContextClassLoader (Thread/currentThread))]
    (boolean (.getResource classloader resource))))

(defn- nvidia-hardware-present? []
  (zero? (:exit (sh/sh "sh" "-lc" "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1"))))

(defn should-use-gpu? []
  (let [force-cpu? (= "1" (env "CLORCH_FORCE_CPU"))
        force-gpu? (= "1" (env "CLORCH_FORCE_GPU"))
        os-name (prop "os.name")
        supported-os? (or (str/starts-with? os-name "Linux")
                          (str/starts-with? os-name "Windows"))]
    (cond
      force-cpu? false
      force-gpu? true
      (not supported-os?) false
      :else (and (gpu-runtime-present?) (nvidia-hardware-present?)))))

(defn configure-platform-extension! []
  (let [extension (if (should-use-gpu?) "-gpu" "")]
    (System/setProperty "org.bytedeco.javacpp.platform.extension" extension)
    extension))

