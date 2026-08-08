(ns clorch.dev
  (:require [clojure.string :as str]
            [clorch.platform :as platform]
            [nrepl.cmdline :as nrepl-cmdline])
  (:import [java.net BindException SocketException]
           [java.lang System]))

(defn- configure-javacpp-platform-extension! []
  (let [extension (platform/configure-platform-extension!)]
    (println "clorch.dev: org.bytedeco.javacpp.platform.extension=" (pr-str extension))))

(defn- env [k]
  (System/getenv k))

(defn- default-nrepl-args []
  (let [bind (or (env "CLORCH_NREPL_BIND") "127.0.0.1")
        port (or (env "CLORCH_NREPL_PORT") "7891")]
    ["--bind" bind "--port" port]))

(defn -main [& args]
  (configure-javacpp-platform-extension!)
  (let [run-args (if (seq args) (vec args) (default-nrepl-args))]
    (try
      (apply nrepl-cmdline/-main run-args)
      (catch BindException e
        (if (and (empty? args)
                 (str/includes? (.getMessage e) "Address already in use"))
          (do
            (println "clorch.dev: port in use, retrying with random port")
            (nrepl-cmdline/-main "--bind" "127.0.0.1" "--port" "0"))
          (throw e)))
      (catch SocketException e
        (println "clorch.dev: nREPL socket bind failed:" (.getMessage e))
        (throw e)))))
