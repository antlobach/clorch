(ns clorch.distributed.worker
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clorch.cuda :as cuda]
            [clorch.distributed :as dist]))

(defn- ensure-java-25! []
  (let [feature (.feature (Runtime/version))]
    (when (< feature 25)
      (throw (ex-info "Distributed workers require Java 25 or newer"
                      {:java-feature feature})))))

(defn- resolve-entrypoint [entrypoint]
  (let [namespace-symbol (symbol (namespace entrypoint))]
    (require namespace-symbol)
    (or (ns-resolve namespace-symbol (symbol (name entrypoint)))
        (throw (ex-info "Distributed worker entrypoint does not exist"
                        {:main entrypoint})))))

(defn- worker-context [process-group args]
  {:rank (dist/rank process-group)
   :local-rank (dist/local-rank process-group)
   :world-size (dist/world-size process-group)
   :backend (dist/backend process-group)
   :process-group process-group
   :args args})

(defn -main [& [config-path]]
  (ensure-java-25!)
  (when-not config-path
    (throw (ex-info "Distributed worker requires a configuration path" {})))
  (let [{:keys [main args]} (edn/read-string (slurp (io/file config-path)))
        backend (keyword (or (System/getenv "CLORCH_DIST_BACKEND") "nccl"))
        local-rank (Long/parseLong (or (System/getenv "LOCAL_RANK") "0"))]
    (cuda/set-device! local-rank)
    (try
      (dist/with-process-group {:backend backend}
        (let [process-group (dist/current-process-group)
              entrypoint (resolve-entrypoint main)
              result (entrypoint (worker-context process-group args))]
          (println (pr-str {:rank (dist/rank process-group)
                            :status :completed
                            :result result}))))
      (finally
        (shutdown-agents)))))
