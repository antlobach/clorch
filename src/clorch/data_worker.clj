(ns clorch.data-worker
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clorch.data :as data]))

(defn- load-dataset [spec-file]
  (let [{:keys [factory args worker-init-fn]} (edn/read-string (slurp spec-file))
        ctor (requiring-resolve factory)
        ds (apply ctor args)]
    (when worker-init-fn
      ((requiring-resolve worker-init-fn)))
    ds))

(defn- serve! [spec-file]
  (let [ds (load-dataset spec-file)
        rdr (io/reader System/in)
        w (io/writer System/out)]
    (doseq [line (line-seq rdr)]
      (let [{:keys [id op idxs]} (edn/read-string line)]
        (if (= op :shutdown)
          (do
            (.write w (str (pr-str {:id id :ok true :stopped true}) "\n"))
            (.flush w))
          (let [resp (try
                       {:id id
                        :ok true
                        :items (mapv #(data/get-item ds %) idxs)}
                       (catch Throwable t
                         {:id id
                          :ok false
                          :error (.getMessage t)}))]
            (.write w (str (pr-str resp) "\n"))
            (.flush w)))))))

(defn- one-shot! [spec-file idxs-file]
  (let [ds (load-dataset spec-file)
        idxs (edn/read-string (slurp idxs-file))
        items (mapv #(data/get-item ds %) idxs)]
    (print (pr-str items))
    (flush)))

(defn -main
  "Worker process entrypoint for process-backed dataloader.
   Args:
   - server mode: --server <spec-file>
   - one-shot mode: <spec-file> <idxs-file>"
  [& args]
  (cond
    (and (= "--server" (first args)) (second args))
    (serve! (second args))

    (= 2 (count args))
    (one-shot! (first args) (second args))

    :else
    (throw (IllegalArgumentException.
            "Usage: clojure -M -m clorch.data-worker --server <spec-file> | <spec-file> <idxs-file>"))))
