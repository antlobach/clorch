(ns clorch.release-check
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :as test]))

(def ^:private excluded-test-namespaces
  '#{clorch.memory-test-suite})

(defn- relative-path [^java.io.File base ^java.io.File file]
  (.toString (.relativize (.toPath base) (.toPath file))))

(defn- file->ns-sym [^java.io.File base ^java.io.File file]
  (-> (relative-path base file)
      (str/replace #"\.clj$" "")
      (str/replace #"_" "-")
      (str/replace #"/" ".")
      symbol))

(defn- test-file? [^java.io.File file]
  (and (.isFile file)
       (str/ends-with? (.getName file) "_test.clj")))

(defn discover-test-namespaces []
  (let [test-dir (io/file "test")]
    (->> (file-seq test-dir)
         (filter test-file?)
         (map #(file->ns-sym test-dir %))
         (remove excluded-test-namespaces)
         sort
         vec)))

(defn- require-test-namespaces! [ns-syms]
  (doseq [ns-sym ns-syms]
    (require ns-sym :reload)))

(defn run-clojure-tests! []
  (let [ns-syms (discover-test-namespaces)]
    (when-not (seq ns-syms)
      (throw (ex-info "No test namespaces discovered under test/." {})))
    (println "Running Clojure test namespaces:")
    (doseq [ns-sym ns-syms]
      (println " -" ns-sym))
    (require-test-namespaces! ns-syms)
    (let [{:keys [fail error]} (apply test/run-tests ns-syms)]
      (when (pos? (+ fail error))
        (throw (ex-info "Clojure tests failed."
                        {:fail fail
                         :error error})))
      (println "Clojure tests passed."))))

(defn- parse-mode [args]
  (let [[flag mode] args]
    (cond
      (empty? args) :cpu
      (= [flag mode] ["--mode" "cpu"]) :cpu
      (= [flag mode] ["--mode" "gpu"]) :gpu
      :else (throw (ex-info "Expected --mode cpu|gpu."
                            {:args args})))))

(defn- maybe-run-cuda-smoke! [mode]
  (when (= mode :gpu)
    (println "Running optional CUDA smoke test...")
    (require 'clorch.cuda-smoke-test :reload)
    ((resolve 'clorch.cuda-smoke-test/run-test))))

(defn -main [& args]
  (let [mode (parse-mode args)]
    (println "Release check mode:" (name mode))
    (run-clojure-tests!)
    (maybe-run-cuda-smoke! mode)))
