(ns clorch.parity
  "PyTorch parity introspection utilities.
   Generates coverage reports between JavaCPP torch symbols and clorch.torch wrappers."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.set :as set]
            [clojure.string :as str])
  (:import [org.bytedeco.pytorch.global torch]))

(def ^:private noisy-prefixes
  #{"Get" "Set" "Log" "Init" "Is" "Has" "DeviceType" "LinalgBackend" "report" "show_" "dispatch_"})

(defn- op-like-name? [s]
  (and (re-matches #"[a-z][A-Za-z0-9_]*" s)
       (not (some #(str/starts-with? s %) noisy-prefixes))))

(defn- canonicalize-torch-op [s]
  (-> s
      (str/replace #"_outf?$" "")
      (str/replace #"_symint$" "")
      (str/replace #"_dimname$" "")
      (str/replace #"_names$" "")
      (str/replace #"_cpu$" "")
      (str/replace #"_cuda$" "")
      (str/replace #"_meta$" "")))

(defn- canonicalize-clorch-fn [s]
  (-> s
      (str/replace "-" "_")
      (str/replace #"\?$" "")
      (str/replace #"!$" "")))

(defn torch-op-candidates
  "Returns canonicalized JavaCPP torch operation names."
  []
  (->> (.getMethods torch)
       (map #(.getName ^java.lang.reflect.Method %))
       (filter op-like-name?)
       (map canonicalize-torch-op)
       (remove str/blank?)
       set))

(def ^:private surfaced-op-namespaces
  ['clorch.torch
   'clorch.linalg])

(defn- namespaced-op-name [ns-sym fn-sym]
  (let [namespace-name (name ns-sym)
        fn-name (name fn-sym)
        canonical-fn (canonicalize-clorch-fn fn-name)]
    (if (= namespace-name "clorch.linalg")
      (str "linalg_" canonical-fn)
      canonical-fn)))

(defn clorch-op-candidates
  "Returns canonicalized public function names across surfaced Clorch namespaces."
  []
  (doseq [ns-sym surfaced-op-namespaces]
    (require ns-sym))
  (->> surfaced-op-namespaces
       (mapcat (fn [ns-sym]
                 (->> (ns-publics ns-sym)
                      keys
                      (map #(namespaced-op-name ns-sym %)))))
       set))

(defn parity-report
  "Computes parity coverage report between JavaCPP torch and clorch.torch wrappers."
  []
  (let [torch-ops (torch-op-candidates)
        clorch-ops (clorch-op-candidates)
        covered (set/intersection torch-ops clorch-ops)
        missing (set/difference torch-ops clorch-ops)]
    {:torch-op-count (count torch-ops)
     :clorch-op-count (count clorch-ops)
     :covered-count (count covered)
     :coverage-ratio (if (pos? (count torch-ops))
                       (/ (double (count covered)) (double (count torch-ops)))
                       1.0)
     :covered (sort covered)
     :missing (sort missing)}))

(defn write-parity-report!
  "Writes parity report EDN to path and returns report map."
  [path]
  (let [report (parity-report)]
    (spit (io/file path) (with-out-str (binding [*print-length* nil *print-level* nil]
                                         (pr report))))
    report))

(defn read-parity-report
  "Reads parity report EDN from path."
  [path]
  (edn/read-string (slurp (io/file path))))
