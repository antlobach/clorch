(ns profiler
  (:require [clorch.torch :as torch]
            [clojure.string :as str]))

(import '[java.lang.management ManagementFactory]
        '[java.lang Runtime])

(defn get-rss []
  (let [pid (first (.split (.getName (ManagementFactory/getRuntimeMXBean)) "@"))
        runtime (Runtime/getRuntime)
        ps-out (.exec runtime (into-array String ["ps" "-o" "rss=" "-p" pid]))
        rss-bytes (with-open [in (.getInputStream ps-out)]
                    (.readAllBytes in))
        s (str/trim (String. rss-bytes))]
    (try
      (Long/parseLong s) ;; returns in KB
      (catch Exception _ 0))))

(defn get-heap []
  (let [runtime (Runtime/getRuntime)]
    (quot (- (.totalMemory runtime) (.freeMemory runtime)) 1024)))

(defn run-profiling [iterations print-every]
  (println "Iteration | Heap (KB) | RSS (KB)")
  (println "----------|-----------|---------")

  (dotimes [i iterations]
    (torch/with-torch
      (let [a (torch/randn [1000 1000])
            b (torch/randn [1000 1000])]
        (torch/matmul a b)
        (torch/sum (torch/mul a b))
        nil)) ;; Should free 4x 1000x1000 = ~16MB per iteration

    (when (or (zero? (mod (inc i) print-every)) (= (inc i) iterations))
      (let [heap (get-heap)
            rss (get-rss)]
        (printf "%9d | %9d | %9d\n" (inc i) heap rss)
        (flush))))

  (println "\nRunning System.gc() to check final heap...")
  (System/gc)
  (Thread/sleep 2000)
  (let [heap (get-heap)
        rss (get-rss)]
    (printf "FINAL     | %9d | %9d\n" heap rss)
    (flush)))

(defn -main [& _args]
  (run-profiling 500 50))
