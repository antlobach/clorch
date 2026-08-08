(ns clorch.memory-test-suite
  (:require [clorch.torch :as t]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.autograd :as autograd]
            [clorch.optim :as optim]
            [clojure.string :as str])
  (:import [java.lang.management ManagementFactory]
           [java.lang Runtime]))

;; --- Monitoring Helpers ---

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

(defn report-memory [label]
  (let [heap (get-heap)
        rss (get-rss)]
    (printf "%-30s | Heap: %8d KB | RSS: %8d KB\n" label heap rss)
    (flush)))

;; --- Heavy Training Loop ---

(defn training-step! [model optimizer data target]
  (let [logits (nn/forward model data)
        loss (F/cross-entropy logits target)]
    (optim/zero-grad optimizer)
    (autograd/backward loss)
    (optim/step optimizer)
    loss))

(defn run-heavy-training [iterations & {:keys [use-session?] :or {use-session? false}}]
  (println (str "\n--- Starting Training Test (Use Session: " use-session? ") ---"))
  (report-memory "Before initialization")
  
  (let [d 100 hidden 512 out 10
        model (nn/sequential (nn/linear d hidden) (nn/relu) (nn/linear hidden out))
        optimizer (optim/sgd (nn/parameters model) :lr 0.01)
        data (t/randn [64 d])
        target (t/rand-int 0 out [64])]
    
    (when use-session? (t/start-session!))
    
    (report-memory "After initialization")
    
    (dotimes [i iterations]
      (training-step! model optimizer data target)
      (when (zero? (mod (inc i) (quot iterations 4)))
        (report-memory (str "Iteration " (inc i)))))
    
    (report-memory "After training loop")
    
    (if use-session?
      (do
        (t/stop-session!)
        (report-memory "After stop-session!"))
      (do
        (t/gc!)
        (report-memory "After t/gc! (Loose mode)")))
    
    (println "--- Finished Training Test ---")))

(defn -main [& _args]
  (println "Starting Memory Management Validation...")
  (println "========================================")
  
  ;; Scenario 1: Loose Mode (No Scopes)
  ;; We expect RSS to grow, and then hopefully drop after t/gc!
  (run-heavy-training 200 :use-session? false)
  
  (Thread/sleep 2000)
  
  ;; Scenario 2: Session Mode
  ;; We expect RSS to grow, and then drop INSTANTLY after stop-session!
  (run-heavy-training 200 :use-session? true)
  
  (println "\nValidation Complete."))
