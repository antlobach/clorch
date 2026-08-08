(ns bayesian-linear-regression-mcmc
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clorch.distributions :as dist]
            [clorch.nn :as nn]
            [clorch.torch :as t]))

(defn- env-int [k default]
  (if-let [v (System/getenv k)]
    (Integer/parseInt v)
    default))

;; Evaluate this file in the REPL top-to-bottom.
;; It builds a Bayesian linear regression model (3 features) and samples the
;; posterior with random-walk Metropolis-Hastings.

(t/with-torch
  ;; 1) Synthetic data
  (def n 120)
  (def true-w [2.5 -1.2 0.7])
  (def true-b -0.4)
  (def true-sigma 0.35)

  (def x0 (mapv #(- (* 4.0 (/ % (dec n))) 2.0) (range n)))
  (def x1 (mapv #(Math/sin (* 1.3 %)) x0))
  (def x2 (mapv #(Math/cos (* 0.7 %)) x0))
  (def xs (mapv vector x0 x1 x2))

  (def noise
    (mapv double (map t/item-float (t/tseq (dist/sample (dist/normal 0.0 true-sigma) [n])))))

  (def ys
    (mapv (fn [x-row e]
            (+ (reduce + (map * true-w x-row))
               true-b
               e))
          xs
          noise))

  (def y-cols (mapv vector ys))
  (def x-t (t/tensor xs))
  (def y-t (t/tensor y-cols))

  ;; 2) Log posterior
  ;; Priors:
  ;;   w_j ~ Normal(0, 5), j in 0..2
  ;;   b ~ Normal(0, 5)
  ;;   log_sigma ~ Normal(0, 1)
  ;; Likelihood:
  ;;   y_i ~ Normal(linear(x_i), exp(log_sigma))
  (defn set-model-params!
    [model {:keys [w b]}]
    (let [sd (nn/state-dict model)]
      (t/copy- (:weight sd) (t/tensor [(mapv double w)]))
      (t/copy- (:bias sd) (t/tensor [(double b)]))))

  (defn log-posterior
    [model {:keys [w b log-sigma]}]
    (let [sigma (Math/exp (double log-sigma))
          lp-prior (+ (reduce + (map (fn [wj]
                                       (t/item-float (dist/log-prob (dist/normal 0.0 5.0) wj)))
                                     w))
                      (t/item-float (dist/log-prob (dist/normal 0.0 5.0) b))
                      (t/item-float (dist/log-prob (dist/normal 0.0 1.0) log-sigma)))
          _ (set-model-params! model {:w w :b b})
          mu (nn/forward model x-t)
          lp-like (t/item-float (t/sum (dist/log-prob (dist/normal mu sigma) y-t)))]
      (+ lp-prior lp-like)))

  ;; 3) Proposal and MH step
  (def proposal-scale {:w 0.055 :b 0.05 :log-sigma 0.02})

  (defn gaussian-jitter [s]
    (* s (t/item-float (dist/sample (dist/normal 0.0 1.0)))))

  (defn propose [{:keys [w b log-sigma]}]
    {:w (mapv (fn [wj] (+ wj (gaussian-jitter (:w proposal-scale)))) w)
     :b (+ b (gaussian-jitter (:b proposal-scale)))
     :log-sigma (+ log-sigma (gaussian-jitter (:log-sigma proposal-scale)))})

  (defn mh-step [model {:keys [state logp accepted total]}]
    (t/with-torch
      (let [cand (propose state)
            cand-logp (log-posterior model cand)
            log-alpha (min 0.0 (- cand-logp logp))
            accept? (< (Math/log (t/item-float (dist/sample (dist/uniform 0.0 1.0)))) log-alpha)]
        (if accept?
          {:state cand :logp cand-logp :accepted (inc accepted) :total (inc total)}
          {:state state :logp logp :accepted accepted :total (inc total)}))))

  ;; 4) Run multiple chains in parallel (multicore)
  (def n-chains (env-int "CLORCH_BAYES_LR_CHAINS" 4))
  (def n-steps (env-int "CLORCH_BAYES_LR_STEPS" 2600))
  (def burn-in (env-int "CLORCH_BAYES_LR_BURN_IN" 700))
  (def thin (env-int "CLORCH_BAYES_LR_THIN" 2))
  ;; In this environment, in-process threaded tensor parameter updates can crash
  ;; libtorch (SIGSEGV). Keep this false unless running chains in isolated processes.
  (def parallel-chains? false)

  (defn make-init-state [chain-id]
    ;; Slightly over-dispersed inits help convergence diagnostics.
    {:w (vec (repeatedly 3 #(+ 0.0 (* 0.8 (t/item-float (dist/sample (dist/normal 0.0 1.0)))))))
     :b (+ 0.0 (* 0.8 (t/item-float (dist/sample (dist/normal 0.0 1.0)))))
     :log-sigma (+ (Math/log 1.0) (* 0.2 chain-id))})

  (defn run-chain [chain-id]
    (let [model (nn/linear 3 1)
          init-state (make-init-state chain-id)
          init {:state init-state :logp (log-posterior model init-state) :accepted 0 :total 0}
          chain (rest (take (inc n-steps) (iterate (partial mh-step model) init)))
          kept (->> chain
                    (drop burn-in)
                    (take-nth thin)
                    (map :state)
                    vec)
          {:keys [accepted total]} (last chain)]
      {:chain-id chain-id
       :kept kept
       :acceptance-rate (/ accepted (double total))}))

  ;; For parallel chains, each chain builds its own model instance in `run-chain`.
  ;; Never share a mutable model object across workers.
  (def chain-results
    (if parallel-chains?
      (->> (range n-chains)
           (mapv (fn [cid] (future (run-chain cid))))
           (mapv deref))
      (mapv run-chain (range n-chains))))

  (def kept-states
    (->> chain-results
         (mapcat (fn [{:keys [chain-id kept]}]
                   (map-indexed (fn [sample state]
                                  (assoc state :chain-id chain-id :sample sample))
                                kept)))
         vec))

  ;; 5) Posterior summaries (pooled over all chains)
  (defn mean-of [f]
    (/ (reduce + (map (comp double f) kept-states))
       (double (count kept-states))))

  (defn std-of [f]
    (let [m (mean-of f)
          sample-count (double (count kept-states))
          ss (reduce + (map (fn [v] (let [d (- (double (f v)) m)] (* d d))) kept-states))]
      (Math/sqrt (/ ss (max 1.0 (dec sample-count))))))

  (defn quantile-of [f q]
    (let [sorted-xs (sort (map (comp double f) kept-states))
          sample-count (count sorted-xs)
          idx (int (Math/floor (* q (dec sample-count))))]
      (nth sorted-xs (max 0 (min (dec sample-count) idx)))))

  (def posterior-w
    (mapv (fn [j]
            (mean-of (fn [s] (nth (:w s) j))))
          (range 3)))

  (def posterior-b (mean-of :b))
  (def posterior-sigma (mean-of (fn [s] (Math/exp (double (:log-sigma s))))))

  (def acceptance-rates (mapv :acceptance-rate chain-results))
  (def acceptance-rate (/ (reduce + acceptance-rates) (double n-chains)))

  (defn sample-variance [samples]
    (let [count-n (count samples)
          m (/ (reduce + samples) (double count-n))
          ss (reduce + (map (fn [x] (let [d (- x m)] (* d d))) samples))]
      (/ ss (max 1.0 (dec count-n)))))

  (defn rhat [f]
    (let [chains (mapv :kept chain-results)
          m (count chains)
          chain-len (count (first chains))
          chain-means (mapv (fn [ch] (/ (reduce + (map (comp double f) ch)) (double chain-len))) chains)
          chain-vars (mapv (fn [ch] (sample-variance (map (comp double f) ch))) chains)
          w (/ (reduce + chain-vars) (double m))
          b (* chain-len (sample-variance chain-means))
          var-hat (+ (* (/ (dec chain-len) (double chain-len)) w) (/ b (double chain-len)))]
      (Math/sqrt (/ var-hat w))))

  (def rhats
    {:w (mapv (fn [j]
                (rhat (fn [s] (nth (:w s) j))))
              (range 3))
     :b (rhat :b)
     :sigma (rhat (fn [s] (Math/exp (double (:log-sigma s)))))})

  ;; 6) Persist posterior artifacts for plotting/inspection
  (def out-dir (io/file "examples/out"))
  (.mkdirs out-dir)

  (def samples-csv (io/file out-dir "bayes_lr_posterior_samples.csv"))
  (with-open [w (io/writer samples-csv)]
    (csv/write-csv
     w
     (cons ["chain" "sample" "w0" "w1" "w2" "b" "sigma" "log_sigma"]
           (map (fn [{:keys [chain-id sample w b log-sigma]}]
                  [chain-id
                   sample
                   (double (nth w 0))
                   (double (nth w 1))
                   (double (nth w 2))
                   (double b)
                   (Math/exp (double log-sigma))
                   (double log-sigma)])
                kept-states))))

  (def summary-file (io/file out-dir "bayes_lr_posterior_summary.edn"))
  (spit summary-file
        (pr-str
         {:n-samples (count kept-states)
          :n-chains n-chains
          :acceptance-rate acceptance-rate
          :acceptance-rates acceptance-rates
          :rhat rhats
          :per-chain
          (mapv (fn [{:keys [chain-id kept] :as chain-result}]
                  (let [chain-acceptance-rate (:acceptance-rate chain-result)]
                    {:chain-id chain-id
                     :acceptance-rate chain-acceptance-rate
                     :n-samples (count kept)}))
                chain-results)
          :posterior
          {:w (mapv (fn [j]
                      {:mean (mean-of (fn [s] (nth (:w s) j)))
                       :sd (std-of (fn [s] (nth (:w s) j)))
                       :q05 (quantile-of (fn [s] (nth (:w s) j)) 0.05)
                       :q95 (quantile-of (fn [s] (nth (:w s) j)) 0.95)})
                    (range 3))
           :b {:mean posterior-b
               :sd (std-of :b)
               :q05 (quantile-of :b 0.05)
               :q95 (quantile-of :b 0.95)}
           :sigma {:mean posterior-sigma
                   :sd (std-of (fn [s] (Math/exp (double (:log-sigma s)))))
                   :q05 (quantile-of (fn [s] (Math/exp (double (:log-sigma s)))) 0.05)
                   :q95 (quantile-of (fn [s] (Math/exp (double (:log-sigma s)))) 0.95)}}}))

  (doseq [{:keys [chain-id] :as chain-result} chain-results]
    (println "Chain" chain-id "acceptance rate:" (:acceptance-rate chain-result)))
  (println "Posterior mean w:" posterior-w "(true" true-w ")")
  (println "Posterior mean b:" posterior-b "(true" true-b ")")
  (println "Posterior mean sigma:" posterior-sigma "(true" true-sigma ")")
  (println "R-hat:" rhats)
  (println "Mean acceptance rate:" acceptance-rate)
  (println "Saved posterior samples to" (.getPath samples-csv))
  (println "Saved posterior summary to" (.getPath summary-file)))
