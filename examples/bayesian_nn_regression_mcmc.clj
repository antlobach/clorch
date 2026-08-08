(ns bayesian-nn-regression-mcmc
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clorch.distributions :as dist]
            [clorch.nn :as nn]
            [clorch.torch :as t]))

(defn- env-int [k default]
  (if-let [v (System/getenv k)]
    (Integer/parseInt v)
    default))

;; REPL-friendly Bayesian NN regression tutorial:
;; 1) Build a small nn/sequential model
;; 2) Sample all network parameters with random-walk MH
;; 3) Save posterior samples for plotting

(t/with-torch
  (t/manual-seed 42)

  ;; 1) Data and model setup
  (def n 160)
  (def hidden 4)
  (def true-sigma 0.20)

  (defn build-model []
    (nn/sequential
     (nn/linear 3 hidden)
     (nn/tanh)
     (nn/linear hidden 1)))

  (def x0 (mapv #(- (* 4.0 (/ % (dec n))) 2.0) (range n)))
  (def x1 (mapv #(Math/sin (* 1.4 %)) x0))
  (def x2 (mapv #(Math/cos (* 0.8 %)) x0))
  (def xs (mapv vector x0 x1 x2))
  (def x-t (t/tensor xs))

  (defn parameter-specs [model]
    (let [ps (t/->vector (nn/parameters model))]
      (mapv (fn [p]
              {:tensor p
               :shape (t/size p)
               :n (.numel ^org.bytedeco.pytorch.Tensor p)})
            ps)))

  (defn total-params [specs]
    (reduce + (map :n specs)))

  (defn assign-theta!
    [specs theta]
    (loop [offset 0
           specs-left specs]
      (when-let [{:keys [tensor shape] :as spec} (first specs-left)]
        (let [param-count (:n spec)
              theta-segment (subvec theta offset (+ offset param-count))
              src (t/reshape (t/tensor theta-segment) shape)]
          (t/copy- tensor src)
          (recur (+ offset param-count) (rest specs-left))))))

  (def data-model (build-model))
  (def data-specs (parameter-specs data-model))
  (def d (total-params data-specs))

  (def true-theta
    (->> (t/tseq (dist/sample (dist/normal 0.0 0.9) [d]))
         (map t/item-float)
         (mapv double)))

  (assign-theta! data-specs true-theta)

  (def y-clean (nn/forward data-model x-t))
  (def y-noise (dist/sample (dist/normal 0.0 true-sigma) [n 1]))
  (def y-t (t/add y-clean y-noise))

  ;; 2) Posterior model
  ;; theta_i ~ Normal(0, 1.5)
  ;; log_sigma ~ Normal(log(0.2), 0.6)
  ;; y_i ~ Normal(f_theta(x_i), exp(log_sigma))
  (defn log-posterior
    [model specs {:keys [theta log-sigma]}]
    (let [sigma (Math/exp (double log-sigma))
          _ (assign-theta! specs theta)
          pred (nn/forward model x-t)
          lp-prior-theta (reduce + (map (fn [v]
                                          (t/item-float (dist/log-prob (dist/normal 0.0 1.5) v)))
                                        theta))
          lp-prior-sigma (t/item-float (dist/log-prob (dist/normal (Math/log 0.2) 0.6) log-sigma))
          lp-like (t/item-float (t/sum (dist/log-prob (dist/normal pred sigma) y-t)))]
      (+ lp-prior-theta lp-prior-sigma lp-like)))

  ;; 3) MH sampler
  (def proposal-scale {:theta 0.03 :log-sigma 0.02})

  (defn normal-draw [mu sd]
    (+ mu (* sd (t/item-float (dist/sample (dist/normal 0.0 1.0))))))

  (defn propose [{:keys [theta log-sigma]}]
    {:theta (mapv (fn [v] (normal-draw v (:theta proposal-scale))) theta)
     :log-sigma (normal-draw log-sigma (:log-sigma proposal-scale))})

  (defn mh-step [model specs {:keys [state logp accepted total]}]
    (t/with-torch
      (let [cand (propose state)
            cand-logp (log-posterior model specs cand)
            log-alpha (min 0.0 (- cand-logp logp))
            accept? (< (Math/log (t/item-float (dist/sample (dist/uniform 0.0 1.0)))) log-alpha)]
        (if accept?
          {:state cand :logp cand-logp :accepted (inc accepted) :total (inc total)}
          {:state state :logp logp :accepted accepted :total (inc total)}))))

  (def n-chains (env-int "CLORCH_BAYES_NN_CHAINS" 4))
  (def n-steps (env-int "CLORCH_BAYES_NN_STEPS" 1000))
  (def burn-in (env-int "CLORCH_BAYES_NN_BURN_IN" 200))
  (def thin (env-int "CLORCH_BAYES_NN_THIN" 2))
  ;; Keep false unless you validate this JVM/libtorch combo with in-process threads.
  (def parallel-chains? false)

  (defn init-state [chain-id]
    {:theta (->> (range d)
                 (map (fn [_] (normal-draw 0.0 0.35)))
                 vec)
     :log-sigma (+ (Math/log 0.4) (* 0.1 chain-id))})

  (defn run-chain [chain-id]
    (let [model (build-model)
          specs (parameter-specs model)
          s0 (init-state chain-id)
          init {:state s0 :logp (log-posterior model specs s0) :accepted 0 :total 0}
          chain (rest (take (inc n-steps) (iterate (partial mh-step model specs) init)))
          kept (->> chain (drop burn-in) (take-nth thin) (map :state) vec)
          {:keys [accepted total]} (last chain)]
      {:chain-id chain-id
       :kept kept
       :acceptance-rate (/ accepted (double total))}))

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

  ;; 4) Summaries
  (defn mean-of [f]
    (/ (reduce + (map (comp double f) kept-states))
       (double (count kept-states))))

  (defn quantile-of [f q]
    (let [sorted-xs (sort (map (comp double f) kept-states))
          sample-count (count sorted-xs)
          idx (int (Math/floor (* q (dec sample-count))))]
      (nth sorted-xs (max 0 (min (dec sample-count) idx)))))

  (defn sample-variance [samples]
    (let [count-n (count samples)
          m (/ (reduce + samples) (double count-n))
          ss (reduce + (map (fn [x] (let [delta (- x m)] (* delta delta))) samples))]
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

  ;; Plot only a readable subset of parameters.
  (def tracked-indices (vec (distinct (filter #(< % d) [0 1 2 3 8 12 16 20]))))

  (def acceptance-rates (mapv :acceptance-rate chain-results))
  (def acceptance-rate (/ (reduce + acceptance-rates) (double n-chains)))
  (def posterior-sigma (mean-of (fn [s] (Math/exp (double (:log-sigma s))))))

  ;; 5) Persist artifacts
  (def out-dir (io/file "examples/out"))
  (.mkdirs out-dir)

  (def samples-csv (io/file out-dir "bayes_nn_posterior_samples.csv"))
  (def theta-cols (mapv (fn [idx] (str "theta_" idx)) (range d)))

  (with-open [w (io/writer samples-csv)]
    (csv/write-csv
     w
     (cons (into ["chain" "sample" "sigma" "log_sigma"] theta-cols)
           (map (fn [{:keys [chain-id sample theta log-sigma]}]
                  (into [chain-id
                         sample
                         (Math/exp (double log-sigma))
                         (double log-sigma)]
                        (mapv double theta)))
                kept-states))))

  (def summary-file (io/file out-dir "bayes_nn_posterior_summary.edn"))
  (spit summary-file
        (pr-str
         {:n-samples (count kept-states)
          :n-chains n-chains
          :n-params d
          :acceptance-rate acceptance-rate
          :acceptance-rates acceptance-rates
          :rhat {:sigma (rhat (fn [s] (Math/exp (double (:log-sigma s)))))
                 :tracked-theta
                 (into {}
                       (map (fn [idx]
                              [(keyword (str "theta-" idx))
                               (rhat (fn [s] (nth (:theta s) idx)))])
                            tracked-indices))}
          :tracked
          (mapv (fn [idx]
                  {:name (str "theta_" idx)
                   :index idx
                   :true (nth true-theta idx)
                   :mean (mean-of (fn [s] (nth (:theta s) idx)))
                   :q05 (quantile-of (fn [s] (nth (:theta s) idx)) 0.05)
                   :q95 (quantile-of (fn [s] (nth (:theta s) idx)) 0.95)})
                tracked-indices)
          :sigma {:true true-sigma
                  :mean posterior-sigma
                  :q05 (quantile-of (fn [s] (Math/exp (double (:log-sigma s)))) 0.05)
                  :q95 (quantile-of (fn [s] (Math/exp (double (:log-sigma s)))) 0.95)}}))

  (println "Posterior mean sigma:" posterior-sigma "(true" true-sigma ")")
  (println "Mean acceptance rate:" acceptance-rate)
  (println "Saved posterior samples to" (.getPath samples-csv))
  (println "Saved posterior summary to" (.getPath summary-file)))
