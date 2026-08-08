(ns plot-bayesian-nn-regression-mcmc
  (:require [clojure.data.csv :as csv]
            [clojure.data.json :as json]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [scicloj.tableplot.v1.plotly :as p]
            [tablecloth.api :as tc]))

(def out-dir (io/file "examples/out"))
(def samples-file (io/file out-dir "bayes_nn_posterior_samples.csv"))
(def summary-file (io/file out-dir "bayes_nn_posterior_summary.edn"))
(def traces-html (io/file out-dir "bayes_nn_traces_tableplot.html"))
(def posteriors-html (io/file out-dir "bayes_nn_posteriors_tableplot.html"))

(defn parse-row [ks row]
  (let [m (zipmap ks row)]
    (into {:chain (str (parse-long (:chain m)))
           :sample (parse-long (:sample m))}
          (for [[k v] m
                :when (and (not= k :chain) (not= k :sample))]
            [k (parse-double v)]))))

(defn read-samples []
  (when-not (.exists samples-file)
    (throw (ex-info "Missing samples CSV. Run bayesian_nn_regression_mcmc.clj first."
                    {:path (.getPath samples-file)})))
  (with-open [r (io/reader samples-file)]
    (let [[header & rows] (csv/read-csv r)
          ks (mapv keyword header)]
      (mapv (partial parse-row ks) rows))))

(defn read-summary []
  (when-not (.exists summary-file)
    (throw (ex-info "Missing summary EDN. Run bayesian_nn_regression_mcmc.clj first."
                    {:path (.getPath summary-file)})))
  (edn/read-string (slurp summary-file)))

(defn tracked-params [summary]
  (let [tracked (:tracked summary)
        sigma (:sigma summary)]
    (vec
     (concat
      (map (fn [{:keys [mean q05 q95] :as tracked-entry}]
             (let [param-name (:name tracked-entry)]
               {:k (keyword param-name)
                :label param-name
                :true (:true tracked-entry)
                :mean mean
                :q05 q05
                :q95 q95}))
           tracked)
      [{:k :sigma
        :label "sigma"
        :true (:true sigma)
        :mean (:mean sigma)
        :q05 (:q05 sigma)
        :q95 (:q95 sigma)}]))))

(defn vline-shapes [true-v mean-v q05 q95]
  [{:type "line" :xref "x" :yref "paper" :x0 true-v :x1 true-v :y0 0 :y1 1
    :line {:color "#d62728" :width 2}}
   {:type "line" :xref "x" :yref "paper" :x0 mean-v :x1 mean-v :y0 0 :y1 1
    :line {:color "#2ca02c" :width 2}}
   {:type "line" :xref "x" :yref "paper" :x0 q05 :x1 q05 :y0 0 :y1 1
    :line {:color "#ff7f0e" :width 1.5 :dash "dash"}}
   {:type "line" :xref "x" :yref "paper" :x0 q95 :x1 q95 :y0 0 :y1 1
    :line {:color "#ff7f0e" :width 1.5 :dash "dash"}}])

(defn hline-shapes [true-v mean-v q05 q95]
  [{:type "line" :xref "paper" :yref "y" :x0 0 :x1 1 :y0 true-v :y1 true-v
    :line {:color "#d62728" :width 2}}
   {:type "line" :xref "paper" :yref "y" :x0 0 :x1 1 :y0 mean-v :y1 mean-v
    :line {:color "#2ca02c" :width 2}}
   {:type "line" :xref "paper" :yref "y" :x0 0 :x1 1 :y0 q05 :y1 q05
    :line {:color "#ff7f0e" :width 1.5 :dash "dash"}}
   {:type "line" :xref "paper" :yref "y" :x0 0 :x1 1 :y0 q95 :y1 q95
    :line {:color "#ff7f0e" :width 1.5 :dash "dash"}}])

(defn trace-spec [ds {:keys [k label mean q05 q95] :as param}]
  (let [true-v (:true param)
        base (p/plot (p/layer-line ds {:=x :sample :=y k :=color :chain}))]
    (-> base
        (assoc-in [:layout :title]
                  (format "Trace: %s (true=%.4f mean=%.4f 95%%=[%.4f, %.4f])"
                          label true-v mean q05 q95))
        (assoc-in [:layout :xaxis] {:title "sample"})
        (assoc-in [:layout :yaxis] {:title label})
        (assoc-in [:layout :legend] {:orientation "h"})
        (assoc-in [:layout :shapes] (hline-shapes true-v mean q05 q95))
        (assoc-in [:layout :template] "plotly_white"))))

(defn histogram-spec [ds {:keys [k label mean q05 q95] :as param}]
  (let [true-v (:true param)
        base (p/plot (p/layer-histogram ds {:=x k :=color :chain}))]
    (-> base
        (assoc-in [:layout :title]
                  (format "Posterior Histogram: %s (true=%.4f mean=%.4f 95%%=[%.4f, %.4f])"
                          label true-v mean q05 q95))
        (assoc-in [:layout :xaxis] {:title label})
        (assoc-in [:layout :yaxis] {:title "count"})
        (assoc-in [:layout :barmode] "overlay")
        (assoc-in [:layout :legend] {:orientation "h"})
        (assoc-in [:layout :shapes] (vline-shapes true-v mean q05 q95))
        (assoc-in [:layout :template] "plotly_white"))))

(defn plot-div [id spec]
  (str "<div id=\"" id "\" style=\"width:100%;height:460px;\"></div>\n"
       "<script>Plotly.newPlot('" id "', "
       (json/write-str (:data spec))
       ", "
       (json/write-str (:layout spec))
       ", {responsive:true});</script>\n"))

(defn page-html [title specs]
  (str "<!doctype html><html><head><meta charset=\"utf-8\"/>"
       "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
       "<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>"
       "<title>" title "</title>"
       "<style>body{font-family:ui-sans-serif,system-ui,sans-serif;margin:16px;}h1{margin:0 0 12px;}"
       ".grid{display:grid;grid-template-columns:1fr;gap:16px;}@media(min-width:1100px){.grid{grid-template-columns:1fr 1fr;}}</style>"
       "</head><body><h1>" title "</h1><div class=\"grid\">"
       (str/join "\n" (map-indexed (fn [i spec] (plot-div (str "plot-" i) spec)) specs))
       "</div></body></html>"))

(defn generate-plots! []
  (.mkdirs out-dir)
  (let [rows (read-samples)
        summary (read-summary)
        tracked (tracked-params summary)
        ds (tc/dataset rows)
        trace-specs (mapv (partial trace-spec ds) tracked)
        hist-specs (mapv (partial histogram-spec ds) tracked)]
    (spit traces-html (page-html "Bayesian NN MCMC Traces" trace-specs))
    (spit posteriors-html (page-html "Bayesian NN MCMC Posterior Histograms" hist-specs))
    (println "Saved:" (.getPath traces-html))
    (println "Saved:" (.getPath posteriors-html))))

(generate-plots!)
