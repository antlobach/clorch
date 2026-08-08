(ns plot-bayesian-linear-regression-mcmc
  (:require [clojure.data.csv :as csv]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [scicloj.tableplot.v1.plotly :as p]
            [tablecloth.api :as tc]))

(def out-dir (io/file "examples/out"))
(def samples-file (io/file out-dir "bayes_lr_posterior_samples.csv"))
(def traces-html (io/file out-dir "bayes_lr_traces_tableplot.html"))
(def posteriors-html (io/file out-dir "bayes_lr_posteriors_tableplot.html"))

(def all-params
  [{:k :w0 :label "w0" :true 2.5}
   {:k :w1 :label "w1" :true -1.2}
   {:k :w2 :label "w2" :true 0.7}
   {:k :b :label "b" :true -0.4}
   {:k :sigma :label "sigma" :true 0.35}])

(defn read-samples []
  (when-not (.exists samples-file)
    (throw (ex-info "Missing samples CSV. Run bayesian_linear_regression_mcmc.clj first."
                    {:path (.getPath samples-file)})))
  (with-open [r (io/reader samples-file)]
    (let [[header & rows] (csv/read-csv r)
          ks (mapv keyword header)]
      (mapv (fn [row]
              (let [m (zipmap ks row)]
                {:chain (str (parse-long (:chain m)))
                 :sample (parse-long (:sample m))
                 :w0 (cond
                       (:w0 m) (parse-double (:w0 m))
                       (:w m) (parse-double (:w m))
                       :else nil)
                 :w1 (when (:w1 m) (parse-double (:w1 m)))
                 :w2 (when (:w2 m) (parse-double (:w2 m)))
                 :b (parse-double (:b m))
                 :sigma (parse-double (:sigma m))}))
            rows))))

(defn active-params [rows]
  (let [sample-row (first rows)]
    (->> all-params
         (filter (fn [{:keys [k]}] (some? (get sample-row k))))
         vec)))

(defn quantile [xs q]
  (let [s (vec (sort xs))
        idx (int (Math/floor (* q (dec (count s)))))]
    (nth s (max 0 (min (dec (count s)) idx)))))

(defn summary-stats [rows k]
  (let [xs (mapv k rows)
        mean (/ (reduce + xs) (double (count xs)))]
    {:mean mean
     :q05 (quantile xs 0.05)
     :q95 (quantile xs 0.95)}))

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

(defn trace-spec [ds rows {:keys [k label] :as param}]
  (let [true-v (:true param)
        {:keys [mean q05 q95]} (summary-stats rows k)
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

(defn histogram-spec [ds rows {:keys [k label] :as param}]
  (let [true-v (:true param)
        {:keys [mean q05 q95]} (summary-stats rows k)
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
       ".grid{display:grid;grid-template-columns:1fr;gap:16px;}@media(min-width:980px){.grid{grid-template-columns:1fr 1fr;}}</style>"
       "</head><body><h1>" title "</h1><div class=\"grid\">"
       (str/join "\n" (map-indexed (fn [i spec] (plot-div (str "plot-" i) spec)) specs))
       "</div></body></html>"))

(defn generate-plots! []
  (.mkdirs out-dir)
  (let [rows (read-samples)
        params (active-params rows)
        ds (tc/dataset rows)
        trace-specs (mapv (partial trace-spec ds rows) params)
        hist-specs (mapv (partial histogram-spec ds rows) params)]
    (spit traces-html (page-html "Bayesian Linear Regression MCMC Traces" trace-specs))
    (spit posteriors-html (page-html "Bayesian Linear Regression MCMC Posterior Histograms" hist-specs))
    (println "Saved:" (.getPath traces-html))
    (println "Saved:" (.getPath posteriors-html))))

(generate-plots!)
