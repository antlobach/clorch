(ns clorch.comparison-utils
  (:require [clorch.torch :as torch]
            [tech.v3.datatype.native-buffer :as native]
            [clojure.data.json :as json]))

(defn tensor->json [t]
  (let [t-obj (.contiguous (torch/->tensor t))
        shape (torch/size t-obj)
        numel (.numel t-obj)
        stype (.scalar_type t-obj)
        stype-str (.toString stype)
        kw (keyword stype-str)
        info (get torch/tech-dtype-info kw {:kw :float32 :bytes 4})
        address (.address (.data_ptr t-obj))
        buffer (native/wrap-address address (clojure.core/* numel (:bytes info)) (:kw info) :little-endian t-obj)
        data (mapv (fn [x]
                     (cond
                       (Double/isInfinite x) (if (pos? x) "Infinity" "-Infinity")
                       (Double/isNaN x) "NaN"
                       :else x))
                   buffer)]
    {:shape shape
     :dtype stype-str
     :data data}))

(defn export-result [result]
  (let [processed (clojure.walk/postwalk
                   (fn [node]
                     (if (instance? org.bytedeco.pytorch.Tensor node)
                       (tensor->json node)
                       node))
                   result)]
    (println (json/write-str processed))))
