(ns data-loading-tutorial
  (:require [clorch.torch :as torch]
            [clorch.data :as data]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

;; --- 1. The Dataset Semantic ---
;; In the tutorial, FaceLandmarksDataset reads a CSV and loads images.
;; In Clorch, we use the IDataset protocol.

(defn create-face-landmarks-dataset
  [csv-file _root-dir transform]
  (let [;; In a real scenario, we'd read the CSV file.
        ;; For this smoke test, we'll simulate the data based on the tutorial structure.
        landmarks-frame (with-open [reader (io/reader csv-file)]
                          (doall (csv/read-csv reader)))
        ;; Header is index 0, data starts at 1
        data-rows (rest landmarks-frame)]

    (data/dataset
     :size (fn [] (count data-rows))
     :get-item (fn [idx]
                 (let [row (nth data-rows idx)
                       img-name (first row)
                       ;; Extract landmarks (tutorial has 68 points, 2 coords each = 136 values)
                       landmarks (->> (rest row) (map #(Double/parseDouble %)) vec)
                       landmarks-tensor (torch/reshape (torch/tensor landmarks) [68 2])

                       ;; Simulate image loading (3 channels, 64x64 for smoke test)
                       image (torch/randn [3 64 64])

                       sample {:image image :landmarks landmarks-tensor :name img-name}]

                   (if transform
                     (transform sample)
                     sample))))))

;; --- 2. Transforms ---
;; PyTorch uses classes. Clojure uses pure functions.

(defn rescale [output-size]
  (fn [sample]
    (println "  [Transform] Rescaling to" output-size)
    ;; Tutorial logic: resize image and adjust landmarks accordingly
    ;; For smoke test, we just return the sample (placeholder for torch/interpolate)
    sample))

(defn random-crop [output-size]
  (fn [sample]
    (println "  [Transform] Random cropping to" output-size)
    ;; Tutorial logic: crop image and shift landmarks
    sample))

(defn to-tensor []
  (fn [sample]
    ;; Our data is already tensors, but in PyTorch this converts ndarrays.
    ;; Here we can use it to normalize or rearrange channels.
    sample))

;; --- 3. Composing Transforms ---
(defn compose [& fs]
  (apply comp (reverse fs)))

(def csv-path "temp_landmarks.csv")

(with-open [writer (io/writer csv-path)]
  (csv/write-csv writer
                 [(into ["image"] (map #(str "p" %) (range 136)))
                  (into ["img1.jpg"] (repeat 136 "10.0"))
                  (into ["img2.jpg"] (repeat 136 "20.0"))
                  (into ["img3.jpg"] (repeat 136 "30.0"))]))

(def transform
  (compose
   (rescale 256)
   (random-crop 224)
   (to-tensor)))

(def transformed-dataset
  (create-face-landmarks-dataset csv-path "." transform))

(def dl (data/dataloader transformed-dataset :batch-size 2 :shuffle false))
(data/get-size transformed-dataset)

(def sample0 (data/get-item transformed-dataset 0))
(keys sample0)
(torch/size (:image sample0))
(torch/size (:landmarks sample0))

(def batch0 (first dl))
(torch/size (:image batch0))
(torch/size (:landmarks batch0))

(doseq [[i batch] (map-indexed vector dl)]
  (printf "Batch %d | Images shape: %s | Landmarks shape: %s\n"
          i (torch/size (:image batch)) (torch/size (:landmarks batch))))

(io/delete-file csv-path)
