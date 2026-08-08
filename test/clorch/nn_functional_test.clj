(ns clorch.nn-functional-test
  (:require [clorch.torch :as t]
            [clorch.nn.functional :as F]
            [clojure.test :refer [deftest is testing]]))

(defn near? [a b]
  (let [eps 1e-4]
    (< (Math/abs (- (float a) (float b))) eps)))

(defn tensor-near? [t1 t2]
  (let [v1 (mapv t/item-float (t/tseq (t/reshape t1 [-1])))
        v2 (mapv t/item-float (t/tseq (t/reshape t2 [-1])))]
    (every? (fn [[a b]] (near? a b)) (map vector v1 v2))))

(deftest basic-functional-test
  (t/with-torch
    (let [x (t/randn [1 10])]
      (testing "Linear"
        (let [w (t/randn [5 10])
              b (t/zeros [5])
              out (F/linear x w b)]
          (is (= [1 5] (t/size out)))))

      (testing "Softmax"
        (let [out (F/softmax (t/tensor [1.0 1.0]) 0)]
          (is (tensor-near? out (t/tensor [0.5 0.5])))))

      (testing "Dropout"
        (let [out (F/dropout x 0.0 :training? false)]
          (is (tensor-near? out x))))

      (testing "Interpolate"
        (let [input (t/randn [1 1 2 2])
              out (F/interpolate input :size [4 4] :mode :nearest)]
          (is (= [1 1 4 4] (t/size out)))))

      (testing "Pad"
        (let [input (t/randn [1 1 2 2])
              out (F/pad input [1 1 1 1] :mode :reflect)]
          (is (= [1 1 4 4] (t/size out))))))))

(deftest convolution-functional-test
  (t/with-torch
    (testing "Conv2d"
      (let [x (t/randn [1 1 10 10])
            w (t/randn [1 1 3 3])
            out (F/conv2d x w)]
        (is (= [1 1 8 8] (t/size out)))))))

(deftest pooling-functional-test
  (t/with-torch
    (let [x (t/randn [1 1 4 4])]
      (testing "MaxPool2d"
        (is (= [1 1 2 2] (t/size (F/max-pool2d x 2)))))
      (testing "AvgPool2d"
        (is (= [1 1 2 2] (t/size (F/avg-pool2d x 2)))))
      (testing "AdaptiveAvgPool2d"
        (is (= [1 1 2 2] (t/size (F/adaptive-avg-pool2d x 2))))))))

(deftest normalization-functional-test
  (t/with-torch
    (let [x (t/randn [1 10 10])]
      (testing "LayerNorm"
        (is (= [1 10 10] (t/size (F/layer-norm x 10))))))))

(deftest loss-functional-test
  (t/with-torch
    (let [input (t/tensor [0.5 0.5 0.5])
          target (t/tensor [1.0 0.0 1.0])]
      (testing "MSE Loss"
        (is (near? (t/item-float (F/mse-loss input target)) 0.25)))
      (testing "L1 Loss"
        (is (near? (t/item-float (F/l1-loss input target)) 0.5)))
      (testing "Smooth L1"
        (is (near? (t/item-float (F/smooth-l1-loss input target)) 0.125)))
      (testing "Cross Entropy"
        (let [logits (t/tensor [[10.0 0.0]])
              targets (t/tensor [0] {:dtype :int64})]
          (is (near? (t/item-float (F/cross-entropy logits targets)) 0.0))))
      (testing "BCE with Logits Loss"
        (is (near? (t/item-float (F/bce-with-logits-loss (t/tensor [0.0]) (t/tensor [1.0]))) 0.6931))))))

(deftest distance-functional-test
  (t/with-torch
    (let [x1 (t/tensor [1.0 0.0])
          x2 (t/tensor [0.0 1.0])]
      (testing "Cosine Similarity"
        (is (near? (t/item-float (F/cosine-similarity x1 x2 :dim 0)) 0.0)))
      (testing "Pairwise Distance"
        (is (near? (t/item-float (F/pairwise-distance x1 x2)) (Math/sqrt 2.0)))))))
