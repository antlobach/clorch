(ns clorch.nn-comprehensive-test
  (:require [clorch.torch :as torch]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clojure.test :refer [deftest is testing]]))

(defn tensor-shape [t] (torch/size t))

(deftest convolution-layers-test
  (torch/with-torch
    (testing "Conv1d"
      (let [m (nn/conv1d 16 33 3 :stride 2)
            input (torch/randn [20 16 50])]
        (is (= [20 33 24] (tensor-shape (nn/forward m input))))))

    (testing "Conv2d"
      (let [m (nn/conv2d 16 33 3 :stride 2)
            input (torch/randn [20 16 50 100])]
        (is (= [20 33 24 49] (tensor-shape (nn/forward m input))))))

    (testing "Conv3d"
      (let [m (nn/conv3d 16 33 3 :stride 2)
            input (torch/randn [20 16 10 50 100])]
        (is (= [20 33 4 24 49] (tensor-shape (nn/forward m input))))))

    (testing "ConvTranspose2d"
      (let [m (nn/conv-transpose2d 16 33 3 :stride 2)
            input (torch/randn [20 16 50 100])]
        (is (= [20 33 101 201] (tensor-shape (nn/forward m input))))))

    (testing "Conv2d Padding Modes"
      (let [m (nn/conv2d 1 1 3 :padding 1)
            input (torch/randn [1 1 10 10])]
        (is (= [1 1 10 10] (tensor-shape (nn/forward m input))))))))

(deftest normalization-layers-test
  (torch/with-torch
    (testing "BatchNorm1d"
      (let [m (nn/batchnorm1d 100)
            input (torch/randn [20 100])]
        (is (= [20 100] (tensor-shape (nn/forward m input))))))

    (testing "BatchNorm2d"
      (let [m (nn/batchnorm2d 16)
            input (torch/randn [20 16 50 100])]
        (is (= [20 16 50 100] (tensor-shape (nn/forward m input))))))

    (testing "BatchNorm3d"
      (let [m (nn/batchnorm3d 16)
            input (torch/randn [20 16 10 50 100])]
        (is (= [20 16 10 50 100] (tensor-shape (nn/forward m input))))))

    (testing "LayerNorm"
      (let [m (nn/layernorm [50 100])
            input (torch/randn [20 16 50 100])]
        (is (= [20 16 50 100] (tensor-shape (nn/forward m input))))))

    (testing "RMSNorm"
      (let [m (nn/rmsnorm 100)
            input (torch/randn [20 100])]
        (is (= [20 100] (tensor-shape (nn/forward m input))))))

    (testing "GroupNorm"
      (let [m (nn/groupnorm 3 6)
            input (torch/randn [20 6 10 10])]
        (is (= [20 6 10 10] (tensor-shape (nn/forward m input))))))

    (testing "InstanceNorm2d"
      (let [m (nn/instancenorm2d 100)
            input (torch/randn [20 100 35 45])]
        (is (= [20 100 35 45] (tensor-shape (nn/forward m input))))))

    (testing "InstanceNorm3d"
      (let [m (nn/instancenorm3d 10)
            input (torch/randn [2 10 4 4 4])]
        (is (= [2 10 4 4 4] (tensor-shape (nn/forward m input))))))))

(deftest recurrent-layers-test
  (torch/with-torch
    (let [input (torch/randn [5 3 10])]
      (testing "RNN"
        (let [m (nn/rnn 10 20 :num-layers 2)
              out (.get0 (nn/forward m input))]
          (is (= [5 3 20] (tensor-shape out)))))

      (testing "LSTM"
        (let [m (nn/lstm 10 20 :num-layers 2)
              out (.get0 (nn/forward m input))]
          (is (= [5 3 20] (tensor-shape out)))))

      (testing "GRU Bidirectional"
        (let [m (nn/gru 10 20 :bidirectional true)
              out (.get0 (nn/forward m input))]
          (is (= [5 3 40] (tensor-shape out))))))))

(deftest pooling-layers-test
  (torch/with-torch
    (let [input1d (torch/randn [20 16 50])
          input2d (torch/randn [20 16 50 32])
          input3d (torch/randn [20 16 10 50 32])]
      (testing "MaxPool1d"
        (is (= [20 16 25] (tensor-shape (nn/forward (nn/max-pool1d 2) input1d)))))
      (testing "MaxPool2d"
        (is (= [20 16 25 16] (tensor-shape (nn/forward (nn/max-pool2d 2) input2d)))))
      (testing "MaxPool3d"
        (is (= [20 16 5 25 16] (tensor-shape (nn/forward (nn/max-pool3d 2) input3d)))))

      (testing "AvgPool1d"
        (is (= [20 16 25] (tensor-shape (nn/forward (nn/avg-pool1d 2) input1d)))))
      (testing "AvgPool2d"
        (is (= [20 16 25 16] (tensor-shape (nn/forward (nn/avg-pool2d 2) input2d)))))
      (testing "AvgPool3d"
        (is (= [20 16 5 25 16] (tensor-shape (nn/forward (nn/avg-pool3d 2) input3d)))))

      (testing "AdaptiveAvgPool1d"
        (is (= [20 16 7] (tensor-shape (nn/forward (nn/adaptive-avg-pool1d 7) input1d)))))
      (testing "AdaptiveAvgPool2d"
        (is (= [20 16 7 7] (tensor-shape (nn/forward (nn/adaptive-avg-pool2d 7) input2d)))))
      (testing "AdaptiveAvgPool3d"
        (is (= [20 16 7 7 7] (tensor-shape (nn/forward (nn/adaptive-avg-pool3d 7) input3d))))))))

(deftest padding-layers-test
  (torch/with-torch
    (let [input (torch/randn [1 1 4 4])]
      (testing "ReflectionPad2d"
        (is (= [1 1 6 6] (tensor-shape (nn/forward (nn/reflection-pad2d 1) input)))))

      (testing "ZeroPad2d"
        (is (= [1 1 8 8] (tensor-shape (nn/forward (nn/zeropad2d 2) input)))))

      (testing "ConstantPad2d"
        (is (= [1 1 6 6] (tensor-shape (nn/forward (nn/constant-pad2d 1 3.5) input))))))))

(deftest utility-layers-test
  (torch/with-torch
    (testing "PixelShuffle"
      (let [input (torch/randn [1 4 4 4])]
        (is (= [1 1 8 8] (tensor-shape (nn/forward (nn/pixel-shuffle 2) input))))))

    (testing "Upsample"
      (let [input (torch/randn [1 1 2 2])]
        (is (= [1 1 4 4] (tensor-shape (nn/forward (nn/upsample :size [4 4] :mode :nearest) input))))))

    (testing "Bilinear"
      (let [m (nn/bilinear 20 30 40)
            input1 (torch/randn [128 20])
            input2 (torch/randn [128 30])]
        (is (= [128 40] (tensor-shape (nn/forward m [input1 input2]))))))))
