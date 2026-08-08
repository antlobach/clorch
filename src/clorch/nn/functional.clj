(ns clorch.nn.functional
  (:require [clorch.torch :as t])
  (:import [org.bytedeco.pytorch.global torch]))

(defn relu [tensor]
  (torch/relu (t/->tensor tensor)))

(defn relu6 [tensor]
  (torch/relu6 (t/->tensor tensor)))

(defn leaky-relu [tensor & [negative-slope]]
  (torch/leaky_relu (t/->tensor tensor) (double (or negative-slope 0.01)) false))

(defn gelu [tensor]
  (torch/gelu (t/->tensor tensor)))

(defn sigmoid [tensor]
  (torch/sigmoid (t/->tensor tensor)))

(defn log-sigmoid [tensor]
  (torch/log_sigmoid (t/->tensor tensor)))

(defn silu [tensor]
  (torch/silu (t/->tensor tensor)))

(defn mish [tensor]
  (torch/mish (t/->tensor tensor)))

(defn tanh [tensor]
  (torch/tanh (t/->tensor tensor)))

(defn hardtanh [tensor & [min-val max-val]]
  (torch/hardtanh (t/->tensor tensor)
                  (org.bytedeco.pytorch.Scalar. (double (or min-val -1.0)))
                  (org.bytedeco.pytorch.Scalar. (double (or max-val 1.0)))))

(defn hardswish [tensor]
  (torch/hardswish (t/->tensor tensor)))

(defn hardsigmoid [tensor]
  (torch/hardsigmoid (t/->tensor tensor)))

(defn elu [tensor & [alpha]]
  (torch/elu (t/->tensor tensor) (double (or alpha 1.0)) false))

(defn selu [tensor]
  (torch/selu (t/->tensor tensor)))

(defn celu [tensor & [alpha]]
  (torch/celu (t/->tensor tensor) (double (or alpha 1.0)) false))

(defn softplus [tensor & [beta threshold]]
  (torch/softplus (t/->tensor tensor)
                  (org.bytedeco.pytorch.Scalar. (double (or beta 1.0)))
                  (org.bytedeco.pytorch.Scalar. (double (or threshold 20.0)))))

(defn softsign [tensor]
  (torch/softsign (t/->tensor tensor)))

(defn softmin [tensor dim]
  (torch/softmin (t/->tensor tensor) (long dim) (org.bytedeco.pytorch.ScalarTypeOptional.)))

(defn log-softmax [tensor dim-or-opts]
  (let [dim (if (map? dim-or-opts) (:dim dim-or-opts) dim-or-opts)]
    (torch/log_softmax (t/->tensor tensor) (long dim) (org.bytedeco.pytorch.ScalarTypeOptional.))))

(defn hardshrink [tensor & [lambd]]
  (torch/hardshrink (t/->tensor tensor) (double (or lambd 0.5))))

(defn softshrink [tensor & [lambd]]
  (torch/softshrink (t/->tensor tensor) (double (or lambd 0.5))))

(defn tanhshrink [tensor]
  (torch/tanhshrink (t/->tensor tensor)))

(defn threshold [tensor threshold-val value & {:keys [inplace] :or {inplace false}}]
  (torch/threshold (t/->tensor tensor) (double threshold-val) (double value) (boolean inplace)))

(defn glu [tensor & [dim]]
  (torch/glu (t/->tensor tensor) (long (or dim -1))))

(defn rrelu [tensor & {:keys [lower upper training inplace] :or {lower 0.125 upper 0.3333333333333333 training false inplace false}}]
  (torch/rrelu (t/->tensor tensor) (double lower) (double upper) (boolean training) (boolean inplace)))

(defn prelu [tensor weight]
  (torch/prelu (t/->tensor tensor) (t/->tensor weight)))

(defn softmax [tensor dim-or-opts]
  (let [dim (if (map? dim-or-opts) (:dim dim-or-opts) dim-or-opts)]
    (torch/softmax (t/->tensor tensor) (long dim) (org.bytedeco.pytorch.ScalarTypeOptional.))))

(defn dropout [tensor p & {:keys [training?] :or {training? true}}]
  (torch/dropout (t/->tensor tensor) (double p) (boolean training?)))

(defn dropout2d [tensor p & {:keys [training?] :or {training? true}}]
  (torch/dropout2d (t/->tensor tensor) (double p) (boolean training?) false))

(defn max-pool1d [tensor kernel & {:keys [stride padding dilation ceil-mode] :or {padding 0 dilation 1 ceil-mode false}}]
  (let [k (if (number? kernel) [kernel] kernel)
        s (if (number? stride) [stride] (or stride k))
        p (if (number? padding) [padding] padding)
        d (if (number? dilation) [dilation] dilation)]
    (torch/max_pool1d (t/->tensor tensor) (long-array k) (long-array s) (long-array p) (long-array d) (boolean ceil-mode))))

(defn max-pool2d [tensor kernel & {:keys [stride padding dilation ceil-mode] :or {padding 0 dilation 1 ceil-mode false}}]
  (let [k (if (number? kernel) [kernel kernel] kernel)
        s (if (number? stride) [stride stride] (or stride k))
        p (if (number? padding) [padding padding] padding)
        d (if (number? dilation) [dilation dilation] dilation)]
    (torch/max_pool2d (t/->tensor tensor) (long-array k) (long-array s) (long-array p) (long-array d) (boolean ceil-mode))))

(defn max-pool3d [tensor kernel & {:keys [stride padding dilation ceil-mode] :or {padding 0 dilation 1 ceil-mode false}}]
  (let [k (if (number? kernel) [kernel kernel kernel] kernel)
        s (if (number? stride) [stride stride stride] (or stride k))
        p (if (number? padding) [padding padding padding] padding)
        d (if (number? dilation) [dilation dilation dilation] dilation)]
    (torch/max_pool3d (t/->tensor tensor) (long-array k) (long-array s) (long-array p) (long-array d) (boolean ceil-mode))))

(defn avg-pool1d [tensor kernel & {:keys [stride padding count-include-pad ceil-mode] :or {padding 0 count-include-pad true ceil-mode false}}]
  (let [k (if (number? kernel) [kernel] kernel)
        s (if (number? stride) [stride] (or stride k))
        p (if (number? padding) [padding] padding)]
    (torch/avg_pool1d (t/->tensor tensor) (long-array k) (long-array s) (long-array p) (boolean ceil-mode) (boolean count-include-pad))))

(defn avg-pool2d [tensor kernel & {:keys [stride padding count-include-pad ceil-mode divisor-override] :or {padding 0 count-include-pad true ceil-mode false}}]
  (let [k (if (number? kernel) [kernel kernel] kernel)
        s (if (number? stride) [stride stride] (or stride k))
        p (if (number? padding) [padding padding] padding)
        k-arr (long-array k)
        s-arr (long-array s)
        p-arr (long-array p)]
    (torch/avg_pool2d (t/->tensor tensor) k-arr s-arr p-arr (boolean ceil-mode) (boolean count-include-pad) (if divisor-override (org.bytedeco.pytorch.LongOptional. (clojure.core/long divisor-override)) (org.bytedeco.pytorch.LongOptional.)))))

(defn avg-pool3d [tensor kernel & {:keys [stride padding count-include-pad ceil-mode divisor-override] :or {padding 0 count-include-pad true ceil-mode false}}]
  (let [k (if (number? kernel) [kernel kernel kernel] kernel)
        s (if (number? stride) [stride stride stride] (or stride k))
        p (if (number? padding) [padding padding padding] padding)]
    (torch/avg_pool3d (t/->tensor tensor) (long-array k) (long-array s) (long-array p) (boolean ceil-mode) (boolean count-include-pad) (if divisor-override (org.bytedeco.pytorch.LongOptional. (clojure.core/long divisor-override)) (org.bytedeco.pytorch.LongOptional.)))))

(defn adaptive-max-pool1d [tensor output-size & {:keys [return-indices?] :or {return-indices? false}}]
  (let [res (torch/adaptive_max_pool1d (t/->tensor tensor) (long-array (if (number? output-size) [output-size] output-size)))]
    (if return-indices?
      [(.get0 res) (.get1 res)]
      (.get0 res))))

(defn adaptive-max-pool2d [tensor output-size & {:keys [return-indices?] :or {return-indices? false}}]
  (let [res (torch/adaptive_max_pool2d (t/->tensor tensor) (long-array (if (number? output-size) [output-size output-size] output-size)))]
    (if return-indices?
      [(.get0 res) (.get1 res)]
      (.get0 res))))

(defn adaptive-max-pool3d [tensor output-size & {:keys [return-indices?] :or {return-indices? false}}]
  (let [res (torch/adaptive_max_pool3d (t/->tensor tensor) (long-array (if (number? output-size) [output-size output-size output-size] output-size)))]
    (if return-indices?
      [(.get0 res) (.get1 res)]
      (.get0 res))))

(defn adaptive-avg-pool1d [tensor output-size]
  (torch/adaptive_avg_pool1d (t/->tensor tensor) (long-array (if (number? output-size) [output-size] output-size))))

(defn adaptive-avg-pool2d [tensor output-size]
  (torch/adaptive_avg_pool2d (t/->tensor tensor) (long-array (if (number? output-size) [output-size output-size] output-size))))

(defn adaptive-avg-pool3d [tensor output-size]
  (torch/adaptive_avg_pool3d (t/->tensor tensor) (long-array (if (number? output-size) [output-size output-size output-size] output-size))))

(defn- ->optional [t]
  (if t
    (org.bytedeco.pytorch.TensorOptional. (t/->tensor t))
    (org.bytedeco.pytorch.TensorOptional.)))

(defn linear [input weight & [bias]]
  (torch/linear (t/->tensor input) (t/->tensor weight) (->optional bias)))

(defn- ->lp [params dims]
  (let [v (cond
            (number? params) (vec (repeatedly dims #(clojure.core/long params)))
            (sequential? params) (mapv clojure.core/long params)
            :else (throw (IllegalArgumentException. "Params must be a number or a sequence")))]
    (long-array v)))

(defn conv1d [input weight & {:keys [bias stride padding dilation groups] :or {stride 1 padding 0 dilation 1 groups 1}}]
  (torch/conv1d (t/->tensor input) (t/->tensor weight) (->optional bias)
                (->lp stride 1) (->lp padding 1) (->lp dilation 1) (clojure.core/long groups)))

(defn conv2d [input weight & {:keys [bias stride padding dilation groups] :or {stride 1 padding 0 dilation 1 groups 1}}]
  (torch/conv2d (t/->tensor input) (t/->tensor weight) (->optional bias)
                (->lp stride 2) (->lp padding 2) (->lp dilation 2) (clojure.core/long groups)))

(defn conv3d [input weight & {:keys [bias stride padding dilation groups] :or {stride 1 padding 0 dilation 1 groups 1}}]
  (torch/conv3d (t/->tensor input) (t/->tensor weight) (->optional bias)
                (->lp stride 3) (->lp padding 3) (->lp dilation 3) (clojure.core/long groups)))

(defn batch-norm [input running-mean running-var & {:keys [weight bias training momentum eps] :or {training false momentum 0.1 eps 1e-5}}]
  (torch/batch_norm (t/->tensor input) (->optional weight) (->optional bias)
                    (->optional running-mean) (->optional running-var) (boolean training) (double momentum) (double eps) true))

(defn layer-norm [input normalized-shape & {:keys [weight bias eps] :or {eps 1e-5}}]
  (let [shape (if (number? normalized-shape) [normalized-shape] normalized-shape)
        v (org.bytedeco.pytorch.LongVector. (long-array shape))
        lar (org.bytedeco.pytorch.LongArrayRef. v)]
    (torch/layer_norm (t/->tensor input) lar
                      (->optional weight) (->optional bias)
                      (double eps) true)))

(defn group-norm [input num-groups & {:keys [weight bias eps] :or {eps 1e-5}}]
  (torch/group_norm (t/->tensor input) (clojure.core/long num-groups) (->optional weight)
                    (->optional bias) (double eps) true))

(defn pixel-shuffle [input upscale-factor]
  (torch/pixel_shuffle (t/->tensor input) (clojure.core/long upscale-factor)))

(defn pixel-unshuffle [input downscale-factor]
  (torch/pixel_unshuffle (t/->tensor input) (clojure.core/long downscale-factor)))

(defn- ->pad-array [pad dims]
  (let [v (cond
            (number? pad) (vec (repeatedly (* 2 dims) #(clojure.core/long pad)))
            (sequential? pad) (mapv clojure.core/long pad)
            :else (throw (IllegalArgumentException. "Padding must be a number or a sequence")))]
    (long-array v)))

(defn pad [input padding & {:keys [mode value] :or {mode :constant value 0.0}}]
  (let [p-arr (if (number? padding)
                (->pad-array padding (- (count (t/size input)) 2))
                (long-array (mapv clojure.core/long padding)))]
    (torch/pad (t/->tensor input) p-arr (clojure.core/name mode) (org.bytedeco.pytorch.DoubleOptional. (double value)))))

(defn interpolate [input & {:keys [size scale-factor mode] :or {mode :nearest}}]
  (cond
    (= mode :nearest)
    (torch/upsample_nearest2d (t/->tensor input) (if size (long-array (if (number? size) [size size] size)) (long-array []))
                              (if scale-factor (org.bytedeco.pytorch.DoubleOptional. (double (if (number? scale-factor) scale-factor (first scale-factor)))) (org.bytedeco.pytorch.DoubleOptional.))
                              (if (and scale-factor (not (number? scale-factor))) (org.bytedeco.pytorch.DoubleOptional. (double (second scale-factor))) (org.bytedeco.pytorch.DoubleOptional.)))
    :else (throw (Exception. "Only nearest interpolation currently implemented functional layer."))))

(defn cosine-similarity [x1 x2 & {:keys [dim eps] :or {dim 1 eps 1e-8}}]
  (torch/cosine_similarity (t/->tensor x1) (t/->tensor x2) (clojure.core/long dim) (double eps)))

(defn pairwise-distance [x1 x2 & {:keys [p eps keepdim] :or {p 2.0 eps 1e-6 keepdim false}}]
  (torch/pairwise_distance (t/->tensor x1) (t/->tensor x2) (double p) (double eps) (boolean keepdim)))

(defn mse-loss [input target]
  (torch/mse_loss (t/->tensor input) (t/->tensor target)))

(defn l1-loss [input target]
  (torch/l1_loss (t/->tensor input) (t/->tensor target)))

(defn smooth-l1-loss [input target]
  (torch/smooth_l1_loss (t/->tensor input) (t/->tensor target)))

(defn cross-entropy [logits targets]
  (torch/cross_entropy (t/->tensor logits) (t/->tensor targets)))

(defn nll-loss [input target]
  (torch/nll_loss (t/->tensor input) (t/->tensor target)))

(defn bce-loss [input target]
  (torch/binary_cross_entropy (t/->tensor input) (t/->tensor target)))

(defn bce-with-logits-loss [input target]
  (torch/binary_cross_entropy_with_logits (t/->tensor input) (t/->tensor target)))
