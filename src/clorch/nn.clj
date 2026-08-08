(ns clorch.nn
  (:refer-clojure :exclude [flatten identity apply])
  (:require [clojure.string :as str]
            [clorch.torch :as torch]
            [clorch.nn.functional :as F]
            [clorch.autograd :as autograd])
  (:import [org.bytedeco.pytorch LinearImpl LinearOptions ReLUImpl SigmoidImpl SoftmaxImpl DropoutImpl DropoutOptions FlattenImpl Conv1dImpl Conv1dOptions Conv2dImpl Conv2dOptions Conv3dImpl Conv3dOptions ConvTranspose1dImpl ConvTranspose1dOptions ConvTranspose2dImpl ConvTranspose2dOptions ConvTranspose3dImpl ConvTranspose3dOptions BatchNorm1dImpl BatchNorm2dImpl BatchNorm3dImpl BatchNormOptions GroupNormImpl GroupNormOptions InstanceNorm1dImpl InstanceNorm2dImpl InstanceNorm3dImpl InstanceNormOptions RNNImpl RNNOptions LSTMImpl LSTMOptions GRUImpl GRUOptions EmbeddingImpl EmbeddingOptions Module Tensor TensorVector TanhImpl ELUImpl SELUImpl CELUImpl CELUOptions ELUOptions LeakyReLUImpl LeakyReLUOptions ReLU6Impl SoftplusImpl SoftplusOptions SoftsignImpl SoftminImpl MishImpl HardtanhImpl HardtanhOptions LogSigmoidImpl LogSoftmaxImpl HardshrinkImpl HardshrinkOptions SoftshrinkImpl SoftshrinkOptions TanhshrinkImpl ThresholdImpl ThresholdOptions GLUImpl GLUOptions RReLUImpl RReLUOptions AlphaDropoutImpl FeatureAlphaDropoutImpl BilinearImpl BilinearOptions]
           [org.bytedeco.javacpp LongPointer]))

;; --- Core Protocols ---

(defprotocol IModule
  (-forward [this x])
  (-train [this bool])
  (-to [this dtype-or-device]))

;; --- Parameter ---

(defrecord Parameter [tensor]
  clorch.torch/ITensorContainer
  (get-tensor [_] tensor))

(defn parameter
  "A kind of Tensor that is to be considered a module parameter."
  [data & {:keys [requires-grad] :or {requires-grad true}}]
  (let [t (torch/->tensor data)
        p-tensor (autograd/detach t)]
    (autograd/set-requires-grad p-tensor (clojure.core/boolean requires-grad))
    (->Parameter p-tensor)))

(defmethod print-method Parameter [p ^java.io.Writer w]
  (.write w "(nn/parameter ")
  (print-method (:tensor p) w)
  (.write w ")"))

;; --- Public Lifecycle API ---

(def ^:dynamic *trace* nil)

(defn forward
  "Performs a forward pass. Works on native modules, vectors, functions, and custom records."
  [model x]
  (if (instance? org.bytedeco.pytorch.Tensor model)
    model ;; Identity if model is actually a tensor
    (let [x-input (if (or (instance? org.bytedeco.pytorch.Tensor x) (map? x) (sequential? x)) x (torch/->tensor x))
          result (-forward model x-input)]
      (when *trace*
        (swap! *trace* conj {:model model :shape (torch/size result)}))
      result)))

(defn train
  "Sets the module in training (true) or evaluation (false) mode."
  [model bool]
  (-train model (clojure.core/boolean bool))
  model)

(defn to
  "Recursively moves a model or collection to a device or dtype."
  [thing dtype-or-device]
  (cond
    (instance? Module thing) (-to thing dtype-or-device)
    (satisfies? IModule thing) (-to thing dtype-or-device)
    (instance? Tensor thing) (torch/to thing dtype-or-device)
    (instance? TensorVector thing)
    (let [out (TensorVector.)]
      (dotimes [i (.size thing)]
        (.push_back out (torch/to (.get thing (clojure.core/long i)) dtype-or-device)))
      out)
    (instance? Parameter thing) (assoc thing :tensor (torch/to (:tensor thing) dtype-or-device))
    (map? thing) (into (empty thing) (for [[k v] thing] [k (to v dtype-or-device)]))
    (sequential? thing) (mapv #(to % dtype-or-device) thing)
    (vector? thing) (mapv #(to % dtype-or-device) thing)
    :else thing))

(defn parameters
  "Recursively finds all parameters (Tensors) in any data structure."
  [thing & {:keys [trainable-only] :or {trainable-only false}}]
  (let [params-collector (atom [])
        out (TensorVector.)]
    (letfn [(walk [node]
              (cond
                (instance? Module node)
                (let [p-vec (.parameters ^Module node)]
                  (dotimes [i (.size p-vec)]
                    (let [param (.get p-vec (clojure.core/long i))]
                      (when (or (not trainable-only) (.requires_grad ^Tensor param))
                        (swap! params-collector conj param)))))

                (instance? Parameter node)
                (let [param (:tensor node)]
                  (when (or (not trainable-only) (.requires_grad ^Tensor param))
                    (swap! params-collector conj param)))

                (instance? Tensor node)
                (when (or (not trainable-only) (.requires_grad ^Tensor node))
                  (swap! params-collector conj node))

                (instance? TensorVector node)
                (dotimes [i (.size node)]
                  (let [p (.get node (clojure.core/long i))]
                    (when (or (not trainable-only) (.requires_grad ^Tensor p))
                      (swap! params-collector conj p))))

                (clojure.core/record? node) (run! walk (vals node))
                (map? node) (run! walk (vals node))
                (sequential? node) (run! walk node)
                (vector? node) (run! walk node)))]
      (walk thing)
      (doseq [p @params-collector]
        (when (instance? Tensor p)
          (.push_back out p)))
      out)))

(defn clip-grad-norm!
  "Clips the total norm of an iterable of parameters."
  [params-list max-norm]
  (let [params (if (instance? org.bytedeco.pytorch.TensorVector params-list)
                 params-list
                 (let [v (org.bytedeco.pytorch.TensorVector.)]
                   (doseq [p params-list] (.push_back v (torch/->tensor p)))
                   v))]
    (torch/clip-grad-norm-raw params (clojure.core/double max-norm))))

(defn state-dict
  "Recursively extracts the state dictionary (weights/biases) of a model."
  [model]
  (letfn [(normalize-key [k]
            (cond
              (keyword? k) k
              (string? k) (keyword k)
              :else k))]
    (cond
      (instance? Module model)
      (let [named-p (.named_parameters ^Module model)
            pairs (.pairs named-p)
            sz (.size pairs)]
        (into {} (for [i (range sz)]
                   [(normalize-key (.getString (.first pairs (clojure.core/long i))))
                    (.second pairs (clojure.core/long i))])))

      (instance? Parameter model) (:tensor model)
      (instance? Tensor model) model
      (map? model) (into {} (for [[k v] model] [(normalize-key k) (state-dict v)]))
      (vector? model) (into {} (map-indexed (fn [i v] [i (state-dict v)]) model))
      :else nil)))

(defn load-state-dict
  "Recursively loads a state dictionary into a model."
  [model dict]
  (letfn [(lookup [m k]
            (if (map? m)
              (or (get m k)
                  (when (keyword? k) (get m (name k)))
                  (when (string? k) (get m (keyword k))))
              nil))]
    (cond
      (instance? Module model) (do (.load_state_dict ^Module model dict) model)
      (instance? Parameter model) (do (torch/copy- (:tensor model) dict) model)
      (instance? Tensor model) (do (torch/copy- model dict) model)
      (map? model) (do (doseq [[k v] model] (load-state-dict v (lookup dict k))) model)
      (vector? model)
      (do
        (doseq [[i v] (map-indexed vector model)]
          (load-state-dict v (if (map? dict)
                               (or (get dict i)
                                   (get dict (keyword (str i)))
                                   (get dict (str i)))
                               (nth dict i nil))))
        model)
      :else model)))

(defn save-weights
  "Saves the model's state-dict (weights and biases) to disk.
   This is the preferred way to save models for portability."
  [model path]
  (torch/save (state-dict model) path))

(defn load-weights
  "Loads weights from a state-dict file into an existing model."
  [model path]
  (torch/load model path))

;; --- Protocol Implementations ---

(extend-type org.bytedeco.pytorch.Module
  IModule
  (-forward [this x] (.forward this x))
  (-train [this bool] (.train this bool))
  (-to [this dtype-or-device]
    (cond
      (keyword? dtype-or-device)
      (if-let [stype (get torch/dtype-map dtype-or-device)]
        (.to this stype false)
        (.to this (org.bytedeco.pytorch.Device. (name dtype-or-device)) false))

      (instance? org.bytedeco.pytorch.Device dtype-or-device)
      (.to this ^org.bytedeco.pytorch.Device dtype-or-device false)

      :else (throw (IllegalArgumentException. (str "Unsupported device/dtype: " dtype-or-device))))
    this))

(extend-type clojure.lang.APersistentVector
  IModule
  (-forward [this x] (reduce (fn [acc layer] (forward layer acc)) x this))
  (-train [this bool] (run! #(train % bool) this) this)
  (-to [this dtype-or-device] (mapv #(to % dtype-or-device) this)))

(extend-type clojure.lang.APersistentMap
  IModule
  (-forward [_ _]
    (throw (Exception. "Cannot call forward on a raw map.")))
  (-train [this bool]
    (run! #(train % bool) (vals this))
    this)
  (-to [this dtype-or-device]
    (into (empty this) (for [[k v] this] [k (to v dtype-or-device)]))))

(extend-type clojure.lang.AFunction
  IModule
  (-forward [this x] (this x))
  (-train [this _] this)
  (-to [this _] this))

;; --- Introspection and Management ---

(defn zero-grad
  "Clears the gradients of all parameters in the model or collection."
  [thing]
  (let [params (parameters thing)]
    (dotimes [i (.size params)]
      (let [p (.get params (clojure.core/long i))
            g (.grad p)]
        (when (not (.isNull g))
          (.zero_ g))))
    thing))

(defn apply
  "Recursively applies f to every module (Native Module or Record) in the tree."
  [thing f]
  (letfn [(walk [node]
            (when (or (instance? Module node) (clojure.core/record? node))
              (f node))
            (cond
              (map? node) (run! walk (vals node))
              (vector? node) (run! walk node)))]
    (walk thing)
    thing))
(defn modules
  "Returns a sequence of all modules (Native or Records) in the tree."
  [thing]
  (let [res (atom [])]
    (letfn [(walk [node]
              (when (or (instance? Module node) (clojure.core/record? node))
                (swap! res conj node))
              (cond
                (map? node) (run! walk (vals node))
                (sequential? node) (run! walk node)
                (vector? node) (run! walk node)))]
      (walk thing)
      @res)))

(defn- get-params-count
  ([m] (get-params-count m false))
  ([m trainable-only]
   (let [p-vec (parameters m :trainable-only trainable-only)
         sz (.size p-vec)]
     (reduce + (for [i (range sz)] (.numel (.get p-vec (clojure.core/long i))))))))

;; --- Printing ---

(defn- module-name [^Module m]
  (let [full-name (.getString (.name m))] (str/replace full-name #"^torch::nn::(.*)Impl$" "$1")))

(defn- tensor-shape [^Tensor t]
  (let [s (.sizes t)] (mapv #(.get s (clojure.core/long %)) (range (.dim t)))))

(defn- module-params [^Module m]
  (let [named-params (.named_parameters m) pairs (.pairs named-params) sz (.size pairs)]
    (mapv (fn [i] [(.getString (.first pairs i)) (tensor-shape (.second pairs i))]) (range sz))))

(defn- module-string
  ([m] (module-string m 0))
  ([m indent]
   (let [indent-str (clojure.core/apply str (repeat (* indent 2) " "))]
     (cond
       (instance? Module m)
       (let [module-label (module-name m) params (module-params m)
             params-str (str/join ", " (for [[k s] params] (str k "=" s)))]
         (str module-label "(" params-str ")"))
       (map? m)
       (str (if (record? m) (.getSimpleName (type m)) "Module") " {\n"
            (str/join "\n" (for [[k v] m] (str indent-str "  " k ": " (module-string v (inc indent)))))
            "\n" indent-str "}")
       (vector? m)
       (str "Sequential [\n"
            (str/join "\n" (map-indexed #(str indent-str "  (" %1 "): " (module-string %2 (inc indent))) m))
            "\n" indent-str "]")
       (fn? m) "FunctionalLayer()"
       :else (str m)))))

(defmethod print-method org.bytedeco.pytorch.Module [m ^java.io.Writer w]
  (.write w (module-string m)))

(defn summary
  "Prints a summary of the model architecture, inspired by torchsummary.
   Works across Cider, Terminal, and Jupyter.
   `input` can be either a shape vector (e.g., [1 10]) or a direct input value/map."
  [model input]
  (let [model (train model false)
        trace-atom (atom [])
        _ (autograd/no-grad
           (binding [*trace* trace-atom]
             (forward model (if (vector? input) (torch/randn input) input))))
        trace @trace-atom
        total-params (get-params-count model)
        trainable-params (get-params-count model true)]
    (println "\n----------------------------------------------------------------")
    (printf "%-30s %-20s %-10s\n" "Layer (type)" "Output Shape" "Param #")
    (println "================================================================")
    (doseq [{:keys [model shape]} trace]
      (let [module-label (if (instance? Module model)
                           (module-name model)
                           (let [c (type model)]
                             (if (fn? model) "Functional" (.getSimpleName ^Class c))))
            params (get-params-count model)]
        (printf "%-30s %-20s %-10d\n" module-label (str shape) params)))
    (println "================================================================")
    (printf "Total params: %d\n" total-params)
    (printf "Trainable params: %d\n" trainable-params)
    (printf "Non-trainable params: %d\n" (- total-params trainable-params))
    (println "----------------------------------------------------------------\n")
    nil))

(defn named-parameters
  "Returns a flattened map of name paths to parameters."
  ([thing] (named-parameters thing ""))
  ([thing prefix]
   (let [res (atom {})]
     (letfn [(walk [node path]
               (cond
                 (instance? Module node)
                 (let [named-p (.named_parameters ^Module node)]
                   (dotimes [i (.size named-p)]
                     (let [pair (.get (.pairs named-p) (clojure.core/long i))
                           param-name (.getString (.first pair))
                           tensor (.second pair)
                           full-name (if (empty? path) param-name (str path "." param-name))]
                       (swap! res assoc full-name tensor))))

                 (instance? Parameter node)
                 (swap! res assoc path (:tensor node))

                 (instance? Tensor node)
                 (swap! res assoc path node)

                 (map? node)
                 (doseq [[k v] node]
                   (walk v (if (empty? path) (clojure.core/name k) (str path "." (clojure.core/name k)))))

                 (vector? node)
                 (doseq [[i v] (map-indexed vector node)]
                   (walk v (str path "[" i "]")))))]
       (walk thing prefix)
       @res))))

;; --- Layer Implementations ---

(defrecord LayerNormRecord [gamma beta eps]
  IModule
  (-forward [this x]
    (let [mean (torch/mean x -1 :keepdim true)
          v (torch/var x -1 :keepdim true :unbiased false)
          norm-x (torch/div (torch/sub x mean) (torch/sqrt (torch/add v (:eps this))))]
      (torch/add (torch/mul norm-x (:gamma this)) (:beta this))))
  (-train [this _] this)
  (-to [this dtype-or-device]
    (assoc this
           :gamma (to (:gamma this) dtype-or-device)
           :beta (to (:beta this) dtype-or-device))))

(defn layernorm
  "Applies Layer Normalization over a mini-batch of inputs."
  [normalized-shape & {:keys [eps] :or {eps 1e-5}}]
  (let [shape (if (number? normalized-shape) [(clojure.core/long normalized-shape)] (mapv clojure.core/long normalized-shape))]
    (->LayerNormRecord
     (parameter (torch/ones shape))
     (parameter (torch/zeros shape))
     eps)))

(defrecord RMSNormRecord [weight eps]
  IModule
  (-forward [this x]
    (let [;; x: [..., dim]
          x-t (torch/->tensor x)
          float-x (torch/to-float x-t)
          pow-x (torch/pow float-x 2)
          mean-x (torch/mean pow-x -1 :keepdim true)
          norm-x (torch/mul float-x (torch/rsqrt (torch/add mean-x (:eps this))))]
      (torch/mul (torch/to norm-x (.scalar_type x-t)) (:weight this))))
  (-train [this _] this)
  (-to [this dtype-or-device]
    (assoc this :weight (to (:weight this) dtype-or-device))))

(defn rmsnorm
  "Applies Root Mean Square Layer Normalization."
  [dim & {:keys [eps] :or {eps 1e-6}}]
  (->RMSNormRecord (parameter (torch/ones [dim])) eps))

;; --- Native Module Helpers ---

(defn- ->lp [params dims]
  (let [v (cond
            (number? params) (vec (repeatedly dims #(clojure.core/long params)))
            (sequential? params) (mapv clojure.core/long params)
            :else (throw (IllegalArgumentException. "Params must be a number or a sequence")))]
    (LongPointer. (long-array v))))

(defn- set-lp! [lp values dims]
  (.put lp (->lp values dims))
  lp)

;; --- Functional Record Wrappers ---

(defrecord FunctionalModule [f args]
  IModule
  (-forward [_ x] (clojure.core/apply f x args))
  (-train [this _] this)
  (-to [this _] this))

(defn- func-layer [f & args]
  (fn [x] (clojure.core/apply f x args)))

;; --- Native Module Constructors ---

(defn embedding [num-embeddings embedding-dim & {:keys [padding-idx max-norm norm-type scale-grad-by-freq sparse _weight _freeze]
                                                 :or {scale-grad-by-freq false sparse false _freeze true}}]
  (let [opts (EmbeddingOptions. (clojure.core/long num-embeddings) (clojure.core/long embedding-dim))]
    (when padding-idx (.put (.padding_idx opts) (org.bytedeco.pytorch.LongOptional. (clojure.core/long padding-idx))))
    (when max-norm
      (.put (.max_norm opts) (org.bytedeco.pytorch.DoubleOptional. (clojure.core/double max-norm)))
      (.put (.norm_type opts) (clojure.core/double (or norm-type 2.0))))
    (when scale-grad-by-freq (.put (.scale_grad_by_freq opts) (clojure.core/boolean scale-grad-by-freq)))
    (when sparse (.put (.sparse opts) (clojure.core/boolean sparse)))
    (let [emb (EmbeddingImpl. opts)]
      (when _weight (torch/copy- (.weight emb) _weight))
      (let [p-vec (parameters emb)]
        (dotimes [i (.size p-vec)]
          (autograd/set-requires-grad (.get p-vec (clojure.core/long i)) (not _freeze))))
      (when _freeze (.eval emb))
      emb)))

(defn embedding-from-pretrained [embeddings & {:keys [freeze] :as opts}]
  (let [shape (torch/size embeddings)
        _freeze (if (contains? opts :freeze) freeze true)
        clean-opts (-> opts (dissoc :freeze) (assoc :_freeze _freeze) (assoc :_weight embeddings))]
    (clojure.core/apply embedding (first shape) (clojure.core/second shape) (clojure.core/flatten (clojure.core/seq clean-opts)))))

(defn linear [in out & {:keys [bias] :or {bias true}}]
  (let [opts (LinearOptions. (clojure.core/long in) (clojure.core/long out))]
    (.put (.bias opts) (clojure.core/boolean bias))
    (LinearImpl. opts)))

(defn conv1d [in out kernel & {:keys [stride padding dilation groups bias] :or {stride 1 padding 0 dilation 1 groups 1 bias true}}]
  (let [opts (Conv1dOptions. (clojure.core/long in) (clojure.core/long out) (->lp kernel 1))]
    (set-lp! (.stride opts) (or stride kernel) 1)
    (set-lp! (.padding opts) padding 1)
    (set-lp! (.dilation opts) dilation 1)
    (.put (.groups opts) (clojure.core/long groups))
    (.put (.bias opts) (clojure.core/boolean bias))
    (Conv1dImpl. opts)))

(defn conv2d [in out kernel & {:keys [stride padding dilation groups bias] :or {stride 1 padding 0 dilation 1 groups 1 bias true}}]
  (let [opts (Conv2dOptions. (clojure.core/long in) (clojure.core/long out) (->lp kernel 2))]
    (set-lp! (.stride opts) (or stride kernel) 2)
    (set-lp! (.padding opts) padding 2)
    (set-lp! (.dilation opts) dilation 2)
    (.put (.groups opts) (clojure.core/long groups))
    (.put (.bias opts) (clojure.core/boolean bias))
    (Conv2dImpl. opts)))

(defn conv3d [in out kernel & {:keys [stride padding dilation groups bias] :or {stride 1 padding 0 dilation 1 groups 1 bias true}}]
  (let [opts (Conv3dOptions. (clojure.core/long in) (clojure.core/long out) (->lp kernel 3))]
    (set-lp! (.stride opts) (or stride kernel) 3)
    (set-lp! (.padding opts) padding 3)
    (set-lp! (.dilation opts) dilation 3)
    (.put (.groups opts) (clojure.core/long groups))
    (.put (.bias opts) (clojure.core/boolean bias))
    (Conv3dImpl. opts)))

(defn conv-transpose1d [in out kernel & {:keys [stride padding output-padding groups bias dilation] :or {stride 1 padding 0 output-padding 0 groups 1 bias true dilation 1}}]
  (let [opts (ConvTranspose1dOptions. (clojure.core/long in) (clojure.core/long out) (->lp kernel 1))]
    (set-lp! (.stride opts) stride 1)
    (set-lp! (.padding opts) padding 1)
    (set-lp! (.output_padding opts) output-padding 1)
    (.put (.groups opts) (clojure.core/long groups))
    (.put (.bias opts) (clojure.core/boolean bias))
    (set-lp! (.dilation opts) dilation 1)
    (ConvTranspose1dImpl. opts)))

(defn conv-transpose2d [in out kernel & {:keys [stride padding output-padding groups bias dilation] :or {stride 1 padding 0 output-padding 0 groups 1 bias true dilation 1}}]
  (let [opts (ConvTranspose2dOptions. (clojure.core/long in) (clojure.core/long out) (->lp kernel 2))]
    (set-lp! (.stride opts) (if (number? stride) [stride stride] stride) 2)
    (set-lp! (.padding opts) (if (number? padding) [padding padding] padding) 2)
    (set-lp! (.output_padding opts) (if (number? output-padding) [output-padding output-padding] output-padding) 2)
    (.put (.groups opts) (clojure.core/long groups))
    (.put (.bias opts) (clojure.core/boolean bias))
    (set-lp! (.dilation opts) (if (number? dilation) [dilation dilation] dilation) 2)
    (ConvTranspose2dImpl. opts)))

(defn conv-transpose3d [in out kernel & {:keys [stride padding output-padding groups bias dilation] :or {stride 1 padding 0 output-padding 0 groups 1 bias true dilation 1}}]
  (let [opts (ConvTranspose3dOptions. (clojure.core/long in) (clojure.core/long out) (->lp kernel 3))]
    (set-lp! (.stride opts) (if (number? stride) [stride stride stride] stride) 3)
    (set-lp! (.padding opts) (if (number? padding) [padding padding padding] padding) 3)
    (set-lp! (.output_padding opts) (if (number? output-padding) [output-padding output-padding output-padding] output-padding) 3)
    (.put (.groups opts) (clojure.core/long groups))
    (.put (.bias opts) (clojure.core/boolean bias))
    (set-lp! (.dilation opts) (if (number? dilation) [dilation dilation dilation] dilation) 3)
    (ConvTranspose3dImpl. opts)))

(defn batchnorm1d [features] (BatchNorm1dImpl. (BatchNormOptions. (clojure.core/long features))))
(defn batchnorm2d [features] (BatchNorm2dImpl. (BatchNormOptions. (clojure.core/long features))))
(defn batchnorm3d [features] (BatchNorm3dImpl. (BatchNormOptions. (clojure.core/long features))))

(defn groupnorm [num-groups num-channels & {:keys [eps affine] :or {eps 1e-5 affine true}}]
  (let [opts (GroupNormOptions. (clojure.core/long num-groups) (clojure.core/long num-channels))]
    (.put (.eps opts) (clojure.core/double eps))
    (.put (.affine opts) (clojure.core/boolean affine))
    (GroupNormImpl. opts)))

(defn instancenorm1d [features] (InstanceNorm1dImpl. (InstanceNormOptions. (clojure.core/long features))))
(defn instancenorm2d [features] (InstanceNorm2dImpl. (InstanceNormOptions. (clojure.core/long features))))
(defn instancenorm3d [features] (InstanceNorm3dImpl. (InstanceNormOptions. (clojure.core/long features))))

(defn- setup-rnn-opts [opts hidden-size num-layers bias batch-first dropout bidirectional]
  (.put (.hidden_size opts) (clojure.core/long hidden-size))
  (when num-layers (.put (.num_layers opts) (clojure.core/long num-layers)))
  (when (some? bias) (.put (.bias opts) (clojure.core/boolean bias)))
  (when (some? batch-first) (.put (.batch_first opts) (clojure.core/boolean batch-first)))
  (when dropout (.put (.dropout opts) (clojure.core/double dropout)))
  (when (some? bidirectional) (.put (.bidirectional opts) (clojure.core/boolean bidirectional)))
  opts)

(defn rnn [input-size hidden-size & {:keys [num-layers bias batch-first dropout bidirectional]}]
  (let [opts (RNNOptions. (clojure.core/long input-size) (clojure.core/long hidden-size))]
    (setup-rnn-opts opts hidden-size num-layers bias batch-first dropout bidirectional)
    (RNNImpl. opts)))

(defn lstm [input-size hidden-size & {:keys [num-layers bias batch-first dropout bidirectional]}]
  (let [opts (LSTMOptions. (clojure.core/long input-size) (clojure.core/long hidden-size))]
    (setup-rnn-opts opts hidden-size num-layers bias batch-first dropout bidirectional)
    (LSTMImpl. opts)))

(defn gru [input-size hidden-size & {:keys [num-layers bias batch-first dropout bidirectional]}]
  (let [opts (GRUOptions. (clojure.core/long input-size) (clojure.core/long hidden-size))]
    (setup-rnn-opts opts hidden-size num-layers bias batch-first dropout bidirectional)
    (GRUImpl. opts)))

(defn max-pool1d [kernel & {:keys [stride padding dilation ceil-mode] :or {padding 0 dilation 1 ceil-mode false}}]
  (func-layer F/max-pool1d kernel :stride stride :padding padding :dilation dilation :ceil-mode ceil-mode))

(defn max-pool2d [kernel & {:keys [stride padding dilation ceil-mode] :or {padding 0 dilation 1 ceil-mode false}}]
  (func-layer F/max-pool2d kernel :stride stride :padding padding :dilation dilation :ceil-mode ceil-mode))

(defn max-pool3d [kernel & {:keys [stride padding dilation ceil-mode] :or {padding 0 dilation 1 ceil-mode false}}]
  (func-layer F/max-pool3d kernel :stride stride :padding padding :dilation dilation :ceil-mode ceil-mode))

(defn avg-pool1d [kernel & {:keys [stride padding count-include-pad ceil-mode] :or {padding 0 count-include-pad true ceil-mode false}}]
  (func-layer F/avg-pool1d kernel :stride stride :padding padding :count-include-pad count-include-pad :ceil-mode ceil-mode))

(defn avg-pool2d [kernel & {:keys [stride padding count-include-pad ceil-mode divisor-override] :or {padding 0 count-include-pad true ceil-mode false}}]
  (func-layer F/avg-pool2d kernel :stride stride :padding padding :count-include-pad count-include-pad :ceil-mode ceil-mode :divisor-override divisor-override))

(defn avg-pool3d [kernel & {:keys [stride padding count-include-pad ceil-mode divisor-override] :or {padding 0 count-include-pad true ceil-mode false}}]
  (func-layer F/avg-pool3d kernel :stride stride :padding padding :count-include-pad count-include-pad :ceil-mode ceil-mode :divisor-override divisor-override))

(defn adaptive-max-pool1d [output-size] (func-layer F/adaptive-max-pool1d output-size))
(defn adaptive-max-pool2d [output-size] (func-layer F/adaptive-max-pool2d output-size))
(defn adaptive-max-pool3d [output-size] (func-layer F/adaptive-max-pool3d output-size))

(defn adaptive-avg-pool1d [output-size] (func-layer F/adaptive-avg-pool1d output-size))
(defn adaptive-avg-pool2d [output-size] (func-layer F/adaptive-avg-pool2d output-size))
(defn adaptive-avg-pool3d [output-size] (func-layer F/adaptive-avg-pool3d output-size))

(defn reflection-pad1d [padding] (func-layer F/pad padding :mode :reflect))
(defn reflection-pad2d [padding] (func-layer F/pad padding :mode :reflect))
(defn reflection-pad3d [padding] (func-layer F/pad padding :mode :reflect))

(defn replication-pad1d [padding] (func-layer F/pad padding :mode :replicate))
(defn replication-pad2d [padding] (func-layer F/pad padding :mode :replicate))
(defn replication-pad3d [padding] (func-layer F/pad padding :mode :replicate))

(defn zeropad1d [padding] (func-layer F/pad padding :mode :constant :value 0.0))
(defn zeropad2d [padding] (func-layer F/pad padding :mode :constant :value 0.0))
(defn zeropad3d [padding] (func-layer F/pad padding :mode :constant :value 0.0))

(defn constant-pad1d [padding value] (func-layer F/pad padding :mode :constant :value value))
(defn constant-pad2d [padding value] (func-layer F/pad padding :mode :constant :value value))
(defn constant-pad3d [padding value] (func-layer F/pad padding :mode :constant :value value))

(defn pixel-shuffle [upscale-factor] (func-layer F/pixel-shuffle upscale-factor))
(defn pixel-unshuffle [downscale-factor] (func-layer F/pixel-unshuffle downscale-factor))

(defn upsample [& {:keys [size scale-factor mode align-corners]}]
  (func-layer F/interpolate :size size :scale-factor scale-factor :mode (or mode :nearest) :align-corners align-corners))

(defrecord PairwiseModuleRecord [impl]
  IModule
  (-forward [_ x] (.forward impl (torch/->tensor (first x)) (torch/->tensor (second x))))
  (-train [this _] this)
  (-to [this _] this))

(defn bilinear [in1 in2 out & {:keys [bias] :or {bias true}}]
  (let [opts (BilinearOptions. (clojure.core/long in1) (clojure.core/long in2) (clojure.core/long out))]
    (.put (.bias opts) (clojure.core/boolean bias))
    (->PairwiseModuleRecord (BilinearImpl. opts))))

(defn cosine-similarity [& {:keys [dim eps] :or {dim 1 eps 1e-8}}]
  (let [opts (org.bytedeco.pytorch.CosineSimilarityOptions.)]
    (.put (.dim opts) (clojure.core/long dim))
    (.put (.eps opts) (clojure.core/double eps))
    (->PairwiseModuleRecord (org.bytedeco.pytorch.CosineSimilarityImpl. opts))))

(defn pairwise-distance [& {:keys [p eps keepdim] :or {p 2.0 eps 1e-6 keepdim false}}]
  (let [opts (org.bytedeco.pytorch.PairwiseDistanceOptions.)]
    (.put (.p opts) (clojure.core/double p))
    (.put (.eps opts) (clojure.core/double eps))
    (.put (.keepdim opts) (clojure.core/boolean keepdim))
    (->PairwiseModuleRecord (org.bytedeco.pytorch.PairwiseDistanceImpl. opts))))

(defn relu [] (ReLUImpl.))
(defn relu6 [] (ReLU6Impl.))
(defn leaky-relu [& [negative-slope]]
  (let [opts (LeakyReLUOptions.)]
    (.put (.negative_slope opts) (clojure.core/double (or negative-slope 0.01)))
    (LeakyReLUImpl. opts)))
(defn gelu [] (org.bytedeco.pytorch.GELUImpl.))
(defn tanh [] (TanhImpl.))
(defn sigmoid [] (SigmoidImpl.))
(defn log-sigmoid [] (LogSigmoidImpl.))
(defn silu [] F/silu)
(defn mish [] (MishImpl.))
(defn hardswish [] F/hardswish)
(defn hardsigmoid [] F/hardsigmoid)
(defn hardtanh [& [min-val max-val]]
  (let [opts (HardtanhOptions.)]
    (.put (.min_val opts) (clojure.core/double (or min-val -1.0)))
    (.put (.max_val opts) (clojure.core/double (or max-val 1.0)))
    (HardtanhImpl. opts)))

(defn elu [& [alpha]]
  (let [opts (ELUOptions.)]
    (.put (.alpha opts) (clojure.core/double (or alpha 1.0)))
    (ELUImpl. opts)))
(defn selu [] (SELUImpl.))
(defn celu [& [alpha]]
  (let [opts (CELUOptions.)]
    (.put (.alpha opts) (clojure.core/double (or alpha 1.0)))
    (CELUImpl. opts)))
(defn softplus [& [beta threshold]]
  (let [opts (SoftplusOptions.)]
    (.put (.beta opts) (clojure.core/double (or beta 1.0)))
    (.put (.threshold opts) (clojure.core/double (or threshold 20.0)))
    (SoftplusImpl. opts)))
(defn softsign [] (SoftsignImpl.))
(defn softmin [dim] (SoftminImpl. (clojure.core/long dim)))

(defn log-softmax [dim] (LogSoftmaxImpl. (clojure.core/long dim)))

(defn hardshrink [& [lambd]]
  (HardshrinkImpl. (HardshrinkOptions. (clojure.core/double (or lambd 0.5)))))

(defn softshrink [& [lambd]]
  (SoftshrinkImpl. (SoftshrinkOptions. (clojure.core/double (or lambd 0.5)))))

(defn tanhshrink [] (TanhshrinkImpl.))

(defn threshold [threshold-val value & {:keys [inplace] :or {inplace false}}]
  (let [opts (ThresholdOptions. (clojure.core/double threshold-val) (clojure.core/double value))]
    (.put (.inplace opts) (clojure.core/boolean inplace))
    (ThresholdImpl. opts)))

(defn glu [& [dim]]
  (let [opts (GLUOptions.)]
    (.put (.dim opts) (clojure.core/long (or dim -1)))
    (GLUImpl. opts)))

(defn rrelu [& {:keys [lower upper inplace] :or {lower 0.125 upper 0.3333333333333333 inplace false}}]
  (let [opts (RReLUOptions.)]
    (.put (.lower opts) (clojure.core/double lower))
    (.put (.upper opts) (clojure.core/double upper))
    (.put (.inplace opts) (clojure.core/boolean inplace))
    (RReLUImpl. opts)))

(defrecord PReLURecord [weight]
  IModule
  (-forward [this x] (F/prelu (torch/->tensor x) (:weight this)))
  (-train [this _] this)
  (-to [this dtype-or-device] (assoc this :weight (to (:weight this) dtype-or-device))))

(defn prelu [& [{:keys [num-parameters init] :or {num-parameters 1 init 0.25}}]]
  (->PReLURecord (parameter (torch/mul (torch/ones [(clojure.core/long num-parameters)]) (clojure.core/double init)))))

(defn softmax [dim] (SoftmaxImpl. (clojure.core/long dim)))
(defn dropout [p] (DropoutImpl. (DropoutOptions. (clojure.core/double p))))
(defn dropout2d [p]
  (fn [input] (F/dropout2d input (clojure.core/double p) :training? true)))

(defn alpha-dropout [p] (AlphaDropoutImpl. (DropoutOptions. (clojure.core/double p))))
(defn feature-alpha-dropout [p] (FeatureAlphaDropoutImpl. (DropoutOptions. (clojure.core/double p))))

(defrecord IdentityRecord []
  IModule
  (-forward [_ x] x)
  (-train [this _] this)
  (-to [this _] this))

(defn identity [] (->IdentityRecord))

(defn flatten [] (FlattenImpl.))

(defrecord UnflattenRecord [dim dims]
  IModule
  (-forward [this x] (torch/unflatten x (:dim this) (:dims this)))
  (-train [this _] this)
  (-to [this _] this))

(defn unflatten [dim dims] (->UnflattenRecord dim dims))

(defn sequential [& layers] (vec layers))

;; --- High-Ergonomic Macros ---

(defmacro defmodel [model-name args init-bindings & forward-specs]
  (let [field-syms (take-nth 2 init-bindings)
        forward-fn (first (filter #(= (first %) 'forward) forward-specs))
        forward-args (second forward-fn)
        forward-body (nnext forward-fn)
        record-name (symbol (str model-name "Record"))]
    `(do
       (defrecord ~record-name [~@field-syms]
         IModule
         (-forward [this# ~(first forward-args)]
           (let [{:keys [~@field-syms]} this#] ~@forward-body))
         (-train [this# bool#]
           (doseq [f# [~@field-syms]]
             (let [v# (get this# f#)]
               (when (or (satisfies? IModule v#) (coll? v#))
                 (train v# bool#))))
           this#)
         (-to [this# dtype-or-device#]
           (let [updates# (into {} (for [f# [~@field-syms]]
                                     (let [v# (get this# f#)]
                                       [f# (if (or (satisfies? IModule v#) (coll? v#) (instance? org.bytedeco.pytorch.Tensor v#))
                                             (to v# dtype-or-device#)
                                             v#)])))]
             (clojure.core/merge this# updates#)))
         clojure.lang.IFn
         (invoke [this# x#] (forward this# x#)))
       (defn ~model-name ~args
         (let [~@init-bindings] (~(symbol (str "->" record-name)) ~@field-syms))))))

(defn generate
  "Generates tokens autoregressively."
  [model idx max-new-tokens context-size]
  (autograd/no-grad
   (loop [step 0
          curr-idx idx]
     (if (>= step max-new-tokens)
       curr-idx
       (let [next-idx
             (torch/with-torch
               (let [sz (clojure.core/long (first (clojure.core/drop 1 (torch/size curr-idx))))
                     cond-idx (if (> sz (clojure.core/long context-size))
                                (torch/ix curr-idx :_ [(clojure.core/- sz (clojure.core/long context-size)) sz])
                                curr-idx)
                     logits (forward model cond-idx)
                     last-logits (torch/ix logits :_ -1 :_)
                     probs (F/softmax last-logits -1)
                     next-token (torch/multinomial probs 1)]
                 (torch/cat [curr-idx next-token] 1)))]
         ;; Keep caller-owned input valid; release only internal accumulators.
         (when (pos? step)
           (torch/release! curr-idx))
         (recur (inc step) next-idx))))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(defmodel SwiGLU [dim hidden-dim]
  [w1 (linear dim hidden-dim)
   w2 (linear dim hidden-dim)
   w3 (linear hidden-dim dim)]
  (forward [x]
           (let [gate (F/silu (forward w1 x))
                 feat (forward w2 x)]
             (forward w3 (torch/mul gate feat)))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(defmodel GroupedQueryAttention [dim n-heads n-kv-heads context-len drop-rate]
  [wq (linear dim (* n-heads (quot dim n-heads)))
   wk (linear dim (* n-kv-heads (quot dim n-heads)))
   wv (linear dim (* n-kv-heads (quot dim n-heads)))
   wo (linear dim dim)
   dropout (dropout drop-rate)
   nh n-heads
   n-kvh n-kv-heads
   head-dim (quot dim n-heads)]
  (forward [input]
           (let [{:keys [x mask freqs kv-cache]} (if (map? input) input {:x input})
                 [B T _] (torch/size x)
                 q (forward wq x)
                 k (forward wk x)
                 v (forward wv x)

          ;; Reshape for multi-head: [B, T, nh, head-dim]
                 q (torch/view q [B T nh head-dim])
                 k (torch/view k [B T n-kvh head-dim])
                 v (torch/view v [B T n-kvh head-dim])

          ;; Apply RoPE if provided
                 [q k] (if freqs
                         (let [[cos-emb sin-emb] freqs]
                           [(torch/apply-rope q cos-emb sin-emb)
                            (torch/apply-rope k cos-emb sin-emb)])
                         [q k])

          ;; KV Cache update
                 [k v] (if kv-cache
                         (let [{:keys [k-prev v-prev]} @kv-cache
                               k-curr (if k-prev (torch/cat [k-prev k] 1) k)
                               v-curr (if v-prev (torch/cat [v-prev v] 1) v)]
                           (reset! kv-cache {:k-prev k-curr :v-prev v-curr})
                           [k-curr v-curr])
                         [k v])

          ;; Grouped Query Attention: Repeat K/V if n-kvh < nh
                 k (if (< n-kvh nh) (torch/repeat-interleave k (quot nh n-kvh) 2) k)
                 v (if (< n-kvh nh) (torch/repeat-interleave v (quot nh n-kvh) 2) v)

          ;; Transpose for attention: [B, nh, T, head-dim]
                 q (torch/transpose q 1 2)
                 k (torch/transpose k 1 2)
                 v (torch/transpose v 1 2)

          ;; Scaled dot-product attention
                 scores (torch/div (torch/matmul q (torch/transpose k -2 -1))
                                   (clojure.core/double (Math/sqrt (clojure.core/double head-dim))))
                 scores (if mask (torch/masked-fill scores mask torch/-inf) scores)
                 probs (F/softmax scores -1)
                 probs (forward dropout probs)

          ;; Output
                 out (torch/matmul probs v)
                 out (torch/transpose out 1 2)
                 out (torch/reshape out [B T (* nh head-dim)])]
             (forward wo out))))
