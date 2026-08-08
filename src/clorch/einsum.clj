(ns clorch.einsum
  "Einstein-notation eDSL on top of clorch.torch/einsum."
  (:require [clojure.string :as str]
            [clorch.torch :as torch]))

(def ^:private index-symbols
  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

(defn- indexed-tensor-term?
  [form]
  (and (seq? form)
       (symbol? (first form))
       (every? symbol? (rest form))))

(defn- flatten-product
  [expr]
  (if (and (seq? expr) (= '* (first expr)))
    (mapcat flatten-product (rest expr))
    [expr]))

(defn- parse-rhs
  [expr]
  (let [factors (flatten-product expr)]
    (reduce (fn [acc factor]
              (cond
                (indexed-tensor-term? factor)
                (update acc :terms conj {:tensor (first factor)
                                         :indices (vec (rest factor))
                                         :form factor})

                (seq? factor)
                (throw (ex-info "Only multiplication of indexed tensor terms is supported."
                                {:expr expr
                                 :unsupported-factor factor}))

                :else
                (update acc :scalars conj factor)))
            {:terms [] :scalars []}
            factors)))

(defn- ensure-output-indices!
  [output-indices]
  (when-not (vector? output-indices)
    (throw (ex-info "Output indices must be a vector of symbols."
                    {:output-indices output-indices})))
  (when-not (every? symbol? output-indices)
    (throw (ex-info "Output indices must contain only symbols."
                    {:output-indices output-indices}))))

(defn- index-map
  [all-indices]
  (let [unique-indices (vec (distinct all-indices))]
    (when (> (count unique-indices) (count index-symbols))
      (throw (ex-info "Too many unique indices for einsum encoding."
                      {:index-count (count unique-indices)
                       :max-supported (count index-symbols)})))
    (zipmap unique-indices (map str (take (count unique-indices) index-symbols)))))

(defn- build-einsum-form
  [output-indices expr]
  (ensure-output-indices! output-indices)
  (let [{:keys [terms scalars]} (parse-rhs expr)]
    (when (empty? terms)
      (throw (ex-info "At least one indexed tensor term is required."
                      {:expr expr})))
    (let [all-indices (mapcat :indices terms)
          idx->label (index-map all-indices)]
      (doseq [idx output-indices]
        (when-not (contains? idx->label idx)
          (throw (ex-info "Output index does not appear in RHS tensor terms."
                          {:output-index idx
                           :rhs expr}))))
      (let [term-subscripts (mapv (fn [{:keys [indices]}]
                                    (apply str (map idx->label indices)))
                                  terms)
            output-subscript (apply str (map idx->label output-indices))
            equation (str (str/join "," term-subscripts) "->" output-subscript)
            tensor-syms (mapv :tensor terms)
            term-specs (mapv (fn [{:keys [tensor indices]}]
                               {:tensor-name (clojure.core/name tensor)
                                :indices (mapv clojure.core/name indices)
                                :tensor tensor})
                             terms)
            output-names (mapv clojure.core/name output-indices)
            base `(let [term-specs# [~@(map (fn [{:keys [tensor-name indices tensor]}]
                                              `{:tensor-name ~tensor-name
                                                :indices ~indices
                                                :tensor ~tensor})
                                            term-specs)]
                        _# (clorch.einsum/validate-index-dimensions! ~equation term-specs# ~output-names)]
                    (torch/einsum ~equation [~@tensor-syms]))]
        (reduce (fn [acc scalar]
                  `(torch/mul ~acc ~scalar))
                base
                scalars)))))

(defn validate-index-dimensions!
  "Validates index dimensionality consistency before dispatching to torch/einsum.
   Each term spec is a map: {:tensor-name string :indices [\"i\" ...] :tensor t}."
  [equation term-specs output-indices]
  (let [index->dim (volatile! {})]
    (doseq [{:keys [tensor-name indices tensor]} term-specs]
      (let [shape (torch/size tensor)]
        (when (not= (count indices) (count shape))
          (throw (ex-info "Rank mismatch between tensor indices and tensor shape."
                          {:equation equation
                           :tensor tensor-name
                           :indices indices
                           :shape shape})))
        (doseq [[axis idx] (map-indexed vector indices)]
          (let [dim (nth shape axis)
                prev (get @index->dim idx)]
            (if (and (some? prev) (not= prev dim))
              (throw (ex-info "Incompatible dimension for einsum index."
                              {:equation equation
                               :index idx
                               :previous-dim prev
                               :current-dim dim
                               :tensor tensor-name
                               :axis axis}))
              (vswap! index->dim assoc idx dim))))))
    (doseq [idx output-indices]
      (when-not (contains? @index->dim idx)
        (throw (ex-info "Output index is not bound to any tensor dimension."
                        {:equation equation
                         :output-index idx}))))
    true))

(defmacro ein
  "Anonymous Einstein-notation form that lowers to `clorch.torch/einsum`.

   Example:
   (ein [i] := (* (A i j) (x j)))"
  [output-indices assign-op expr]
  (when-not (= assign-op :=)
    (throw (ex-info "Expected := between output indices and expression."
                    {:assign-op assign-op})))
  (build-einsum-form output-indices expr))

(defmacro defein
  "Defines a var whose value is computed using Einstein notation.

   Example:
   (defein y [i] := (* (A i j) (x j)))"
  [var-name output-indices assign-op expr]
  (when-not (= assign-op :=)
    (throw (ex-info "Expected := between output indices and expression."
                    {:assign-op assign-op})))
  `(def ~var-name ~(build-einsum-form output-indices expr)))
