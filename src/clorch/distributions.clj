(ns clorch.distributions
  "Probability distributions built on top of `clorch.torch`.

   API is data-first:
   - constructors return plain maps with keyword keys
   - `sample`, `log-prob`, `mean`, `variance` consume those maps."
  (:require [clorch.torch :as t])
  (:import [org.apache.commons.math3.distribution BetaDistribution ChiSquaredDistribution
            FDistribution GammaDistribution TDistribution WeibullDistribution ParetoDistribution]
           [org.apache.commons.math3.special Gamma]
           [org.bytedeco.pytorch.global torch]))

(def ^:private eps 1e-8)
(def ^:private log-2pi 1.8378770664093453)
(def ^:private log-pi 1.1447298858494002)
(def ^:private euler-gamma 0.5772156649015329)

(defn normal [loc scale] {:dist :normal :loc loc :scale scale})
(defn log-normal [loc scale] {:dist :log-normal :loc loc :scale scale})
(defn bernoulli [probs] {:dist :bernoulli :probs probs})
(defn categorical [probs] {:dist :categorical :probs probs})
(defn binomial [total-count probs] {:dist :binomial :total-count total-count :probs probs})
(defn uniform [low high] {:dist :uniform :low low :high high})
(defn poisson [rate] {:dist :poisson :rate rate})
(defn exponential [rate] {:dist :exponential :rate rate})
(defn cauchy [loc scale] {:dist :cauchy :loc loc :scale scale})
(defn geometric [probs] {:dist :geometric :probs probs})
(defn gumbel [loc scale] {:dist :gumbel :loc loc :scale scale})
(defn laplace [loc scale] {:dist :laplace :loc loc :scale scale})
(defn gamma [concentration rate] {:dist :gamma :concentration concentration :rate rate})
(defn beta [concentration1 concentration0]
  {:dist :beta :concentration1 concentration1 :concentration0 concentration0})
(defn dirichlet [concentration] {:dist :dirichlet :concentration concentration})
(defn student-t [df loc scale] {:dist :student-t :df df :loc loc :scale scale})
(defn chi2 [df] {:dist :chi2 :df df})
(defn fisher-f [df1 df2] {:dist :fisher-f :df1 df1 :df2 df2})
(defn weibull [scale concentration] {:dist :weibull :scale scale :concentration concentration})
(defn pareto [scale alpha] {:dist :pareto :scale scale :alpha alpha})
(defn logistic [loc scale] {:dist :logistic :loc loc :scale scale})
(defn negative-binomial [total-count probs]
  {:dist :negative-binomial :total-count total-count :probs probs})

(defn- maybe-shape [sample-shape]
  (when (seq sample-shape) (vec sample-shape)))

(defn- as-tensor [x]
  (if (number? x) (t/tensor x) (t/->tensor x)))

(defn- const-like [value c]
  (if (number? value)
    c
    (t/add (t/mul (as-tensor value) 0.0) c)))

(defn- numel [shape]
  (if (seq shape)
    (reduce * 1 shape)
    1))

(defn- tensor->flat-vec [x]
  (let [tx (as-tensor x)]
    (mapv t/item-float (t/tseq (t/reshape tx [-1])))))

(defn- nest-shape [flat shape]
  (if (empty? shape)
    (first flat)
    (let [n (first shape)
          step (int (/ (count flat) n))]
      (if (= 1 (count shape))
        (vec flat)
        (mapv #(nest-shape % (rest shape)) (partition step flat))))))

(defn- scalar-samples->tensor [shape sampler]
  (let [n (numel shape)
        flat (repeatedly n sampler)
        nested (nest-shape flat (if (seq shape) shape [1]))]
    (t/tensor nested)))

(defn- vector-samples->tensor [shape sample-one]
  (let [n (numel shape)
        flat (repeatedly n sample-one)]
    (if (seq shape)
      (t/tensor (nest-shape flat shape))
      (t/tensor (first flat)))))

(defn- sample-exponential-like [rate shape]
  (let [u (t/clamp (torch/rand (long-array shape)) eps (- 1.0 eps))]
    (t/div (t/neg (t/log u)) rate)))

(defn sample
  ([dist] (sample dist nil))
  ([{:keys [dist loc scale probs low high rate total-count concentration concentration1 concentration0
            df df1 df2 alpha]
     :as d}
    sample-shape]
   (let [shape (or (maybe-shape sample-shape) [1])]
     (case dist
       :normal
       (if (and (number? loc) (number? scale))
         (t/add (t/mul (t/randn shape) scale) loc)
         (if sample-shape
           (throw (IllegalArgumentException.
                   "normal with sample-shape currently supports numeric loc/scale"))
           (torch/normal (as-tensor loc) (as-tensor scale))))

       :log-normal
       (if (and (number? loc) (number? scale))
         (torch/log_normal (t/zeros shape)
                           (double loc)
                           (double scale)
                           (org.bytedeco.pytorch.GeneratorOptional.))
         (if sample-shape
           (throw (IllegalArgumentException.
                   "log-normal with sample-shape currently supports numeric loc/scale"))
           (t/exp (torch/normal (as-tensor loc) (as-tensor scale)))))

       :bernoulli
       (if sample-shape
         (throw (IllegalArgumentException. "bernoulli sample-shape is not supported yet"))
         (t/bernoulli probs))

       :categorical
       (if sample-shape
         (if (= 1 (count shape))
           (t/multinomial probs (long (first shape)) true)
           (throw (IllegalArgumentException. "categorical sample-shape currently supports one dimension")))
         (t/multinomial probs 1 true))

       :binomial
       (if (and sample-shape (or (not (number? total-count)) (not (number? probs))))
         (throw (IllegalArgumentException.
                 "binomial with sample-shape currently supports numeric total-count/probs"))
         (let [n (if sample-shape (t/full shape (double total-count)) (as-tensor total-count))
               p (if sample-shape (t/full shape (double probs)) (as-tensor probs))]
           (torch/binomial n p)))

       :uniform
       (let [width (- (double high) (double low))]
         (t/add (t/mul (torch/rand (long-array shape)) width) (double low)))

       :poisson
       (if sample-shape
         (if (number? rate)
           (torch/poisson (t/full shape (double rate)))
           (throw (IllegalArgumentException.
                   "poisson with sample-shape currently supports numeric rate")))
         (torch/poisson (as-tensor rate)))

       :exponential
       (if (and sample-shape (not (number? rate)))
         (throw (IllegalArgumentException.
                 "exponential with sample-shape currently supports numeric rate"))
         (sample-exponential-like (if (number? rate) (double rate) rate) shape))

       :cauchy
       (if (and (number? loc) (number? scale))
         (torch/cauchy (t/zeros shape)
                       (double loc)
                       (double scale)
                       (org.bytedeco.pytorch.GeneratorOptional.))
         (let [u (t/clamp (torch/rand (long-array (if sample-shape shape (t/size (as-tensor loc)))))
                          eps
                          (- 1.0 eps))]
           (t/add (as-tensor loc)
                  (t/mul (as-tensor scale)
                         (t/tan (t/mul Math/PI (t/sub u 0.5)))))))

       :geometric
       (if (and (number? probs) (or sample-shape true))
         (torch/geometric (t/zeros shape)
                          (double probs)
                          (org.bytedeco.pytorch.GeneratorOptional.))
         (let [p (t/clamp (as-tensor probs) eps (- 1.0 eps))
               u (t/clamp (torch/rand (long-array (if sample-shape shape (t/size p)))) eps (- 1.0 eps))]
           (t/add (t/floor (t/div (t/log u) (t/log (t/sub 1.0 p)))) 1.0)))

       :gumbel
       (let [u (t/clamp (torch/rand (long-array shape)) eps (- 1.0 eps))]
         (t/sub loc (t/mul scale (t/log (t/neg (t/log u))))))

       :laplace
       (let [u (t/sub (t/clamp (torch/rand (long-array shape)) eps (- 1.0 eps)) 0.5)
             abs-u (t/abs u)
             inner (t/log (t/sub 1.0 (t/mul 2.0 abs-u)))
             sign-u (t/sign u)]
         (t/sub loc (t/mul scale (t/mul sign-u inner))))

       :gamma
       (let [k (double concentration)
             theta (/ 1.0 (double rate))
             d (GammaDistribution. k theta)]
         (scalar-samples->tensor shape #(.sample d)))

       :beta
       (let [a (double concentration1)
             b (double concentration0)
             d (BetaDistribution. a b)]
         (scalar-samples->tensor shape #(.sample d)))

       :dirichlet
       (let [alphas (if (number? concentration)
                      [(double concentration)]
                      (mapv double (tensor->flat-vec concentration)))]
         (vector-samples->tensor
          sample-shape
          (fn []
            (let [gs (mapv (fn [a]
                             (.sample (GammaDistribution. a 1.0)))
                           alphas)
                  s (reduce + gs)]
              (mapv #(/ % s) gs)))))

       :student-t
       (let [d (TDistribution. (double df))]
         (scalar-samples->tensor shape #(+ (double loc) (* (double scale) (.sample d)))))

       :chi2
       (let [d (ChiSquaredDistribution. (double df))]
         (scalar-samples->tensor shape #(.sample d)))

       :fisher-f
       (let [d (FDistribution. (double df1) (double df2))]
         (scalar-samples->tensor shape #(.sample d)))

       :weibull
       (let [d (WeibullDistribution. (double concentration) (double scale))]
         (scalar-samples->tensor shape #(.sample d)))

       :pareto
       (let [d (ParetoDistribution. (double scale) (double alpha))]
         (scalar-samples->tensor shape #(.sample d)))

       :logistic
       (let [u (t/clamp (torch/rand (long-array shape)) eps (- 1.0 eps))]
         (t/add loc (t/mul scale (t/log (t/div u (t/sub 1.0 u))))))

       :negative-binomial
       (let [r (double total-count)
             p (double probs)
             g (GammaDistribution. r (/ (- 1.0 p) p))]
         (scalar-samples->tensor shape #(double (.sample (org.apache.commons.math3.distribution.PoissonDistribution. (.sample g))))))

       (throw (IllegalArgumentException. (str "Unsupported distribution: " d)))))))

(defn log-prob
  [{:keys [dist loc scale probs low high rate total-count concentration concentration1 concentration0
           df df1 df2 alpha]} value]
  (case dist
    :normal
    (let [x (as-tensor value)
          loc-t (as-tensor loc)
          scale-t (as-tensor scale)
          z (t/div (t/sub x loc-t) scale-t)]
      (t/sub (t/sub (t/mul (t/pow z 2.0) -0.5) (t/log scale-t))
             (* 0.5 log-2pi)))

    :log-normal
    (let [x (as-tensor value)
          normal-part (log-prob (normal loc scale) (t/log x))]
      (t/sub normal-part (t/log x)))

    :bernoulli
    (let [x (as-tensor value)
          p (t/clamp (as-tensor probs) eps (- 1.0 eps))]
      (t/add (t/mul x (t/log p))
             (t/mul (t/sub 1.0 x) (t/log (t/sub 1.0 p)))))

    :categorical
    (let [p (t/clamp (as-tensor probs) eps 1.0)
          logp (t/log p)]
      (if (= 1 (count (t/size p)))
        (if (number? value)
          (t/ix logp (long value))
          (let [idx (as-tensor value)]
            (t/index-select logp 0 idx)))
        (let [idx (if (number? value)
                    (t/tensor [[(long value)]] {:dtype :int64})
                    (let [v (as-tensor value)]
                      (if (= 1 (count (t/size v))) (t/unsqueeze v -1) v)))
              g (t/gather logp -1 idx)]
          (if (number? value) (t/ix g 0 0) g))))

    :binomial
    (let [x (as-tensor value)
          n (as-tensor total-count)
          p (t/clamp (as-tensor probs) eps (- 1.0 eps))]
      (t/add
       (t/sub (t/sub (t/lgamma (t/add n 1.0))
                     (t/lgamma (t/add x 1.0)))
              (t/lgamma (t/add (t/sub n x) 1.0)))
       (t/add (t/mul x (t/log p))
              (t/mul (t/sub n x) (t/log (t/sub 1.0 p))))))

    :uniform
    (let [width (- (double high) (double low))]
      (const-like value (- (Math/log width))))

    :poisson
    (let [x (as-tensor value)
          r (as-tensor rate)]
      (t/sub (t/sub (t/mul x (t/log r)) r)
             (t/lgamma (t/add x 1.0))))

    :exponential
    (let [x (as-tensor value)
          r (as-tensor rate)]
      (t/sub (t/log r) (t/mul r x)))

    :cauchy
    (let [x (as-tensor value)
          loc-t (as-tensor loc)
          scale-t (as-tensor scale)
          z (t/div (t/sub x loc-t) scale-t)]
      (t/sub (t/sub (const-like value (- log-pi))
                    (t/log scale-t))
             (t/log (t/add 1.0 (t/pow z 2.0)))))

    :geometric
    (let [x (as-tensor value)
          p (t/clamp (as-tensor probs) eps (- 1.0 eps))]
      (t/add (t/log p)
             (t/mul (t/sub x 1.0) (t/log (t/sub 1.0 p)))))

    :gumbel
    (let [x (as-tensor value)
          loc-t (as-tensor loc)
          scale-t (as-tensor scale)
          z (t/div (t/sub x loc-t) scale-t)]
      (t/sub (t/sub (t/neg z) (t/exp (t/neg z)))
             (t/log scale-t)))

    :laplace
    (let [x (as-tensor value)
          loc-t (as-tensor loc)
          scale-t (as-tensor scale)]
      (t/sub (t/div (t/abs (t/sub x loc-t)) (t/neg scale-t))
             (t/log (t/mul 2.0 scale-t))))

    :gamma
    (let [x (as-tensor value)
          a (as-tensor concentration)
          b (as-tensor rate)]
      (t/add (t/sub (t/sub (t/mul a (t/log b))
                           (t/lgamma a))
                    (t/mul b x))
             (t/mul (t/sub a 1.0) (t/log x))))

    :beta
    (let [x (as-tensor value)
          a (as-tensor concentration1)
          b (as-tensor concentration0)
          log-beta (t/sub (t/add (t/lgamma a) (t/lgamma b))
                          (t/lgamma (t/add a b)))]
      (t/sub (t/add (t/mul (t/sub a 1.0) (t/log x))
                    (t/mul (t/sub b 1.0) (t/log (t/sub 1.0 x))))
             log-beta))

    :dirichlet
    (let [x (as-tensor value)
          a (as-tensor concentration)
          event-dim (dec (count (t/size x)))
          log-norm (t/sub (t/lgamma (t/sum a)) (t/sum (t/lgamma a)))
          log-body (t/sum (t/mul (t/sub a 1.0) (t/log x)) event-dim)]
      (t/add log-norm log-body))

    :student-t
    (let [x (as-tensor value)
          nu (as-tensor df)
          mu (as-tensor loc)
          sigma (as-tensor scale)
          z (t/div (t/sub x mu) sigma)
          half-nu (t/div nu 2.0)
          half-nu+1 (t/div (t/add nu 1.0) 2.0)]
      (t/sub
       (t/sub (t/sub (t/lgamma half-nu+1)
                     (t/lgamma half-nu))
              (t/log sigma))
       (t/add (t/mul 0.5 (t/log (t/mul nu Math/PI)))
              (t/mul half-nu+1 (t/log (t/add 1.0 (t/div (t/pow z 2.0) nu)))))))

    :chi2
    (log-prob (gamma (if (number? df) (/ (double df) 2.0) (t/div df 2.0)) 0.5) value)

    :fisher-f
    (let [x (as-tensor value)
          d1 (as-tensor df1)
          d2 (as-tensor df2)
          a (t/div d1 2.0)
          b (t/div d2 2.0)
          log-beta (t/sub (t/add (t/lgamma a) (t/lgamma b)) (t/lgamma (t/add a b)))]
      (t/sub
       (t/add (t/mul a (t/log (t/div d1 d2)))
              (t/mul (t/sub a 1.0) (t/log x)))
       (t/add (t/mul (t/add a b) (t/log (t/add 1.0 (t/mul (t/div d1 d2) x))))
              log-beta)))

    :weibull
    (let [x (as-tensor value)
          lam (as-tensor scale)
          k (as-tensor concentration)
          z (t/div x lam)]
      (t/add (t/sub (t/log (t/div k lam))
                    (t/pow z k))
             (t/mul (t/sub k 1.0) (t/log z))))

    :pareto
    (let [x (as-tensor value)
          xm (as-tensor scale)
          a (as-tensor alpha)]
      (t/sub (t/add (t/log a)
                    (t/mul a (t/log xm)))
             (t/mul (t/add a 1.0) (t/log x))))

    :logistic
    (let [x (as-tensor value)
          loc-t (as-tensor loc)
          scale-t (as-tensor scale)
          z (t/div (t/sub x loc-t) scale-t)]
      (t/sub (t/sub (t/neg z)
                    (t/mul 2.0 (t/log1p (t/exp (t/neg z)))))
             (t/log scale-t)))

    :negative-binomial
    (let [x (as-tensor value)
          r (as-tensor total-count)
          p (t/clamp (as-tensor probs) eps (- 1.0 eps))]
      (t/add
       (t/sub (t/sub (t/lgamma (t/add x r))
                     (t/lgamma r))
              (t/lgamma (t/add x 1.0)))
       (t/add (t/mul r (t/log p))
              (t/mul x (t/log (t/sub 1.0 p))))))

    (throw (IllegalArgumentException. (str "Unsupported distribution for log-prob: " dist)))))

(defn mean
  [{:keys [dist loc scale probs low high rate total-count concentration concentration1 concentration0
           df df2 alpha]}]
  (case dist
    :normal loc
    :log-normal (if (and (number? loc) (number? scale))
                  (Math/exp (+ (double loc) (* 0.5 (double scale) (double scale))))
                  (t/exp (t/add loc (t/mul 0.5 (t/pow scale 2.0)))))
    :bernoulli probs
    :categorical nil
    :binomial (if (and (number? total-count) (number? probs))
                (* (double total-count) (double probs))
                (t/mul total-count probs))
    :uniform (/ (+ (double low) (double high)) 2.0)
    :poisson rate
    :exponential (if (number? rate) (/ 1.0 (double rate)) (t/div 1.0 rate))
    :cauchy nil
    :geometric (if (number? probs) (/ 1.0 (double probs)) (t/div 1.0 probs))
    :gumbel (if (and (number? loc) (number? scale))
              (+ (double loc) (* euler-gamma (double scale)))
              (t/add loc (t/mul euler-gamma scale)))
    :laplace loc
    :gamma (if (and (number? concentration) (number? rate))
             (/ (double concentration) (double rate))
             (t/div concentration rate))
    :beta (if (and (number? concentration1) (number? concentration0))
            (/ (double concentration1) (+ (double concentration1) (double concentration0)))
            (t/div concentration1 (t/add concentration1 concentration0)))
    :dirichlet (if (number? concentration)
                 1.0
                 (let [a (as-tensor concentration)]
                   (t/div a (t/sum a))))
    :student-t loc
    :chi2 df
    :fisher-f (if (and (number? df2) (> (double df2) 2.0))
                (/ (double df2) (- (double df2) 2.0))
                nil)
    :weibull (if (and (number? scale) (number? concentration))
               (* (double scale)
                  (Math/exp (Gamma/logGamma (+ 1.0 (/ 1.0 (double concentration))))))
               (let [k (as-tensor concentration)
                     g (t/lgamma (t/add 1.0 (t/div 1.0 k)))]
                 (t/mul scale (t/exp g))))
    :pareto (if (and (number? alpha) (> (double alpha) 1.0))
              (/ (* (double alpha) (double scale)) (- (double alpha) 1.0))
              nil)
    :logistic loc
    :negative-binomial (if (and (number? total-count) (number? probs))
                         (* (double total-count) (/ (- 1.0 (double probs)) (double probs)))
                         (t/mul total-count (t/div (t/sub 1.0 probs) probs)))
    (throw (IllegalArgumentException. (str "Unsupported distribution for mean: " dist)))))

(defn variance
  [{:keys [dist loc scale probs low high rate total-count concentration concentration1 concentration0
           df df1 df2 alpha]}]
  (case dist
    :normal (if (number? scale) (* (double scale) (double scale)) (t/pow scale 2.0))
    :log-normal (if (and (number? loc) (number? scale))
                  (let [s2 (* (double scale) (double scale))]
                    (* (- (Math/exp s2) 1.0)
                       (Math/exp (+ (* 2.0 (double loc)) s2))))
                  (let [s2 (t/pow scale 2.0)]
                    (t/mul (t/sub (t/exp s2) 1.0)
                           (t/exp (t/add (t/mul 2.0 loc) s2)))))
    :bernoulli (if (number? probs) (* probs (- 1.0 probs)) (t/mul probs (t/sub 1.0 probs)))
    :categorical nil
    :binomial (if (and (number? total-count) (number? probs))
                (* total-count probs (- 1.0 probs))
                (t/mul total-count (t/mul probs (t/sub 1.0 probs))))
    :uniform (let [w (- (double high) (double low))] (/ (* w w) 12.0))
    :poisson rate
    :exponential (if (number? rate) (/ 1.0 (* (double rate) (double rate))) (t/div 1.0 (t/pow rate 2.0)))
    :cauchy nil
    :geometric (if (number? probs)
                 (/ (- 1.0 probs) (* probs probs))
                 (t/div (t/sub 1.0 probs) (t/pow probs 2.0)))
    :gumbel (if (number? scale)
              (* (/ (* Math/PI Math/PI) 6.0) scale scale)
              (t/mul (/ (* Math/PI Math/PI) 6.0) (t/pow scale 2.0)))
    :laplace (if (number? scale) (* 2.0 scale scale) (t/mul 2.0 (t/pow scale 2.0)))
    :gamma (if (and (number? concentration) (number? rate))
             (/ (double concentration) (* (double rate) (double rate)))
             (t/div concentration (t/pow rate 2.0)))
    :beta (if (and (number? concentration1) (number? concentration0))
            (let [a (double concentration1)
                  b (double concentration0)
                  ab (+ a b)]
              (/ (* a b) (* ab ab (+ ab 1.0))))
            (let [ab (t/add concentration1 concentration0)]
              (t/div (t/mul concentration1 concentration0)
                     (t/mul (t/pow ab 2.0) (t/add ab 1.0)))))
    :dirichlet (if (number? concentration)
                 0.0
                 (let [a (as-tensor concentration)
                       a0 (t/sum a)
                       numer (t/mul a (t/sub a0 a))
                       denom (t/mul (t/pow a0 2.0) (t/add a0 1.0))]
                   (t/div numer denom)))
    :student-t (if (and (number? df) (> (double df) 2.0))
                 (* (/ (double df) (- (double df) 2.0)) (double scale) (double scale))
                 nil)
    :chi2 (* 2.0 (double df))
    :fisher-f (if (and (number? df1) (number? df2) (> (double df2) 4.0))
                (let [n (double df1)
                      m (double df2)]
                  (/ (* 2.0 m m (+ n m -2.0))
                     (* n (- m 2.0) (- m 2.0) (- m 4.0))))
                nil)
    :weibull (if (and (number? scale) (number? concentration))
               (let [k (double concentration)
                     l (double scale)
                     g1 (Math/exp (Gamma/logGamma (+ 1.0 (/ 1.0 k))))
                     g2 (Math/exp (Gamma/logGamma (+ 1.0 (/ 2.0 k))))]
                 (* l l (- g2 (* g1 g1))))
               nil)
    :pareto (if (and (number? alpha) (> (double alpha) 2.0))
              (let [a (double alpha)
                    xm (double scale)]
                (/ (* a xm xm)
                   (* (- a 1.0) (- a 1.0) (- a 2.0))))
              nil)
    :logistic (if (number? scale)
                (* (/ (* Math/PI Math/PI) 3.0) (double scale) (double scale))
                (t/mul (/ (* Math/PI Math/PI) 3.0) (t/pow scale 2.0)))
    :negative-binomial (if (and (number? total-count) (number? probs))
                         (let [r (double total-count)
                               p (double probs)]
                           (/ (* r (- 1.0 p)) (* p p)))
                         (t/div (t/mul total-count (t/sub 1.0 probs)) (t/pow probs 2.0)))
    (throw (IllegalArgumentException. (str "Unsupported distribution for variance: " dist)))))
