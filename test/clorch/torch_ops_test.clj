(ns clorch.torch-ops-test
  (:require [clorch.torch :as t]
            [clojure.test :refer [deftest is testing]]))

(defn near? [a b]
  (let [eps 1e-4]
    (< (Math/abs (- (float a) (float b))) eps)))

(defn tensor-near? [t1 t2]
  (let [v1 (mapv t/item-float (t/tseq (t/reshape t1 [-1])))
        v2 (mapv t/item-float (t/tseq (t/reshape t2 [-1])))]
    (every? (fn [[a b]] (near? a b)) (map vector v1 v2))))

(deftest factory-methods-test
  (t/with-torch
    (testing "zeros"
      (let [z (t/zeros [2 3])]
        (is (= [2 3] (t/size z)))
        (is (every? #(= 0.0 %) (mapv t/item-float (t/tseq (t/reshape z [-1])))))))

    (testing "ones"
      (let [o (t/ones [2 3])]
        (is (= [2 3] (t/size o)))
        (is (every? #(= 1.0 %) (mapv t/item-float (t/tseq (t/reshape o [-1])))))))

    (testing "eye"
      (let [e (t/eye 3)]
        (is (= [3 3] (t/size e)))
        (is (= [1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0]
               (mapv t/item-float (t/tseq (t/reshape e [-1])))))))

    (testing "arange"
      (is (= [0.0 1.0 2.0] (mapv t/item-float (t/tseq (t/arange 3)))))
      (is (= [1.0 2.0] (mapv t/item-float (t/tseq (t/arange 1 3)))))
      (is (= [1.0 3.0] (mapv t/item-float (t/tseq (t/arange 1 5 2))))))

    (testing "randn"
      (let [r (t/randn [100 100])]
        (is (= [100 100] (t/size r)))
        (let [m (t/item-float (t/mean r))]
          (is (< (Math/abs m) 0.1))))) ;; loose mean check

    (testing "rand-int"
      (let [r (t/rand-int 0 10 [100])]
        (is (every? #(and (>= % 0) (< % 10)) (mapv t/item-float (t/tseq r))))))

    (testing "full"
      (let [f (t/full [2 2] 3.14)]
        (is (= [2 2] (t/size f)))
        (is (near? (t/item-float (t/ix f 0 0)) 3.14))))

    (testing "linspace"
      (let [l (t/linspace 0 10 5)]
        (is (= [5] (t/size l)))
        (is (= [0.0 2.5 5.0 7.5 10.0] (mapv t/item-float (t/tseq l))))))

    (testing "logspace"
      (let [l (t/logspace 0 2 3 :base 10)]
        (is (= [3] (t/size l)))
        (is (= [1.0 10.0 100.0] (mapv t/item-float (t/tseq l))))))))

(deftest math-ops-test
  (t/with-torch
    (let [a (t/tensor [1 2 3])
          b (t/tensor [4 5 6])]
      (testing "add"
        (is (= [5.0 7.0 9.0] (mapv t/item-float (t/tseq (t/add a b))))))
      (testing "sub"
        (is (= [-3.0 -3.0 -3.0] (mapv t/item-float (t/tseq (t/sub a b))))))
      (testing "mul"
        (is (= [4.0 10.0 18.0] (mapv t/item-float (t/tseq (t/mul a b))))))
      (testing "div"
        (is (tensor-near? (t/div a b) (t/tensor [0.25 0.4 0.5]))))
      (testing "lerp"
        (is (tensor-near? (t/lerp a b 0.5) (t/tensor [2.5 3.5 4.5]))))
      (testing "addcmul"
        (is (tensor-near? (t/addcmul a a b :value 2.0) (t/tensor [9.0 22.0 39.0]))))
      (testing "addcdiv"
        (is (tensor-near? (t/addcdiv a b a :value 2.0) (t/tensor [9.0 7.0 7.0]))))
      (testing "polygamma"
        (is (tensor-near? (t/polygamma 1 a) (t/tensor [1.644934 0.644934 0.394934]))))
      (testing "bitwise ops"
        (let [i1 (t/tensor [1 2 3] {:dtype :int64})
              i2 (t/tensor [3 2 1] {:dtype :int64})]
          (is (= [1.0 2.0 1.0] (mapv t/item-float (t/tseq (t/bitwise-and i1 i2)))))
          (is (= [3.0 2.0 3.0] (mapv t/item-float (t/tseq (t/bitwise-or i1 i2)))))
          (is (= [2.0 0.0 2.0] (mapv t/item-float (t/tseq (t/bitwise-xor i1 i2)))))
          (is (= [-2.0 -3.0 -4.0] (mapv t/item-float (t/tseq (t/bitwise-not i1)))))
          (is (= [2.0 4.0 6.0] (mapv t/item-float (t/tseq (t/bitwise-left-shift i1 1)))))
          (is (= [0.0 1.0 1.0] (mapv t/item-float (t/tseq (t/bitwise-right-shift i1 1)))))))
      (testing "pow"
        (is (= [1.0 4.0 9.0] (mapv t/item-float (t/tseq (t/pow a 2))))))
      (testing "sqrt"
        (is (near? (t/item-float (t/ix (t/sqrt (t/tensor [4.0])) 0)) 2.0)))
      (testing "rsqrt"
        (is (near? (t/item-float (t/ix (t/rsqrt (t/tensor [4.0])) 0)) 0.5)))
      (testing "reciprocal"
        (is (near? (t/item-float (t/ix (t/reciprocal (t/tensor [2.0])) 0)) 0.5)))
      (testing "sin/cos"
        (is (near? (t/item-float (t/ix (t/sin (t/tensor [0.0])) 0)) 0.0))
        (is (near? (t/item-float (t/ix (t/cos (t/tensor [0.0])) 0)) 1.0)))
      (testing "logical ops"
        (let [b1 (t/to (t/tensor [1 0 1]) :bool)
              b2 (t/to (t/tensor [1 1 0]) :bool)]
          (is (= [1.0 0.0 0.0] (mapv t/item-float (t/tseq (t/to (t/logical-and b1 b2) :float32)))))
          (is (= [1.0 1.0 1.0] (mapv t/item-float (t/tseq (t/to (t/logical-or b1 b2) :float32)))))
          (is (= [0.0 1.0 1.0] (mapv t/item-float (t/tseq (t/to (t/logical-xor b1 b2) :float32)))))
          (is (= [0.0 1.0 0.0] (mapv t/item-float (t/tseq (t/to (t/logical-not b1) :float32)))))))
      (testing "type-as"
        (let [float-t (t/tensor [1.0 2.0])
              int-t (t/tensor [3 4] {:dtype :int64})
              converted (t/type-as float-t int-t)]
          (is (= "Long" (.toString (.scalar_type (t/->tensor converted)))))))
      (testing "max/minimum"
        (let [x (t/tensor [1.0 5.0])
              y (t/tensor [2.0 4.0])]
          (is (= [2.0 5.0] (mapv t/item-float (t/tseq (t/maximum x y)))))
          (is (= [1.0 4.0] (mapv t/item-float (t/tseq (t/minimum x y)))))
          (is (= [2.0 5.0] (mapv t/item-float (t/tseq (t/fmax x y)))))
          (is (= [1.0 4.0] (mapv t/item-float (t/tseq (t/fmin x y)))))))
      (testing "matmul"
        (let [m1 (t/tensor [[1 2] [3 4]])
              m2 (t/tensor [[5 6] [7 8]])
              res (t/matmul m1 m2)]
          (is (= [[19.0 22.0] [43.0 50.0]]
                 (mapv (fn [row] (mapv t/item-float (t/tseq row))) (t/tseq res)))))))))

(deftest reduction-test
  (t/with-torch
    (let [x (t/tensor [[1 2 3] [4 5 6]])]
      (testing "sum"
        (is (= 21.0 (t/item-float (t/sum x))))
        (is (= [5.0 7.0 9.0] (mapv t/item-float (t/tseq (t/sum x 0)))))
        (is (= [6.0 15.0] (mapv t/item-float (t/tseq (t/sum x 1))))))

      (testing "mean"
        (is (= 3.5 (t/item-float (t/mean x))))
        (is (= [2.5 3.5 4.5] (mapv t/item-float (t/tseq (t/mean x 0))))))

      (testing "argmax/argmin"
        (is (= 5.0 (t/item-float (t/argmax x))))
        (is (= 0.0 (t/item-float (t/argmin x))))
        (is (= [1.0 1.0 1.0] (mapv t/item-float (t/tseq (t/argmax x 0)))))
        (is (= [0.0 0.0 0.0] (mapv t/item-float (t/tseq (t/argmin x 0)))))
        (is (= [2.0 2.0] (mapv t/item-float (t/tseq (t/argmax x 1)))))
        (is (= [0.0 0.0] (mapv t/item-float (t/tseq (t/argmin x 1)))))))))

(deftest transform-test
  (t/with-torch
    (let [x (t/arange 6)]
      (testing "clone"
        (let [cloned (t/clone x)]
          (is (= (t/size x) (t/size cloned)))
          (is (tensor-near? x cloned))))
      (testing "permute"
        (let [p-test (t/reshape x [2 3])
              permuted (t/permute p-test [1 0])]
          (is (= [3 2] (t/size permuted)))))
      (testing "empty-strided"
        (let [es (t/empty-strided [2 3] [3 1] {:dtype :float32})]
          (is (= [2 3] (t/size es)))
          (is (= "Float" (.toString (.scalar_type (t/->tensor es)))))))
      (testing "split-with-sizes"
        (let [splits (t/split-with-sizes x [2 4] 0)]
          (is (= 2 (count splits)))
          (is (= [2] (t/size (first splits))))
          (is (= [4] (t/size (second splits))))))
      (testing "slice"
        (let [sliced (t/slice x 0 1 4 1)]
          (is (= [3] (t/size sliced)))
          (is (= [1.0 2.0 3.0] (mapv t/item-float (t/tseq sliced))))))
      (testing "reshape"
        (let [r (t/reshape x [2 3])]
          (is (= [2 3] (t/size r)))
          (is (= [[0.0 1.0 2.0] [3.0 4.0 5.0]]
                 (mapv (fn [row] (mapv t/item-float (t/tseq row))) (t/tseq r))))))

      (testing "transpose"
        (let [r (t/reshape x [2 3])
              tr (t/transpose r 0 1)]
          (is (= [3 2] (t/size tr)))
          (is (= [[0.0 3.0] [1.0 4.0] [2.0 5.0]]
                 (mapv (fn [row] (mapv t/item-float (t/tseq row))) (t/tseq tr))))))

      (testing "stack/cat"
        (let [s (t/stack [x x] 0)]
          (is (= [2 6] (t/size s))))
        (let [c (t/cat [x x] 0)]
          (is (= [12] (t/size c))))
        (let [c1 (t/cat [(t/reshape x [2 3]) (t/reshape x [2 3])] 1)]
          (is (= [2 6] (t/size c1))))))))

(deftest logic-ops-test
  (t/with-torch
    (let [x (t/tensor [1 3 2 5 4])]
      (testing "topk"
        (let [[v i] (t/topk x 3)]
          (is (= [5.0 4.0 3.0] (mapv t/item-float (t/tseq v))))
          (is (= [3.0 4.0 1.0] (mapv t/item-float (t/tseq i))))))

      (testing "sort"
        (let [[v _] (t/sort x)]
          (is (= [1.0 2.0 3.0 4.0 5.0] (mapv t/item-float (t/tseq v))))))

      (testing "where"
        (let [condition (t/eq (t/tensor [1 0 1 0 1]) 1)
              res (t/where condition x (t/zeros [5]))]
          (is (= [1.0 0.0 2.0 0.0 4.0] (mapv t/item-float (t/tseq res))))))

      (testing "nonzero"
        (let [res (t/nonzero (t/tensor [1 0 2 0]))]
          (is (= [[0.0] [2.0]] (mapv (fn [row] (mapv t/item-float (t/tseq row))) (t/tseq res))))))

      (testing "all/any"
        (is (= 1.0 (t/item-float (t/all (t/tensor [1 1 1])))))
        (is (= 0.0 (t/item-float (t/all (t/tensor [1 0 1])))))
        (is (= 1.0 (t/item-float (t/any (t/tensor [0 1 0])))))
        (is (= 0.0 (t/item-float (t/any (t/tensor [0 0 0])))))))

    (let [x (t/tensor [1.0 2.0 3.0 4.0 5.0])]
      (testing "clamp"
        (is (tensor-near? (t/clamp x 2.0 4.0) (t/tensor [2.0 2.0 3.0 4.0 4.0]))))

      (testing "cumsum"
        (is (tensor-near? (t/cumsum x 0) (t/tensor [1.0 3.0 6.0 10.0 15.0]))))

      (testing "gt"
        (is (tensor-near? (t/to (t/gt x 3.0) :float32) (t/tensor [0.0 0.0 0.0 1.0 1.0])))))

    (testing "repeat"
      (let [x (t/tensor [1 2])
            res (t/repeat x [2 2])]
        (is (= [2 4] (t/size res)))
        (is (= [1.0 2.0 1.0 2.0 1.0 2.0 1.0 2.0] (mapv t/item-float (t/tseq (t/reshape res [-1])))))))

    (testing "softmax"
      (let [x (t/tensor [1.0 1.0])]
        (is (tensor-near? (t/softmax x 0) (t/tensor [0.5 0.5])))))

    (testing "gather"
      (let [t (t/tensor [[1 2] [3 4]])
            index (t/tensor [[0 0] [1 0]] {:dtype :int64})
            res (t/gather t 1 index)]
        (is (= [[1.0 1.0] [4.0 3.0]] (mapv (fn [row] (mapv t/item-float (t/tseq row))) (t/tseq res))))))

    (testing "top-p sampling"
      (let [logits (t/tensor [[1.0 2.0 3.0 4.0 5.0]])
            res (t/top-p logits 0.9)]
        (is (= [1 1] (t/size res)))
        (is (some? (t/item-float res)))))))

(deftest long-tail-batch1-ops-test
  (t/with-torch
    (testing "utility reductions, ordering, and predicates"
      (let [x (t/tensor [[1.0 -2.0 3.0]
                         [4.0 0.0 -6.0]])]
        (is (= 5.0 (t/item-float (t/count-nonzero x))))
        (is (= [2 3] (t/size (t/argsort x 1 false))))
        (is (= [3 2] (t/size (t/argwhere (t/gt x 0.0)))))
        (is (= [3.0 4.0] (mapv t/item-float (t/tseq (t/amax x 1)))))
        (is (= [-2.0 -6.0] (mapv t/item-float (t/tseq (t/amin x 1)))))
        (is (= [2.0 2.0 3.0 4.0 2.0 6.0]
               (mapv t/item-float (t/tseq (t/reshape (t/maximum (t/abs x) (t/tensor [2.0 2.0 2.0])) [-1]))))))

      (let [v (t/tensor [0.0 (/ 1.0 0.0) (/ -1.0 0.0) (/ 0.0 0.0)])
            cleaned (t/nan-to-num v {:nan 7.0 :posinf 9.0 :neginf -9.0})]
        (is (= [0.0 9.0 -9.0 7.0] (mapv t/item-float (t/tseq cleaned))))
        (is (= [0.0 1.0 1.0 0.0] (mapv t/item-float (t/tseq (t/to (t/isinf v) :float32)))))
        (is (= [1.0 0.0 0.0] (mapv t/item-float (t/tseq (t/to (t/isfinite (t/tensor [1.0 (/ 1.0 0.0) (/ 0.0 0.0)])) :float32)))))))

    (testing "shape/stack/move ops"
      (let [a (t/tensor [[1.0 2.0] [3.0 4.0]])
            b (t/tensor [[5.0 6.0] [7.0 8.0]])]
        (is (= [4 2] (t/size (t/vstack [a b]))))
        (is (= [2 4] (t/size (t/hstack [a b]))))
        (is (= [2 2 2] (t/size (t/dstack [a b]))))
        (is (= [4 2] (t/size (t/row-stack [(t/tensor [1.0 2.0]) (t/tensor [3.0 4.0]) (t/tensor [5.0 6.0]) (t/tensor [7.0 8.0])]))))
        (is (= [2 2] (t/size (t/column-stack [(t/tensor [1.0 2.0]) (t/tensor [3.0 4.0])]))))
        (is (= [4 2] (t/size (t/tile (t/tensor [[1.0] [2.0]]) [2 2]))))
        (is (= [2] (t/size (t/diagonal a))))
        (is (= [4 4] (t/size (t/diagflat (t/tensor [1.0 2.0 3.0 4.0])))))
        (is (= [2 2] (t/size (t/swapaxes a 0 1))))
        (is (= [3 2 4] (t/size (t/movedim (t/zeros [2 3 4]) 0 1))))
        (is (= [2 2] (t/size (t/take-along-dim a (t/tensor [[1 0] [0 1]] {:dtype :int64}) 1))))))

    (testing "kron/tensordot/vdot/matrix-exp and special functions"
      (let [u (t/tensor [1.0 2.0])
            v (t/tensor [3.0 4.0])]
        (is (= [4] (t/size (t/kron u v))))
        (is (near? 11.0 (t/item-float (t/tensordot u v 1))))
        (is (near? 11.0 (t/item-float (t/vdot u v)))))
      (let [z (t/tensor [0.0 1.0 2.0])]
        (is (tensor-near? (t/sinc z) (t/tensor [1.0 0.0 0.0])))
        (is (tensor-near? (t/xlog1py (t/tensor [0.0 1.0 1.0]) (t/tensor [0.0 1.7182818 6.389056]))
                          (t/tensor [0.0 1.0 2.0])))
        (is (= [0.0 0.0 0.0] (mapv t/item-float (t/tseq (t/xlogy (t/tensor [0.0 0.0 0.0]) (t/tensor [1.0 2.0 3.0]))))))
        (is (= [3] (t/size (t/i0 z))))
        (is (= [3] (t/size (t/i1 z))))
        (is (= [2] (t/size (t/zeta (t/tensor [2.0 3.0]) (t/tensor [2.0 2.0]))))))
      (let [m (t/tensor [[0.0 0.0] [0.0 0.0]])]
        (is (tensor-near? (t/matrix-exp m) (t/eye 2)))))))

(deftest fft-batch2-ops-test
  (t/with-torch
    (let [x (t/tensor [1.0 2.0 3.0 4.0])
          xf (t/fft x)
          xfi (t/ifft xf)
          xr (t/rfft x)
          xri (t/irfft xr 4)]
      (is (= [4] (t/size xf)))
      (is (= [4] (t/size xfi)))
      (is (= [3] (t/size xr)))
      (is (= [4] (t/size xri)))
      (is (t/allclose x xri)))))

(deftest fft-and-stats-batch3-ops-test
  (t/with-torch
    (let [x2d (t/reshape (t/arange 16) [4 4])
          f2 (t/fft2 x2d)
          i2 (t/ifft2 f2)
          rf2 (t/rfft2 x2d)
          ir2 (t/irfft2 rf2 [4 4])
          fnn (t/fftn x2d [4 4] [0 1])
          ifn (t/ifftn fnn [4 4] [0 1])
          rfn (t/rfftn x2d [4 4] [0 1])
          irn (t/irfftn rfn [4 4] [0 1])]
      (is (= [4 4] (t/size f2)))
      (is (= [4 4] (t/size i2)))
      (is (= [4 3] (t/size rf2)))
      (is (= [4 4] (t/size ir2)))
      (is (= [4 4] (t/size fnn)))
      (is (= [4 4] (t/size ifn)))
      (is (= [4 3] (t/size rfn)))
      (is (= [4 4] (t/size irn))))

    (let [x (t/tensor [0.0 1.0 2.0 3.0])
          shifted (t/fftshift x)
          unshifted (t/ifftshift shifted)]
      (is (t/allclose x unshifted))
      (is (= [4] (t/size (t/fftfreq 4))))
      (is (= [3] (t/size (t/rfftfreq 4)))))

    (let [x (t/tensor [1.0 3.0 6.0 10.0])
          d (t/diff x)
          g (t/gradient x)]
      (is (= [3] (t/size d)))
      (is (= 1 (count g)))
      (is (= [4] (t/size (first g)))))

    (let [x (t/tensor [0.0 1.0 1.0 2.0 3.0 4.0])
          histc (t/histc x)
          [h e] (t/histogram x (t/tensor [0.0 1.0 2.0 3.0 4.0]))
          q (t/quantile x (t/tensor [0.25 0.5 0.75]))
          nq (t/nanquantile (t/tensor [0.0 1.0 (/ 0.0 0.0) 3.0]) (t/tensor [0.5]))]
      (is (= [100] (t/size histc)))
      (is (= [4] (t/size h)))
      (is (= [5] (t/size e)))
      (is (= [3] (t/size q)))
      (is (= [1] (t/size nq))))))

(deftest distance-search-indexreduce-batch4-ops-test
  (t/with-torch
    (let [x1 (t/tensor [[[0.0 0.0] [1.0 0.0]]])
          x2 (t/tensor [[[0.0 1.0] [1.0 1.0]]])
          d (t/cdist x1 x2)
          p (t/pdist (t/tensor [[0.0 0.0] [1.0 0.0] [1.0 1.0]]))]
      (is (= [1 2 2] (t/size d)))
      (is (= [3] (t/size p))))

    (let [sorted (t/tensor [1.0 3.0 5.0 7.0])
          values (t/tensor [0.0 2.0 5.0 9.0])
          ss (t/searchsorted sorted values)
          bs (t/bucketize values sorted)]
      (is (= [4] (t/size ss)))
      (is (= [4] (t/size bs)))
      (is (= [0.0 1.0 2.0 4.0] (mapv t/item-float (t/tseq ss)))))

    (let [bc (t/bincount (t/tensor [0 1 1 3 4 4 4] {:dtype :int64}))]
      (is (= [5] (t/size bc)))
      (is (= [1.0 2.0 0.0 1.0 3.0] (mapv t/item-float (t/tseq bc)))))

    (let [input (t/tensor [[1.0 1.0 1.0] [1.0 1.0 1.0]])
          index (t/tensor [[0 1 0] [1 0 1]] {:dtype :int64})
          src (t/tensor [[2.0 3.0 4.0] [5.0 6.0 7.0]])
          sr (t/scatter-reduce input 1 index src "sum" true)]
      (is (= [2 3] (t/size sr))))

    (let [input (t/tensor [[1.0 2.0] [3.0 4.0] [5.0 6.0]])
          idx (t/tensor [0 2] {:dtype :int64})
          src (t/tensor [[10.0 20.0] [30.0 40.0]])
          ir (t/index-reduce input 0 idx src "amax" true)]
      (is (= [3 2] (t/size ir))))))
