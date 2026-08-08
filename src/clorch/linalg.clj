(ns clorch.linalg
  "Linear algebra namespace with PyTorch-style function names.

   User-facing API should prefer this namespace over `clorch.torch`'s
   compatibility wrappers like `linalg-*`."
  (:refer-clojure :exclude [cond])
  (:require [clorch.torch :as t]))

(defn norm [x] (t/linalg-norm x))
(defn vector-norm [x] (t/linalg-vector-norm x))
(defn matrix-norm [x] (t/linalg-matrix-norm x))
(defn inv [x] (t/linalg-inv x))
(defn det [x] (t/linalg-det x))
(defn slogdet [x] (t/linalg-slogdet x))

(defn qr
  ([x] (t/linalg-qr x))
  ([x mode] (t/linalg-qr x mode)))

(defn cholesky
  ([x] (t/linalg-cholesky x))
  ([x upper?] (t/linalg-cholesky x upper?)))

(defn cholesky-inverse
  ([chol] (t/cholesky-inverse chol))
  ([chol upper?] (t/cholesky-inverse chol upper?)))

(defn cholesky-solve
  ([b chol] (t/cholesky-solve b chol))
  ([b chol upper?] (t/cholesky-solve b chol upper?)))

(defn solve
  ([a b] (t/linalg-solve a b))
  ([a b left?] (t/linalg-solve a b left?)))

(defn pinv
  ([x] (t/linalg-pinv x))
  ([x rcond] (t/linalg-pinv x rcond)))

(defn matrix-power [x n] (t/linalg-matrix-power x n))

(defn matrix-exp [x]
  (t/matrix-exp x))

(defn diagonal
  ([x] (t/diagonal x))
  ([x offset dim1 dim2] (t/diagonal x offset dim1 dim2)))

(defn cross
  ([a b] (t/cross a b))
  ([a b dim] (t/cross a b dim)))

(defn svd
  ([x] (t/linalg-svd x))
  ([x opts] (t/linalg-svd x opts)))

(defn svdvals
  ([x] (t/linalg-svdvals x))
  ([x opts] (t/linalg-svdvals x opts)))

(defn eig [x] (t/linalg-eig x))
(defn eigvals [x] (t/linalg-eigvals x))

(defn eigh
  ([x] (t/linalg-eigh x))
  ([x opts] (t/linalg-eigh x opts)))

(defn eigvalsh
  ([x] (t/linalg-eigvalsh x))
  ([x opts] (t/linalg-eigvalsh x opts)))

(defn lu
  ([x] (t/linalg-lu x))
  ([x opts] (t/linalg-lu x opts)))

(defn lu-factor
  ([x] (t/linalg-lu-factor x))
  ([x opts] (t/linalg-lu-factor x opts)))

(defn lu-solve
  ([lu-factorized pivots b] (t/linalg-lu-solve lu-factorized pivots b))
  ([lu-factorized pivots b opts] (t/linalg-lu-solve lu-factorized pivots b opts)))

(defn ldl-factor
  ([x] (t/linalg-ldl-factor x))
  ([x opts] (t/linalg-ldl-factor x opts)))

(defn ldl-solve
  ([ldl pivots b] (t/linalg-ldl-solve ldl pivots b))
  ([ldl pivots b opts] (t/linalg-ldl-solve ldl pivots b opts)))

(defn solve-triangular
  ([a b upper] (t/linalg-solve-triangular a b upper))
  ([a b upper opts] (t/linalg-solve-triangular a b upper opts)))

(defn triangular-solve
  ([a b upper] (solve-triangular a b upper))
  ([a b upper opts] (solve-triangular a b upper opts)))

(defn matrix-rank
  ([x] (t/linalg-matrix-rank x))
  ([x tol-or-opts] (t/linalg-matrix-rank x tol-or-opts))
  ([x tol opts] (t/linalg-matrix-rank x tol opts)))

(defn cond
  ([x] (t/linalg-cond x))
  ([x ord] (t/linalg-cond x ord)))

(defn lstsq
  ([a b] (t/linalg-lstsq a b))
  ([a b opts] (t/linalg-lstsq a b opts)))

(defn least-squares
  ([a b] (lstsq a b))
  ([a b opts] (lstsq a b opts)))

(defn solve-ex
  ([a b] (t/linalg-solve-ex a b))
  ([a b opts] (t/linalg-solve-ex a b opts)))

(defn inv-ex
  ([x] (t/linalg-inv-ex x))
  ([x opts] (t/linalg-inv-ex x opts)))

(defn cholesky-ex
  ([x] (t/linalg-cholesky-ex x))
  ([x opts] (t/linalg-cholesky-ex x opts)))

(defn lu-factor-ex
  ([x] (t/linalg-lu-factor-ex x))
  ([x opts] (t/linalg-lu-factor-ex x opts)))

(defn ldl-factor-ex
  ([x] (t/linalg-ldl-factor-ex x))
  ([x opts] (t/linalg-ldl-factor-ex x opts)))

(defn tensorinv
  ([x] (t/linalg-tensorinv x))
  ([x ind] (t/linalg-tensorinv x ind)))

(defn tensorsolve
  ([a b] (t/linalg-tensorsolve a b))
  ([a b dims] (t/linalg-tensorsolve a b dims)))

(defn multi-dot [tensors]
  (t/linalg-multi-dot tensors))

(defn vecdot
  ([a b] (t/linalg-vecdot a b))
  ([a b dim] (t/linalg-vecdot a b dim)))

(defn vander
  ([x] (t/linalg-vander x))
  ([x n] (t/linalg-vander x n)))

(defn householder-product [input tau]
  (t/linalg-householder-product input tau))

(defn lu-unpack
  ([lu-data pivots] (t/lu-unpack lu-data pivots))
  ([lu-data pivots opts] (t/lu-unpack lu-data pivots opts)))
