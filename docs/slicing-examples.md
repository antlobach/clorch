---
layout: default
title: Slicing Examples
---

# Clorch Tensor Slicing Examples

This page contains runnable code examples for all tensor slicing operations in Clorch.

## Quick Links

- [Basic Indexing](#basic-indexing)
- [Slicing](#slicing)
- [Negative Step (Reverse)](#negative-step-slicing)
- [Ellipsis](#ellipsis)
- [Advanced Indexing](#advanced-indexing)
- [Real-World Use Cases](#real-world-use-cases)

---

## Basic Indexing

### 1D Tensor

```clojure
(require '[clorch.torch :as torch])

(def x (torch/tensor [10 11 12 13 14]))

(torch/ix x 0)   ; → 10.0
(torch/ix x 1)   ; → 11.0
(torch/ix x -1)  ; → 14.0 (last)
(torch/ix x -2)  ; → 13.0 (second to last)
```

### 2D Tensor

```clojure
(def x (torch/tensor [[1 2 3]
                     [4 5 6]
                     [7 8 9]]))

(torch/ix x 0)        ; → [1.0, 2.0, 3.0] (first row)
(torch/ix x -1)       ; → [7.0, 8.0, 9.0] (last row)
(torch/ix x 0 0)      ; → 1.0 (first element)
(torch/ix x 0 1)      ; → 2.0
(torch/ix x :_ 0)     ; → [1.0, 4.0, 7.0] (first column)
(torch/ix x :_ -1)    ; → [3.0, 6.0, 9.0] (last column)
```

---

## Slicing

### Basic Slices `[start stop]`

```clojure
(def y (torch/tensor (range 10)))  ; [0 1 2 3 4 5 6 7 8 9]

(torch/ix y [2 5])      ; → [2.0, 3.0, 4.0]
(torch/ix y [0 4])      ; → [0.0, 1.0, 2.0, 3.0]
(torch/ix y [6 10])     ; → [6.0, 7.0, 8.0, 9.0]
```

### Open-Ended `[start nil]` or `[nil stop]`

```clojure
(torch/ix y [0 5])     ; → [0.0, 1.0, 2.0, 3.0, 4.0]  (first 5)
(torch/ix y [5 nil])    ; → [5.0, 6.0, 7.0, 8.0, 9.0]  (last 5)
(torch/ix y [nil nil])  ; → [0.0, ..., 9.0]           (all)
```

### Step Slices `[start stop step]`

```clojure
(torch/ix y [nil nil 2])   ; → [0.0, 2.0, 4.0, 6.0, 8.0]  (every 2nd)
(torch/ix y [1 nil 2])      ; → [1.0, 3.0, 5.0, 7.0, 9.0]  (every 2nd, start at 1)
(torch/ix y [1 8 2])        ; → [1.0, 3.0, 5.0, 7.0]        (every 2nd, range 1-8)
(torch/ix y [nil nil 3])    ; → [0.0, 3.0, 6.0, 9.0]        (every 3rd)
```

---

## Negative Step Slicing

Full reversal and partial reversal use the same exclusive-stop semantics as Python slices.

### Full Reverse `[nil nil -1]`

```clojure
(def t (torch/tensor (range 10)))  ; [0 1 2 3 4 5 6 7 8 9]

;; Full reverse
(torch/ix t [nil nil -1])   ; → [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
```

### Reverse from Index `[start nil -1]`

```clojure
;; From index 5 to start
(torch/ix t [5 nil -1])    ; → [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]

;; From index 3 to start
(torch/ix t [3 nil -1])    ; → [3.0, 2.0, 1.0, 0.0]
```

### Partial Reverse `[start stop -1]`

```clojure
;; Indices 5, 4, 3 (exclusive of 2)
(torch/ix t [5 2 -1])     ; → [5.0, 4.0, 3.0]

;; Indices 7, 6, 5, 4 (exclusive of 3)
(torch/ix t [7 3 -1])     ; → [7.0, 6.0, 5.0, 4.0]
```

### Reverse Every Nth `[nil nil -N]`

```clojure
;; Reverse every 2nd element
(torch/ix t [nil nil -2])  ; → [9.0, 7.0, 5.0, 3.0, 1.0]

;; Reverse every 3rd element  
(torch/ix t [nil nil -3])  ; → [9.0, 6.0, 3.0, 0.0]

;; Reverse every 4th element
(torch/ix t [nil nil -4])  ; → [8.0, 4.0, 0.0]
```

### 2D Negative Step

```clojure
(def m (torch/tensor [[1 2 3 4]
                     [5 6 7 8]
                     [9 10 11 12]]))

;; Reverse rows
(torch/ix m [nil nil -1] :_)
; → [[9.0, 10.0, 11.0, 12.0],
;    [5.0, 6.0, 7.0, 8.0],
;    [1.0, 2.0, 3.0, 4.0]]

;; Reverse columns
(torch/ix m :_ [nil nil -1])
; → [[4.0, 3.0, 2.0, 1.0],
;    [8.0, 7.0, 6.0, 5.0],
;    [12.0, 11.0, 10.0, 9.0]]

;; Reverse both
(torch/ix m [nil nil -1] [nil nil -1])
; → [[12.0, 11.0, 10.0, 9.0],
;    [8.0, 7.0, 6.0, 5.0],
;    [4.0, 3.0, 2.0, 1.0]]
```

---

## Ellipsis

The ellipsis `...` automatically fills remaining dimensions.

```clojure
(def t3d (torch/reshape (torch/tensor (range 24)) [2 3 4]))

;; t[0, ...] - first element along first dim, all others
(torch/ix t3d 0 (quote ...))   ; → shape [3, 4]

;; t[..., 0] - all along first dims, first along last
(torch/ix t3d (quote ...) 0)   ; → shape [2, 3]

;; t[1, ..., 2] - specific indices
(torch/ix t3d 1 (quote ...) 2)  ; → shape [3]
```

---

## Advanced Indexing

### Integer Tensor (Fancy) Indexing

```clojure
(def m (torch/tensor [[1 2 3]
                     [4 5 6]
                     [7 8 9]
                     [10 11 12]]))

(def idx (torch/tensor [0 2 0 1] {:dtype :int64}))

;; Index with integer tensor
(torch/ix m idx)        ; → shape [4, 3]
(torch/ix m idx :_)    ; → shape [4, 3]
(torch/ix m :_ idx)    ; → shape [4, 4]
(torch/ix m idx idx)   ; → shape [4]
```

### Boolean Mask Indexing

```clojure
(def t (torch/tensor [1 2 3 4 5]))
(def mask (torch/tensor [false false true true true] {:dtype :bool}))

;; Select where mask is true
(torch/ix t mask)       ; → [3.0, 4.0, 5.0]
```

---

## Real-World Use Cases

### Extract Batches from Training Data

```clojure
(def batch-data (torch/tensor (range 100)))
(def batch-size 16)
(def batch-idx 2)

(def start (* batch-idx batch-size))
(def end (+ start batch-size))

(torch/ix batch-data [start end])
```

### Train/Test Split

```clojure
(def data (torch/tensor (range 100)))
(def split-point 80)

(def train (torch/ix data [0 split-point]))
(def test (torch/ix data [split-point nil]))
```

### Sliding Window Sequences

```clojure
(def sequence (torch/tensor (range 20)))
(def window-size 5)
(def stride 3)

;; Window 0
(torch/ix sequence [0 window-size])

;; Window 1
(torch/ix sequence [3 (+ 3 window-size)])

;; Window 2
(torch/ix sequence [6 (+ 6 window-size)])
```

### Extract Image Patches

```clojure
(def image (torch/reshape (torch/tensor (range 16)) [4 4]))

;; Top-left 2x2 patch
(torch/ix image [0 2] [0 2])

;; Bottom-right 2x2 patch
(torch/ix image [2 4] [2 4])
```

### Access Model Weights

```clojure
(def weights (torch/tensor (range 100) {:dtype :float32}))

;; First 10 parameters
(torch/ix weights [0 10])

;; Last 10 parameters
(torch/ix weights [-10 nil])

;; Middle parameters
(torch/ix weights [40 60])
```

---

## Helper Functions

These helper functions make working with tensors easier:

```clojure
;; Convert tensor to Clojure vector
(defn tensor->vec [t]
  (mapv torch/item-float (torch/tseq t)))

;; Convert 2D tensor to nested vectors
(defn tensor->vecs [t]
  (mapv #(mapv torch/item-float (torch/tseq %)) (torch/tseq t)))

;; Usage
(def t (torch/tensor [1 2 3]))
(tensor->vec t)  ; → [1.0 2.0 3.0]
```

---

## Summary Table

| Operation | Clorch Syntax | Result |
|-----------|--------------|--------|
| First element | `(ix t 0)` | Single value |
| Last element | `(ix t -1)` | Single value |
| Range slice | `(ix t [2 5])` | Vector |
| Open start | `(ix t [nil 5])` | Vector |
| Open end | `(ix t [5 nil])` | Vector |
| Every Nth | `(ix t [nil nil 2])` | Vector |
| **Reverse** | `(ix t [nil nil -1])` | Vector |
| Reverse from i | `(ix t [i nil -1])` | Vector |
| 2D row | `(ix t 0 :_)` | Vector |
| 2D column | `(ix t :_ 0)` | Vector |

---

## Running Examples

Load the runnable example file from the repository root. It evaluates every slicing example and throws if an operation is invalid:

```clojure
(load-file "examples/slicing_examples.clj")
```
