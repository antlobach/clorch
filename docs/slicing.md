# PyTorch to Clorch Slicing Translation Guide

This document provides a comprehensive mapping between PyTorch tensor indexing/slicing syntax and Clorch equivalents.

## Quick Reference

| PyTorch | Clorch | Description |
|---------|--------|-------------|
| `t[0]` | `(ix t 0)` | Single element |
| `t[-1]` | `(ix t -1)` | Last element (negative) |
| `t[0, 1]` | `(ix t 0 1)` | Multi-dimensional |
| `t[:, :]` | `(ix t :all :all)` | Select all |
| `t[:, :_]` | `(ix t :all :_)` | Alternative select all |
| `t[0:5]` | `(ix t [0 5])` | Range slice |
| `t[:5]` | `(ix t [0 5])` | From start |
| `t[5:]` | `(ix t [5 nil])` | To end |
| `t[::2]` | `(ix t [nil nil 2])` | Every 2nd element |
| `t[1:8:2]` | `(ix t [1 8 2])` | Step slice |
| `t[..., 0]` | `(ix t (quote ...) 0)` | Ellipsis |
| `t[0, ...]` | `(ix t 0 (quote ...))` | Ellipsis |
| `t[::-1]` | `(ix t [nil nil -1])` | Reverse (NEW!) |
| `t[5::-1]` | `(ix t [5 nil -1])` | Reverse from index |
| `t[::-2]` | `(ix t [nil nil -2])` | Reverse every 2nd |

---

## Basic Indexing

### 1D Tensors

```python
# PyTorch
x = torch.tensor([10, 11, 12, 13, 14])
x[0]      # → tensor(10)
x[-1]     # → tensor(14)
```

```clojure
;; Clorch
(def x (torch/tensor [10 11 12 13 14]))
(torch/ix x 0)       ;; → 10.0
(torch/ix x -1)      ;; → 14.0
```

### Multi-dimensional Tensors

```python
# PyTorch
x = torch.tensor([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])
x[0, 1]    # → tensor(2)
x[0]       # → tensor([1, 2, 3])
x[:, 1]    # → tensor([2, 5, 8])
```

```clojure
;; Clorch
(def x (torch/tensor [[1 2 3] 
                      [4 5 6] 
                      [7 8 9]]))
(torch/ix x 0 1)      ;; → 2.0
(torch/ix x 0)        ;; → [1.0, 2.0, 3.0]
(torch/ix x :all 1)   ;; → [2.0, 5.0, 8.0]
```

---

## Slicing

### Basic Slices

```python
# PyTorch
y = torch.arange(10)  # [0,1,2,3,4,5,6,7,8,9]
y[2:5]    # → tensor([2, 3, 4])
y[:4]     # → tensor([0, 1, 2, 3])
y[6:]     # → tensor([6, 7, 8, 9])
```

```clojure
;; Clorch
(def y (torch/tensor (range 10)))
(torch/ix y [2 5])    ;; → [2.0, 3.0, 4.0]
(torch/ix y [0 4])    ;; → [0.0, 1.0, 2.0, 3.0]
(torch/ix y [6 10])   ;; → [6.0, 7.0, 8.0, 9.0]
```

### Step Slices

```python
# PyTorch
y = torch.arange(10)
y[::2]      # → tensor([0, 2, 4, 6, 8])   # every 2nd
y[1:8:2]    # → tensor([1, 3, 5, 7])     # start:stop:step
y[::-1]     # → tensor([9,8,7,6,5,4,3,2,1,0])  # reverse
```

```clojure
;; Clorch
(def y (torch/tensor (range 10)))
(torch/ix y [nil nil 2])    ;; → [0.0, 2.0, 4.0, 6.0, 8.0]
(torch/ix y [1 8 2])        ;; → [1.0, 3.0, 5.0, 7.0]
```

### Negative Step Slicing (NEW!)

PyTorch and Clorch support negative steps for reversing tensors:

```python
# PyTorch
y = torch.arange(10)
y[::-1]     # → tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # Full reverse
y[5::-1]    # → tensor([5, 4, 3, 2, 1, 0])              # From index 5 to start
y[:5:-1]    # → tensor([9, 8, 7, 6])                    # From end to index 5
y[5:2:-1]   # → tensor([5, 4, 3])                      # Partial reverse
y[::-2]      # → tensor([9, 7, 5, 3, 1])                # Reverse every 2nd
y[::-3]      # → tensor([9, 6, 3, 0])                  # Reverse every 3rd
```

```clojure
;; Clorch
(def y (torch/tensor (range 10)))
(torch/ix y [nil nil -1])    ;; → [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
(torch/ix y [5 nil -1])       ;; → [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
(torch/ix y [nil 5 -1])       ;; → [9.0, 8.0, 7.0, 6.0]
(torch/ix y [5 2 -1])         ;; → [5.0, 4.0, 3.0]
(torch/ix y [nil nil -2])      ;; → [9.0, 7.0, 5.0, 3.0, 1.0]
(torch/ix y [nil nil -3])     ;; → [9.0, 6.0, 3.0, 0.0]
```

### Negative Indexing

```python
# PyTorch
y = torch.arange(10)
y[-1]       # → tensor(9)
y[-3:]      # → tensor([7, 8, 9])
y[:-3]      # → tensor([0, 1, 2, 3, 4, 5, 6])
```

```clojure
;; Clorch
(def y (torch/tensor (range 10)))
(torch/ix y -1)             ;; → 9.0
(torch/ix y [-3 10])        ;; → [7.0, 8.0, 9.0]
(torch/ix y [0 -3])         ;; → [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

---

## Advanced Indexing

### Ellipsis

```python
# PyTorch
t = torch.arange(24).reshape(2, 3, 4)
t[0, ...]   # → tensor([[ 0,  1,  2,  3], ...])
t[..., 0]   # → tensor([[ 0,  4,  8], 
                 #        [12, 16, 20]])
```

```clojure
;; Clorch
(def t (torch/reshape (torch/tensor (range 24)) [2 3 4]))
(torch/ix t 0 (quote ...))   ;; → shape [3, 4]
(torch/ix t (quote ...) 0)   ;; → shape [2, 3]
```

### Select All (Identity)

```python
# PyTorch
t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6]])
t[:, :]     # → full tensor
t[:,]      # → full tensor
```

```clojure
;; Clorch
(def t (torch/tensor [[1 2 3] 
                      [4 5 6]]))
(torch/ix t :all :all)       ;; → full tensor
(torch/ix t :_ :_)           ;; → full tensor (alternative)
```

---

## 2D Tensor Examples

```python
# PyTorch
x = torch.tensor([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11]])

x[0:2, 1:3]   # → tensor([[1, 2], 
                 #            [5, 6]])
x[:, -2:]     # → tensor([[ 2,  3],
                 #            [ 6,  7],
                 #            [10, 11]])
x[0, 1:]      # → tensor([1, 2, 3])
x[::2, :]     # → tensor([[ 0,  1,  2,  3],
                 #            [ 8,  9, 10, 11]])
```

```clojure
;; Clorch
(def x (torch/tensor [[0 1 2 3]
                      [4 5 6 7]
                      [8 9 10 11]]))

(torch/ix x [0 2] [1 3])       ;; → [[1.0, 2.0], [5.0, 6.0]]
(torch/ix x :all [2 4])        ;; → [[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]
(torch/ix x 0 [1 4])           ;; → [1.0, 2.0, 3.0]
(torch/ix x [0 3 2] :all)      ;; → [[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]]
```

### Negative Step with 2D Tensors

```python
# PyTorch
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9,10,11,12]])
x[::-1, :]    # → reverse rows
x[:, ::-1]    # → reverse columns
```

```clojure
;; Clorch
(def x (torch/tensor [[1 2 3 4] [5 6 7 8] [9 10 11 12]]))

;; Reverse rows
(torch/ix x [nil nil -1] :_)  
;; → [[9.0, 10.0, 11.0, 12.0], [5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]]

;; Reverse columns
(torch/ix x :_ [nil nil -1])
;; → [[4.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0], [12.0, 11.0, 10.0, 9.0]]

;; Reverse both
(torch/ix x [nil nil -1] [nil nil -1])
;; → [[12.0, 11.0, 10.0, 9.0], [8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]
```

---

## Helper Functions

### Converting Tensor to Clojure Vector

```clojure
(defn tensor->vec [t]
  (mapv torch/item-float (torch/tseq t)))

;; Usage
(def x (torch/tensor [1 2 3]))
(tensor->vec (torch/ix x [0 2]))  ;; → [1.0, 2.0]
```

---

## Indexer Syntax Summary

| Clorch Syntax | Meaning |
|---------------|---------|
| `0`, `1`, `-1` | Integer index |
| `:all` | Select entire dimension |
| `:_` | Select entire dimension (alternative) |
| `(quote ...)` or `'...` | Ellipsis (fill remaining dims) |
| `[start stop]` | Slice from start to stop |
| `[start stop step]` | Slice with step (positive) |
| `[start stop -1]` | Slice with negative step (reverse) |
| `[nil stop]` | From start to stop |
| `[start nil]` | From start to end |
| `[nil nil step]` | Every step-th element |
| `[nil nil -1]` | Reverse entire tensor |
| `[nil nil -2]` | Reverse every 2nd element |

---

## Notes

1. **Float results**: All tensor values are floats by default. Use `torch/item-float` to extract values.

2. **Vectors vs Keywords**: Slice vectors use `nil` for default values rather than Clojure's `::` auto-resolved keywords which would conflict with actual keywords.

3. **Ellipsis**: Use `(quote ...)` or `(... )` in the REPL - note this may require reader conditional for scripts.

4. **Bounds checking**: Clorch includes bounds checking for most operations.