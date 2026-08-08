---
layout: default
title: Distributions
---

# Distributions

`clorch.distributions` provides probability distributions with a data-first API.

```clojure
(require '[clorch.distributions :as dist]
         '[clorch.torch :as t])
```

## Constructors

```clojure
(dist/normal 0.0 1.0)
(dist/log-normal 0.0 0.5)
(dist/bernoulli (t/tensor [0.2 0.8]))
(dist/categorical (t/tensor [0.1 0.2 0.7]))
(dist/binomial 10.0 0.3)
(dist/uniform -1.0 2.0)
(dist/poisson 3.0)
(dist/exponential 2.0)
(dist/cauchy 0.0 1.0)
(dist/geometric 0.4)
(dist/gumbel 0.0 1.0)
(dist/laplace 0.0 1.0)
```

## Core Operations

```clojure
(def d (dist/normal 0.0 1.0))

(dist/sample d [4 3])   ;; sample-shape
(dist/log-prob d 0.0)   ;; tensor or scalar
(dist/mean d)
(dist/variance d)
```

## Notes

- `sample-shape` is supported for numeric-parameter distributions and for the common cases above.
- `categorical` sampling uses `multinomial`.
- `log-prob` is implemented analytically for all distributions in this namespace.
