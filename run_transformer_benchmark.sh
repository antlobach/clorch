#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --python 3.11 --no-project \
  --with "torch==2.1.2" \
  python benchmarks/transformer_pytorch.py

clojure -Sthreads 1 \
  -Sdeps '{:paths ["src" "benchmarks"] :deps {org.bytedeco/pytorch-platform {:mvn/version "2.7.1-1.5.12"} org.bytedeco/pytorch-platform-gpu {:mvn/version "2.7.1-1.5.12"}}}' \
  -M -e '(require (quote transformer-clorch)) (prn (transformer-clorch/benchmark))'
