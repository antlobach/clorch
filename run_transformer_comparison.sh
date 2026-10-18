#!/bin/bash
set -euo pipefail

reference_file="$(mktemp -t clorch-transformer-reference.XXXXXX.json)"
trap 'rm -f "$reference_file"' EXIT

uv run --python 3.11 --no-project \
  --with "torch==2.1.2" \
  python tests_comparison/transformer_reference.py "$reference_file"

clojure -Sthreads 1 \
  -Sdeps '{:paths ["src" "tests_comparison"] :deps {org.bytedeco/pytorch-platform {:mvn/version "2.7.1-1.5.12"} org.bytedeco/pytorch-platform-gpu {:mvn/version "2.7.1-1.5.12"}}}' \
  -M -m compare-transformer "$reference_file"
