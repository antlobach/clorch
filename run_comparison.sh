#!/bin/bash
echo "Running Clorch vs PyTorch comparison tests..."
uv run --python 3.11 --no-project \
  --with "torch==2.1.2" \
  --with "numpy<2" \
  --with packaging \
  --with setuptools \
  python tests_comparison/compare_torch.py
