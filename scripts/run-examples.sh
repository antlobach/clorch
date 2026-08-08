#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_clj_require() {
  local name="$1"
  local ns="$2"
  echo "==> $name"
  clojure -M -e "(require '$ns :reload)"
  echo
}

run_clj_main() {
  local name="$1"
  local ns="$2"
  echo "==> $name"
  clojure -M -m "$ns"
  echo
}

run_cmd() {
  local name="$1"
  shift
  echo "==> $name"
  "$@"
  echo
}

run_example_smoke() {
  local name="$1"
  local ns="$2"
  shift 2
  echo "==> $name"
  env "$@" clojure -M -e "(require '$ns :reload)"
  echo
}

run_clj_require "autograd_tutorial.clj" autograd-tutorial
run_clj_require "custom_model_dataset.clj" custom-model-dataset
run_clj_require "data_loading_tutorial.clj" data-loading-tutorial
run_clj_require "llms_from_scratch.clj" llms-from-scratch
run_clj_require "modern_llama.clj" modern-llama
run_clj_require "pytorch_1h_tutorial.clj" pytorch-1h-tutorial
run_clj_require "pytorch_basics_tutorial.clj" pytorch-basics-tutorial
run_clj_require "synthetic.clj" synthetic
run_clj_require "simple.clj" simple
run_cmd "einsum_edsl + slicing_examples" clojure -M -e "(load-file \"examples/einsum_edsl.clj\") (load-file \"examples/slicing_examples.clj\")"
run_example_smoke "bayesian_linear_regression_mcmc.clj" bayesian-linear-regression-mcmc \
  CLORCH_BAYES_LR_CHAINS=2 \
  CLORCH_BAYES_LR_STEPS=160 \
  CLORCH_BAYES_LR_BURN_IN=40 \
  CLORCH_BAYES_LR_THIN=4
run_clj_require "plot_bayesian_linear_regression_mcmc.clj" plot-bayesian-linear-regression-mcmc
run_example_smoke "bayesian_nn_regression_mcmc.clj" bayesian-nn-regression-mcmc \
  CLORCH_BAYES_NN_CHAINS=2 \
  CLORCH_BAYES_NN_STEPS=160 \
  CLORCH_BAYES_NN_BURN_IN=40 \
  CLORCH_BAYES_NN_THIN=4
run_clj_require "plot_bayesian_nn_regression_mcmc.clj" plot-bayesian-nn-regression-mcmc
run_cmd "nanochat.clj" env \
  CLORCH_NANOCHAT_TRAIN_BATCHES=3 \
  CLORCH_NANOCHAT_SAMPLE_TOKENS=8 \
  CLORCH_NANOCHAT_CHAT_MAX_NEW_TOKENS=16 \
  clojure -M -e "(require 'nanochat :reload) (nanochat/train)"
run_cmd "pytorch_comparison.py" uv run --python 3.12 --no-project --with torch --with "numpy<2" python examples/pytorch_comparison.py

echo "All example runs passed."
