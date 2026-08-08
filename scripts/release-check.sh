#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: scripts/release-check.sh --mode cpu|gpu|both

Modes:
  cpu   Run release checks with CLORCH_FORCE_CPU=1
  gpu   Run release checks with CLORCH_FORCE_GPU=1 and CUDA smoke test
  both  Run cpu checks, then gpu checks, in separate processes
EOF
}

MODE="cpu"

if [[ "${1-}" == "--mode" && -n "${2-}" ]]; then
  MODE="$2"
elif [[ "${1-}" == "--cuda" ]]; then
  MODE="gpu"
elif [[ -n "${1-}" ]]; then
  usage
  exit 1
fi

run_mode() {
  local mode="$1"
  local force_cpu=""
  local force_gpu=""

  case "$mode" in
    cpu)
      force_cpu="1"
      ;;
    gpu)
      force_gpu="1"
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      usage
      exit 1
      ;;
  esac

  echo "==> Running Clojure test suite ($mode)"
  CLORCH_FORCE_CPU="$force_cpu" \
  CLORCH_FORCE_GPU="$force_gpu" \
  clojure -Sthreads 1 -M -m clorch.release-check --mode "$mode"

  echo
  echo "==> Running Clorch vs PyTorch parity suite ($mode)"
  CLORCH_FORCE_CPU="$force_cpu" \
  CLORCH_FORCE_GPU="$force_gpu" \
  ./run_comparison.sh

  echo
  echo "Release checks passed for mode: $mode"
}

case "$MODE" in
  cpu|gpu)
    run_mode "$MODE"
    ;;
  both)
    run_mode cpu
    echo
    run_mode gpu
    ;;
  *)
    usage
    exit 1
    ;;
esac
