#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${EXP_PYTHON:-python}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONDONTWRITEBYTECODE=1
stage="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  build) "$python_bin" src/build_db.py "$@" ;;
  prompts) "$python_bin" src/prompts.py "$@" ;;
  generate) "$python_bin" src/generation.py "$@" ;;
  ask) "$python_bin" src/ask.py "$@" ;;
  evaluate) "$python_bin" src/evaluate.py "$@" ;;
  statistics) "$python_bin" src/statistical_analysis.py "$@" ;;
  export) "$python_bin" src/output.py "$@" ;;
  *) echo 'Usage: bash scripts/run.sh {build|prompts|generate|ask|evaluate|statistics|export} [options]' >&2; exit 2 ;;
esac
