#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-all}"
SEEDS="${SEEDS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
BASELINES="${BASELINES:-cn aa CSGM deepwalk node2vec graphsage vf2 final regal bright seal turboiso fanmod gspan subgnn NeuGN TPAB ISONET HLOT}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m src.run_all \
  --datasets "${DATASETS}" \
  --methods "${BASELINES}" \
  --seeds "${SEEDS}" \
  --output_root "${OUTPUT_ROOT}" \
  "$@"
