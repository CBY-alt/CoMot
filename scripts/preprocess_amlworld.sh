#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TRANS_PATH="${AMLWORLD_TRANS_PATH:-${PROJECT_ROOT}/data/raw/amlworld/HI-Small_Trans.csv}"
PATTERN_PATH="${AMLWORLD_PATTERN_PATH:-${PROJECT_ROOT}/data/raw/amlworld/HI-Small_Patterns.txt}"

RUN_NAME="${RUN_NAME:-amlworld_hi_small}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SEED="${SEED:-42}"

if [[ ! -f "${TRANS_PATH}" ]]; then
  echo "Missing transaction CSV: ${TRANS_PATH}" >&2
  echo "Place AMLWorld HI-Small_Trans.csv under data/raw/amlworld, or set AMLWORLD_TRANS_PATH." >&2
  exit 1
fi

if [[ ! -f "${PATTERN_PATH}" ]]; then
  echo "Missing pattern TXT: ${PATTERN_PATH}" >&2
  echo "Place AMLWorld HI-Small_Patterns.txt under data/raw/amlworld, or set AMLWORLD_PATTERN_PATH." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/amlworld_pipeline.py" \
  --config "${PROJECT_ROOT}/configs/amlworld_hi_small.json" \
  --dataset amlworld \
  --method comot \
  --output_dir "${RUN_DIR}" \
  --seed "${SEED}" \
  --trans_path "${TRANS_PATH}" \
  --pattern_path "${PATTERN_PATH}"

echo "Pipeline completed. Outputs: ${RUN_DIR}"
