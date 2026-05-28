#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_dataset.sh <dataset> [extra run_all args]" >&2
  exit 2
fi

DATASET="$1"
shift

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m src.experiments.run_all --datasets "${DATASET}" --methods all "$@"
