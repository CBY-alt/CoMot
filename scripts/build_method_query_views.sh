#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/build_method_query_views.sh <dataset|all>" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m src.data.query_view "$1"
