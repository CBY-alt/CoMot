#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SUMMARY_DIR="${PROJECT_ROOT}/outputs/summary"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m src.experiments.results \
  --output_root outputs \
  --summary_dir outputs/summary \
  --processed_root data/processed \
  --datasets all \
  --methods all \
  --seeds auto \
  --main_metric motif_f1

mkdir -p "${SUMMARY_DIR}"

"${PYTHON_BIN}" scripts/generate_dataset_construction_stats.py \
  --processed_root data/processed \
  --summary_dir outputs/summary

if [[ -f "${SUMMARY_DIR}/main_table.csv" ]]; then
  cp "${SUMMARY_DIR}/main_table.csv" "${SUMMARY_DIR}/main_comparison.csv"
fi
if [[ -f "${SUMMARY_DIR}/main_table_latex.tex" ]]; then
  cp "${SUMMARY_DIR}/main_table_latex.tex" "${SUMMARY_DIR}/main_comparison_latex.tex"
fi
if [[ -f "${SUMMARY_DIR}/ablation_table.csv" ]]; then
  cp "${SUMMARY_DIR}/ablation_table.csv" "${SUMMARY_DIR}/ablation.csv"
fi

echo "paper tables refreshed under outputs/summary"
echo "main metrics: motif_f1 edge_f1 node_f1 motif_jaccard runtime num_candidates"
echo "AUC is intentionally excluded from paper table columns."
