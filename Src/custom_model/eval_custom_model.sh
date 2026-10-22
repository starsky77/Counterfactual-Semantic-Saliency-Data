#!/usr/bin/env bash
# Plot Top-1 Accuracy and mean Kendall tau (human vs model) for every run under
# CSS_result/Cache/, including your own scored_descriptions CSVs, next to the
# paper's models. Writes only under results/custom_model/ (paper results untouched).
#
# Usage:
#   bash Src/custom_model/eval_custom_model.sh
#
# Environment overrides:
#   PY       python interpreter (default: python3)
#   OUT_ROOT output folder (default: results/custom_model)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-python3}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/custom_model}"
WORK="${OUT_ROOT}/intermediate"
mkdir -p "${WORK}"

OUT_DIR="${WORK}/eval_outputs" bash Src/eval.sh

n_runs=$(find "${WORK}/eval_outputs" -name '*_rank_correlation.csv' | wc -l)
grid_cols=3
grid_rows=$(( (n_runs + 1 + grid_cols - 1) / grid_cols ))

"${PY}" Src/pairwise_kendall_tau.py --mode human \
  --eval-dir "${WORK}/eval_outputs" \
  --out-dir "${WORK}/eval_output_tau" \
  --human-grid-rows "${grid_rows}" --human-grid-cols "${grid_cols}"

SUMMARY="${WORK}/eval_output_tau/human_vs_model/kendall_vs_human_summary.csv"
"${PY}" fig_create/plot_kendall_top1_acc_bars.py "${SUMMARY}" -o "${OUT_ROOT}/kendall_top1_acc_bars.png"
"${PY}" fig_create/plot_kendall_mean_tau_bars.py "${SUMMARY}" -o "${OUT_ROOT}/kendall_mean_tau_bars.png"

echo
echo "Done. Figures in ${OUT_ROOT}; per-model numbers in ${SUMMARY}"
