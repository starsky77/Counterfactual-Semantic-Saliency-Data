#!/usr/bin/env bash
# Generate descriptions with your own model, then score them into
# CSS_result/Cache/<timestamp><MODEL_NAME>/scored_descriptions_<timestamp>.csv.
#
# Usage:
#   bash Src/custom_model/run_custom_model.sh <MODEL_NAME> [generate_descriptions.py args...]
#
# Examples:
#   bash Src/custom_model/run_custom_model.sh gpt-4o --backend openai --model gpt-4o --workers 8
#   bash Src/custom_model/run_custom_model.sh Qwen_Qwen3-VL-8B-Instruct \
#       --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
#   bash Src/custom_model/run_custom_model.sh my_model --backend custom --custom my_model.py:generate
#
# Environment overrides:
#   PY            python interpreter (default: python3)
#   SCORE_DEVICE  device for the Jasper embedding model (default: cuda if available)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,12p' "$0"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PY:-python3}"
MODEL_NAME="$1"
shift

RAW="${REPO_ROOT}/CSS_result/custom_runs/${MODEL_NAME}/raw_descriptions.jsonl"

"${PY}" "${SCRIPT_DIR}/generate_descriptions.py" --model-name "${MODEL_NAME}" "$@"

score_args=(--model-name "${MODEL_NAME}" --raw "${RAW}")
if [[ -n "${SCORE_DEVICE:-}" ]]; then
  score_args+=(--device "${SCORE_DEVICE}")
fi
"${PY}" "${SCRIPT_DIR}/score_descriptions.py" "${score_args[@]}"

echo
echo "Next: plot Top-1 Accuracy and Kendall tau with your model included:"
echo "  bash Src/custom_model/eval_custom_model.sh"
