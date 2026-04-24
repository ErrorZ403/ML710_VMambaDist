#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/datasets}"
shift || true
EXTRA_ARGS=("$@")

echo "Sequential pipeline runs for 2 and 4 GPUs."
echo "Note: pipeline mode needs >=2 GPUs, so 1 GPU is intentionally skipped."
bash "${ROOT_DIR}/classification/scripts/run_pipeline_grid_2gpu.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"
bash "${ROOT_DIR}/classification/scripts/run_pipeline_grid_4gpu.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"
echo "All sequential pipeline GPU-count runs finished."
