#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/datasets}"
shift || true
EXTRA_ARGS=("$@")

echo "Sequential grid runs for 1, 2, 4 GPUs."
bash "${ROOT_DIR}/classification/scripts/run_ddp_grid_1gpu.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"
bash "${ROOT_DIR}/classification/scripts/run_ddp_grid_2gpu.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"
bash "${ROOT_DIR}/classification/scripts/run_ddp_grid_4gpu.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"
echo "All sequential GPU-count runs finished."
