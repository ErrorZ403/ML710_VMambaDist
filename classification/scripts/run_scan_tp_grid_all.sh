#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/datasets}"
shift || true
EXTRA_ARGS=("$@")

echo "Sequential scan_tp runs for 2 and 4 GPUs."
echo "Note: scan_tp requires nproc_per_node > 1, so 1 GPU is intentionally skipped."
bash "${ROOT_DIR}/classification/scripts/run_scan_tp_grid_2gpu.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"
bash "${ROOT_DIR}/classification/scripts/run_scan_tp_grid_4gpu.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"
echo "All sequential scan_tp GPU-count runs finished."
