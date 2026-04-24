#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/datasets}"
shift || true
EXTRA_ARGS=("$@")

echo "Running all parallel strategies sequentially with batch sizes 64,128,256,512 and 50 epochs."
echo "Strategy 1/3: DDP (1,2,4 GPUs)"
bash "${ROOT_DIR}/classification/scripts/run_ddp_grid_all.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"

echo "Strategy 2/3: scan_tp (2,4 GPUs)"
bash "${ROOT_DIR}/classification/scripts/run_scan_tp_grid_all.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"

echo "Strategy 3/3: pipeline (2,4 GPUs)"
bash "${ROOT_DIR}/classification/scripts/run_pipeline_grid_all.sh" "${DATA_DIR}" "${EXTRA_ARGS[@]}"

echo "All strategies completed."
