#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/datasets}"
shift || true
EXTRA_ARGS=("$@")

EPOCHS=50
BATCH_SIZES=(64 128 256 512)
NPROC=4
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${ROOT_DIR}/output/cifar100_vmamba_grid/scan_tp_gpu${NPROC}_${STAMP}"

mkdir -p "${BASE_OUT}"
echo "Running scan_tp grid on ${NPROC} GPUs | data_dir=${DATA_DIR} | out=${BASE_OUT}"

for BS in "${BATCH_SIZES[@]}"; do
  EXP="scan_tp_g${NPROC}_bs${BS}_e${EPOCHS}"
  EXP_OUT="${BASE_OUT}/${EXP}"
  mkdir -p "${EXP_OUT}"
  echo "==== ${EXP} ===="
  torchrun --nproc_per_node="${NPROC}" "${ROOT_DIR}/classification/cifar100_dist_train.py" \
    --parallel-mode scan_tp \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BS}" \
    --output-dir "${EXP_OUT}" \
    --tensorboard \
    --tensorboard-dir "${EXP_OUT}/tensorboard" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${EXP_OUT}/train.log"
done

echo "Completed all scan_tp runs for ${NPROC} GPUs."
