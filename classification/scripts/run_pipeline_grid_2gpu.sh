#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/datasets}"
shift || true
EXTRA_ARGS=("$@")

EPOCHS=50
BATCH_SIZES=(64 128 256 512)
NDEV=2
PIPELINE_DEVICES="${PIPELINE_DEVICES:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${ROOT_DIR}/output/cifar100_vmamba_grid/pipeline_gpu${NDEV}_${STAMP}"

mkdir -p "${BASE_OUT}"
echo "Running pipeline grid on ${NDEV} GPUs (${PIPELINE_DEVICES}) | data_dir=${DATA_DIR} | out=${BASE_OUT}"

for BS in "${BATCH_SIZES[@]}"; do
  EXP="pipeline_g${NDEV}_bs${BS}_e${EPOCHS}"
  EXP_OUT="${BASE_OUT}/${EXP}"
  mkdir -p "${EXP_OUT}"
  echo "==== ${EXP} ===="
  "${PYTHON_BIN}" "${ROOT_DIR}/classification/cifar100_dist_train.py" \
    --parallel-mode pipeline \
    --pipeline-devices "${PIPELINE_DEVICES}" \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BS}" \
    --output-dir "${EXP_OUT}" \
    --tensorboard \
    --tensorboard-dir "${EXP_OUT}/tensorboard" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${EXP_OUT}/train.log"
done

echo "Completed all pipeline runs for ${NDEV} GPUs."
