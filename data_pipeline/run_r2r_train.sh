#!/usr/bin/env bash
set -euo pipefail
# Usage: bash data_pipeline/run_r2r_train.sh OUTPUT_DIR [build_r2r_train.py flags]
BETA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BETA_TRAIN_OUTPUT="$(realpath -m "${1:?supply the final LeRobotDataset output directory}")"
shift
cd "$BETA_REPO"
exec env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE="$BETA_TRAIN_OUTPUT.work/cache/merged" \
  HF_DATASETS_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$BETA_REPO/data_pipeline/.venv/lerobot/bin/python" \
  "$BETA_REPO/data_pipeline/build_r2r_train.py" --output-dir "$BETA_TRAIN_OUTPUT" "$@"
