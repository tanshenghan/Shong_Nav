#!/usr/bin/env bash
set -euo pipefail
# Usage: bash data_pipeline/run.sh NEW_OUTPUT_DIR [10|-1] [collector flags...]
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BETA_HABITAT_PYTHON=${HABITAT_PYTHON:-/mnt/pfs/jiayilin/Image2Sim-Robot/envs/habitat-image2sim/bin/python}
BETA_OUTPUT=${1:?supply a NEW output directory}
BETA_EPISODES=${2:-10}
cd "$REPO"
exec env -u CUDA_VISIBLE_DEVICES \
  PYTHONPATH="$REPO/data_pipeline/.venv/deps${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONDONTWRITEBYTECODE=1 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet \
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
  "$BETA_HABITAT_PYTHON" "$REPO/data_pipeline/collect_r2r.py" \
  --output-dir "$BETA_OUTPUT" \
  --data-root "${BETA_DATA_ROOT:-/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed}" \
  --scenes-dir "${BETA_SCENES_DIR:-/mnt/cfs/data/habitat/scene_datasets}" \
  --rvq-bundle "${BETA_RVQ_BUNDLE:-/mnt/cfs/Models/Light/action_tokenizer}" \
  --width 470 --height 280 --gpu-id "${BETA_GPU_ID:-3}" \
  --seed 20260911 --episodes "$BETA_EPISODES" "${@:3}"
