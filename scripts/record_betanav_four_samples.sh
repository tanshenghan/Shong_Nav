#!/usr/bin/env bash
# One allocated GPU; re-record only the four fixed random samples.
set -Eeuo pipefail
BETA_VIDEO_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BETA_VIDEO_REPO"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BETA_VIDEO_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# Preserve the first GPU/UUID granted by the platform; do not pick a physical GPU
# outside its inherited visibility mask. Set this before Python initializes CUDA.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -z "${BETA_EVAL_PYTHON:-}" ]]; then
    bash "$BETA_VIDEO_REPO/scripts/setup_betanav_stop_train.sh"
    BETA_EVAL_PYTHON="$BETA_VIDEO_REPO/.venv-train-stop/bin/python"
fi
exec "$BETA_EVAL_PYTHON" -u -m lightnav.habitat.sample_videos "$@"
