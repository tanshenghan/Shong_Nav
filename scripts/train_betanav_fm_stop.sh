#!/usr/bin/env bash
# Platform entry point: exec bash /absolute/repo/scripts/train_betanav_fm_stop.sh
# The platform allocates GPUs and queues this foreground process.
set -Eeuo pipefail

BETA_STOP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BETA_STOP_REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BETA_STOP_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${BETA_TRAIN_PYTHON:-}" ]]; then
    bash "$BETA_STOP_REPO/scripts/setup_betanav_stop_train.sh"
    BETA_TRAIN_PYTHON="$BETA_STOP_REPO/.venv-train-stop/bin/python"
fi
if [[ ! -x "$BETA_TRAIN_PYTHON" ]]; then
    echo "BETA_TRAIN_PYTHON is not executable: $BETA_TRAIN_PYTHON" >&2
    exit 1
fi

# Keep the platform's CUDA_VISIBLE_DEVICES intact. No background GPU-wait loop.
exec "$BETA_TRAIN_PYTHON" -u -m lightnav.training.launch_fm_stop "$@"
