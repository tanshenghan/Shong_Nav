#!/usr/bin/env bash
# Paste `exec bash /absolute/repo/scripts/eval_betanav_fm_stop_8gpu.sh` into the platform.
set -Eeuo pipefail
BETA_EVAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BETA_EVAL_REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BETA_EVAL_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${BETA_EVAL_PYTHON:-}" ]]; then
    bash "$BETA_EVAL_REPO/scripts/setup_betanav_stop_train.sh"
    BETA_EVAL_PYTHON="$BETA_EVAL_REPO/.venv-train-stop/bin/python"
fi
if [[ ! -x "$BETA_EVAL_PYTHON" ]]; then
    echo "BETA_EVAL_PYTHON is not executable: $BETA_EVAL_PYTHON" >&2
    exit 1
fi
exec "$BETA_EVAL_PYTHON" -u -m lightnav.habitat.launch_fm_stop_eval "$@"
