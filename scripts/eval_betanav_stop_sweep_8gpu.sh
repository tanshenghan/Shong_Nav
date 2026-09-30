#!/usr/bin/env bash
# Platform command: exec bash /absolute/repo/scripts/eval_betanav_stop_sweep_8gpu.sh
set -Eeuo pipefail
BETA_SWEEP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BETA_SWEEP_REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BETA_SWEEP_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# A plan needs only stdlib, so it must also work on login nodes without environments.
for BETA_SWEEP_ARG in "$@"; do
    if [[ "$BETA_SWEEP_ARG" == "--plan-only" ]]; then
        exec "${BETA_EVAL_PYTHON:-python3}" -u -m lightnav.habitat.launch_stop_sweep "$@"
    fi
done
if [[ -z "${BETA_EVAL_PYTHON:-}" ]]; then
    bash "$BETA_SWEEP_REPO/scripts/setup_betanav_stop_train.sh"
    BETA_EVAL_PYTHON="$BETA_SWEEP_REPO/.venv-train-stop/bin/python"
fi
if [[ ! -x "$BETA_EVAL_PYTHON" ]]; then
    echo "BETA_EVAL_PYTHON is not executable: $BETA_EVAL_PYTHON" >&2
    exit 1
fi
exec "$BETA_EVAL_PYTHON" -u -m lightnav.habitat.launch_stop_sweep "$@"
