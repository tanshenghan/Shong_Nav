#!/usr/bin/env bash
# Start a trained Beta server, wait for warm-up, then run the MuJoCo web demo.
set -euo pipefail
DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DEMO_ROOT")"
BETA_PYTHON="${BETA_PYTHON:-$REPO_ROOT/.venv-beta-demo/bin/python}"
if [[ -r "$REPO_ROOT/checkpoints/Light-demo/eval_config.json" ]]; then
    DEFAULT_MODEL_PATH="$REPO_ROOT/checkpoints/Light-demo"
else
    DEFAULT_MODEL_PATH="/mnt/cfs/Models/Light"
fi
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
BETA_CHECKPOINT="${BETA_CHECKPOINT:-/mnt/cfs/Models/beta_nav/r2r_fm_only_h30_prefix_kv_all36_beta1_1p5}"
BETA_PORT="${BETA_PORT:-8050}"
BETA_GPU="${BETA_GPU:-0}"
BETA_STOP_MODE="${BETA_STOP_MODE:-rvq}"
if [[ ! -x "$BETA_PYTHON" ]]; then
    echo "Set BETA_PYTHON to a Python environment with LightNav installed." >&2
    exit 1
fi
mkdir -p "$REPO_ROOT/logs"
BETA_READY_DIR="$(mktemp -d "$REPO_ROOT/logs/beta-demo.XXXXXX")"
BETA_SERVER_PID=""
BETA_DEMO_PID=""
cleanup() {
    if [[ -n "$BETA_DEMO_PID" ]]; then
        kill "$BETA_DEMO_PID" 2>/dev/null || true
        wait "$BETA_DEMO_PID" 2>/dev/null || true
    fi
    if [[ -n "$BETA_SERVER_PID" ]]; then
        kill "$BETA_SERVER_PID" 2>/dev/null || true
        wait "$BETA_SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$BETA_READY_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$REPO_ROOT/.venv-beta-demo/egl/usr/lib/x86_64-linux-gnu" ]]; then
    export LD_LIBRARY_PATH="$REPO_ROOT/.venv-beta-demo/egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -z "${DISPLAY:-}" ]]; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
CUDA_VISIBLE_DEVICES="$BETA_GPU" "$BETA_PYTHON" -c \
    'import torch; assert torch.cuda.is_available(), "Beta demo requires CUDA"'
CUDA_VISIBLE_DEVICES="$BETA_GPU" "$BETA_PYTHON" -m lightnav.serving.ws_server \
    --task vln --backend hf --model_path "$MODEL_PATH" \
    --beta_checkpoint "$BETA_CHECKPOINT" --beta_stop_mode "$BETA_STOP_MODE" \
    --host 127.0.0.1 --port "$BETA_PORT" \
    --ready_file "$BETA_READY_DIR/ready" >"$REPO_ROOT/logs/beta-demo-server.log" 2>&1 &
BETA_SERVER_PID=$!
echo "Loading Beta; server log: $REPO_ROOT/logs/beta-demo-server.log"
while [[ ! -f "$BETA_READY_DIR/ready" ]]; do
    if ! kill -0 "$BETA_SERVER_PID" 2>/dev/null; then
        cat "$REPO_ROOT/logs/beta-demo-server.log" >&2
        exit 1
    fi
    sleep 1
done
echo "Beta ready. Open http://127.0.0.1:8088 (or your --host/--port)."
cd "$DEMO_ROOT"
"${DEMO_PYTHON:-$BETA_PYTHON}" -m vln_mujoco \
    --vln-server "ws://127.0.0.1:$BETA_PORT" "$@" &
BETA_DEMO_PID=$!
wait "$BETA_DEMO_PID"
