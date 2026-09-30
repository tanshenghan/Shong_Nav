#!/usr/bin/env bash
# Start the original LightNav model and the MuJoCo demo in the local runtime.
set -euo pipefail
DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DEMO_ROOT")"
LIGHTNAV_PYTHON="${LIGHTNAV_PYTHON:-$REPO_ROOT/.venv-beta-demo/bin/python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/Light-demo}"
LIGHTNAV_PORT="${LIGHTNAV_PORT:-8050}"
LIGHTNAV_GPU="${LIGHTNAV_GPU:-0}"
if [[ ! -x "$LIGHTNAV_PYTHON" ]]; then
    echo "Set LIGHTNAV_PYTHON to a Python environment with LightNav installed." >&2
    exit 1
fi
mkdir -p "$REPO_ROOT/logs"
LIGHTNAV_READY_DIR="$(mktemp -d "$REPO_ROOT/logs/lightnav-demo.XXXXXX")"
LIGHTNAV_SERVER_PID=""
LIGHTNAV_DEMO_PID=""
cleanup() {
    if [[ -n "$LIGHTNAV_DEMO_PID" ]]; then
        kill "$LIGHTNAV_DEMO_PID" 2>/dev/null || true
        wait "$LIGHTNAV_DEMO_PID" 2>/dev/null || true
    fi
    if [[ -n "$LIGHTNAV_SERVER_PID" ]]; then
        kill "$LIGHTNAV_SERVER_PID" 2>/dev/null || true
        wait "$LIGHTNAV_SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$LIGHTNAV_READY_DIR"
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
CUDA_VISIBLE_DEVICES="$LIGHTNAV_GPU" "$LIGHTNAV_PYTHON" -c \
    'import torch; assert torch.cuda.is_available(), "LightNav demo requires CUDA"'
CUDA_VISIBLE_DEVICES="$LIGHTNAV_GPU" "$LIGHTNAV_PYTHON" -m lightnav.serving.ws_server \
    --task vln --backend hf --model_path "$MODEL_PATH" \
    --host 127.0.0.1 --port "$LIGHTNAV_PORT" \
    --ready_file "$LIGHTNAV_READY_DIR/ready" >"$REPO_ROOT/logs/lightnav-demo-server.log" 2>&1 &
LIGHTNAV_SERVER_PID=$!
echo "Loading original LightNav; server log: $REPO_ROOT/logs/lightnav-demo-server.log"
while [[ ! -f "$LIGHTNAV_READY_DIR/ready" ]]; do
    if ! kill -0 "$LIGHTNAV_SERVER_PID" 2>/dev/null; then
        cat "$REPO_ROOT/logs/lightnav-demo-server.log" >&2
        exit 1
    fi
    sleep 1
done
echo "LightNav ready. Open http://127.0.0.1:8088 (or your --host/--port)."
cd "$DEMO_ROOT"
"${DEMO_PYTHON:-$LIGHTNAV_PYTHON}" -m vln_mujoco \
    --vln-server "ws://127.0.0.1:$LIGHTNAV_PORT" "$@" &
LIGHTNAV_DEMO_PID=$!
wait "$LIGHTNAV_DEMO_PID"
