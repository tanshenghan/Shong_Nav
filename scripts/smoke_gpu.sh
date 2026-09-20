#!/usr/bin/env bash
# One-shot GPU smoke test: exercises every real-hardware path with a real checkpoint at
# the smallest possible scale and prints a PASS / FAIL / SKIP table.
#
#   1. predict-hf        lightnav-predict --backend hf on a few frames
#   2. predict-vllm      lightnav-predict --backend vllm_local (vLLM 0.19.1 patch under the real install)
#   3. serve+client      lightnav-serve (--record_dir) + lightnav-ws-client, then lightnav-render
#   4. habitat-eval      scripts/eval_habitat.sh with EPISODES=1 on one GPU   (needs the habitat env + data)
#   5. evt-bench         one EVT-Bench shard against the server from step 3   (needs an EVT-Bench checkout)
#
# Steps 4 and 5 are skipped unless their prerequisites are configured.
#
# Knobs (env vars):
#   MODEL_PATH               checkpoint dir (required)
#   ACTION_TOKENIZER_BUNDLE  RVQ bundle override (optional: a checkpoint that ships
#                            its own decoder needs no decoder flag).
#   TASK                     tracking | vln for the server (default tracking)
#   INSTRUCTION              default "follow the person in the red shirt"
#   SMOKE_VIDEO | SMOKE_FRAMES  an mp4 or a frame directory (default: 12 synthetic frames)
#   GPU                      GPU index (default 0)
#   PORT                     server port (default 8060)
#   INFER_VENV               lightnav virtualenv (default <repo>/.venv)
#   # ---- step 4 (Habitat) ----
#   HABITAT_CONFIG, HABITAT_TASK (vlnce|objectnav), HABITAT_SPLIT, SUCCESS_DISTANCE, DATA_PATH, SCENES_DIR,
#   HABITAT_CONDA_ENV / HABITAT_PYTHON      (see scripts/eval_habitat.sh); set HABITAT_CONFIG to enable
#   # ---- step 5 (EVT-Bench) ----
#   EVT_BENCH_REPO, EVT_CONDA_ENV / EVT_PYTHON, EVT_VARIANT (default dt); set EVT_BENCH_REPO to enable
#   OUTPUT_ROOT              default output/smoke_<timestamp>
#
# Example:
#   MODEL_PATH=/path/to/hf_ckpt \
#   HABITAT_CONFIG=habitat_server/configs/vlnce_r2r.yaml EVT_BENCH_REPO=$HOME/EVT-Bench \
#   bash scripts/smoke_gpu.sh
set -uo pipefail

TAG="[smoke_gpu]"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the checkpoint dir}
ACTION_TOKENIZER_BUNDLE=${ACTION_TOKENIZER_BUNDLE:-}
TASK=${TASK:-tracking}
INSTRUCTION=${INSTRUCTION:-"follow the person in the red shirt"}
GPU=${GPU:-0}
PORT=${PORT:-8060}
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/output/smoke_${TS}}
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

# A checkpoint that ships its decoder (eval_config.json + action_tokenizer/, as the
# released ones do) needs no decoder flag: leave ACTION_TOKENIZER_BUNDLE unset and every
# entry point resolves it from the checkpoint. MODEL_PATH alone is then a complete
# invocation.
decoder_args=()
if [ -n "$ACTION_TOKENIZER_BUNDLE" ]; then
    decoder_args=(--action_tokenizer_bundle "$ACTION_TOKENIZER_BUNDLE")
fi

# lightnav interpreter (same resolution as scripts/start_servers.sh)
INFER_VENV="${INFER_VENV:-$REPO_ROOT/.venv}"
if [ -x "$INFER_VENV/bin/python" ]; then
    PY="$INFER_VENV/bin/python"
    NV_LIBS=$(echo "$INFER_VENV"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr ' ' ':')
    [ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="$NV_LIBS:${LD_LIBRARY_PATH:-}"
else
    PY=python
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

# frame source
if [ -n "${SMOKE_VIDEO:-}" ]; then
    frame_args=(--video "$SMOKE_VIDEO" --fps 4)
elif [ -n "${SMOKE_FRAMES:-}" ]; then
    frame_args=(--frames "$SMOKE_FRAMES")
else
    FRAMES_DIR="$OUTPUT_ROOT/frames"
    mkdir -p "$FRAMES_DIR"
    "$PY" - "$FRAMES_DIR" <<'EOF'
import sys
import numpy as np
from PIL import Image
out = sys.argv[1]
h, w = 270, 480
for i in range(12):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // w), (yy * 255 // h), np.full((h, w), 40 + 15 * i)], -1).astype(np.uint8)
    cx = int(w * (0.3 + 0.04 * i)); img[h // 2 - 30 : h // 2 + 30, cx - 15 : cx + 15] = (220, 40, 40)
    Image.fromarray(img).save(f"{out}/frame_{i:04d}.jpg", quality=92)
print(f"wrote 12 synthetic frames to {out}")
EOF
    frame_args=(--frames "$FRAMES_DIR")
fi

declare -A RESULT
SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

run_step() {   # run_step <name> <log> <cmd...>
    local name=$1 log=$2; shift 2
    echo "$TAG ── $name  (log: $log)"
    if "$@" >"$log" 2>&1; then RESULT[$name]=PASS; echo "$TAG    PASS"
    else RESULT[$name]=FAIL; echo "$TAG    FAIL — tail of $log:"; tail -20 "$log" | sed 's/^/      /'; fi
}

# 1/2. offline prediction, both backends
run_step predict-hf "$OUTPUT_ROOT/predict_hf.log" \
    "$PY" -m lightnav.cli.predict --model_path "$MODEL_PATH" "${decoder_args[@]}" \
        --backend hf "${frame_args[@]}" --instruction "$INSTRUCTION"
run_step predict-vllm "$OUTPUT_ROOT/predict_vllm.log" \
    "$PY" -m lightnav.cli.predict --model_path "$MODEL_PATH" "${decoder_args[@]}" \
        --backend vllm_local --gpu_memory_utilization 0.6 "${frame_args[@]}" --instruction "$INSTRUCTION"
grep -h "raw model output\|latency" "$OUTPUT_ROOT"/predict_*.log 2>/dev/null | sed "s/^/$TAG    /" || true

# 3. serve + client + record + render
READY="$OUTPUT_ROOT/server.ready"
RECORD_DIR="$OUTPUT_ROOT/episodes"
"$PY" -m lightnav.serving.ws_server --model_path "$MODEL_PATH" "${decoder_args[@]}" --task "$TASK" \
    --backend vllm_local --gpu_memory_utilization 0.6 --port "$PORT" --ready_file "$READY" \
    --record_dir "$RECORD_DIR" >"$OUTPUT_ROOT/server.log" 2>&1 &
SERVER_PID=$!
echo "$TAG ── serve+client  (server log: $OUTPUT_ROOT/server.log)"
deadline=$((SECONDS + 900))
while [ ! -f "$READY" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "$TAG    server exited early"; break; fi
    [ "$SECONDS" -ge "$deadline" ] && { echo "$TAG    server not ready after 900s"; break; }
    sleep 3
done
if [ -f "$READY" ]; then
    if "$PY" -m lightnav.cli.ws_client --server "ws://localhost:$PORT" "${frame_args[@]}" \
            --instruction "$INSTRUCTION" >"$OUTPUT_ROOT/client.log" 2>&1 \
       && "$PY" -m lightnav.cli.render "$RECORD_DIR" >"$OUTPUT_ROOT/render.log" 2>&1 \
       && find "$RECORD_DIR" -name 'traj_pointing.mp4' | grep -q .; then
        RESULT[serve+client]=PASS; echo "$TAG    PASS  ($(find "$RECORD_DIR" -name 'traj_pointing.mp4' | head -1))"
    else
        RESULT[serve+client]=FAIL; echo "$TAG    FAIL — see client.log / render.log"; tail -10 "$OUTPUT_ROOT/client.log" | sed 's/^/      /'
    fi
else
    RESULT[serve+client]=FAIL; tail -20 "$OUTPUT_ROOT/server.log" | sed 's/^/      /'
fi

# 4. Habitat: one episode on this GPU
if [ -n "${HABITAT_CONFIG:-}" ]; then
    # `env -u CUDA_VISIBLE_DEVICES`: this script narrows CUDA_VISIBLE_DEVICES to one GPU
    # for the lightnav steps above, but the Habitat env server must see EVERY GPU --
    # habitat-sim matches its CUDA and EGL devices by UUID, so with only GPU $GPU visible
    # and habitat.simulator.habitat_sim_v0.gpu_device_id=$GPU it fails to create a context
    # ("unable to find CUDA device N among M EGL devices"). eval_habitat.sh sets
    # CUDA_VISIBLE_DEVICES per eval client itself; the server is left unrestricted.
    run_step habitat-eval "$OUTPUT_ROOT/habitat_eval.log" \
        env -u CUDA_VISIBLE_DEVICES MODEL_PATH="$MODEL_PATH" TASK="${HABITAT_TASK:-vlnce}" HABITAT_CONFIG="$HABITAT_CONFIG" \
            ${HABITAT_SPLIT+SPLIT="$HABITAT_SPLIT"} SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-}" \
            DATA_PATH="${DATA_PATH:-}" SCENES_DIR="${SCENES_DIR:-}" \
            GPU_IDS="$GPU" EPISODES=1 BASE_PORT=5599 GPU_MEM_UTIL=0.45 \
            ACTION_TOKENIZER_BUNDLE="$ACTION_TOKENIZER_BUNDLE" CLIENT_ARGS="--save_video" \
            OUTPUT_ROOT="$OUTPUT_ROOT/habitat" INFER_VENV="$INFER_VENV" \
            ${HABITAT_CONDA_ENV+HABITAT_CONDA_ENV="$HABITAT_CONDA_ENV"} ${HABITAT_PYTHON+HABITAT_PYTHON="$HABITAT_PYTHON"} \
            bash "$REPO_ROOT/scripts/eval_habitat.sh"
    [ -f "$OUTPUT_ROOT/habitat/summary.json" ] && grep -m1 '"table_format"' "$OUTPUT_ROOT/habitat/summary.json" | sed "s/^/$TAG    /"
else
    RESULT[habitat-eval]=SKIP; echo "$TAG ── habitat-eval  SKIP (set HABITAT_CONFIG to enable)"
fi

# 5. EVT-Bench: one shard against the running server
if [ -n "${EVT_BENCH_REPO:-}" ] && [ -f "$READY" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    EVT_PYTHON=${EVT_PYTHON:-}
    if [ -z "$EVT_PYTHON" ]; then
        EVT_PYTHON=$(bash -c "source \"\${CONDA_SH:-\$HOME/miniconda3/etc/profile.d/conda.sh}\" 2>/dev/null && conda activate \"${EVT_CONDA_ENV:-evt_bench}\" && echo \"\$CONDA_PREFIX/bin/python\"" 2>/dev/null || true)
    fi
    if [ -x "${EVT_PYTHON:-/nonexistent}" ] && [ -f "$EVT_BENCH_REPO/trackvla_client_agent.py" ]; then
        V=${EVT_VARIANT:-dt}
        run_step evt-bench "$OUTPUT_ROOT/evt_bench.log" \
            bash -c "cd '$EVT_BENCH_REPO' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=habitat-lab '$EVT_PYTHON' run.py \
                --run-type eval --model-name trackvla --split-num 30 --split-id 0 \
                --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_infer_${V}.yaml \
                --model-path ws://localhost:$PORT --save-path '$OUTPUT_ROOT/evt/$V' \
                && '$EVT_PYTHON' analyze_results.py --path '$OUTPUT_ROOT/evt/$V'"
        tail -1 "$OUTPUT_ROOT/evt_bench.log" 2>/dev/null | sed "s/^/$TAG    /"
    else
        RESULT[evt-bench]=SKIP; echo "$TAG ── evt-bench  SKIP (need EVT_PYTHON/EVT_CONDA_ENV and trackvla_client_agent.py in $EVT_BENCH_REPO)"
    fi
else
    RESULT[evt-bench]=SKIP; echo "$TAG ── evt-bench  SKIP (set EVT_BENCH_REPO to enable; needs the server from step 3)"
fi

echo
echo "$TAG ═══ summary ═══   (outputs: $OUTPUT_ROOT)"
fail=0
for step in predict-hf predict-vllm serve+client habitat-eval evt-bench; do
    printf '%s   %-14s %s\n' "$TAG" "$step" "${RESULT[$step]:-FAIL}"
    [ "${RESULT[$step]:-FAIL}" = FAIL ] && fail=1
done
exit $fail
