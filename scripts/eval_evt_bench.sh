#!/usr/bin/env bash
# End-to-end EVT-Bench tracking evaluation:
#   1. start N x K model servers (scripts/start_servers.sh, one port each)
#   2. wait until every server has written its .ready file
#   3. run EVT-Bench's run.py over CHUNKS dataset shards, in parallel waves,
#      each Habitat process talking to one server URL
#   4. aggregate SR / FR / CR with EVT-Bench's analyze_results.py
# Servers are killed on exit (trap).
#
# Prerequisites (see docs/EVAL_EVT_BENCH.md):
#   - EVT-Bench checkout with scenes + humanoid data, conda env with habitat-sim
#     0.3.1, `pip install -e habitat-lab`, `pip install websocket-client`
#   - evt_bench/trackvla_client_agent.py copied next to run.py and
#     evt_bench/run_py.patch applied
#
# Knobs (env vars):
#   # ---- model / vocabulary (passed through to scripts/start_servers.sh) ----
#   MODEL_PATH               checkpoint dir (required)
#   ACTION_TOKENIZER_BUNDLE  RVQ bundle dir (optional: only to override the
#                            decoder the checkpoint ships)
#   BACKEND                  vllm_local | hf (default vllm_local)
#   NUM_HISTORY_FRAMES, MAX_BATCH_SIZE, GPU_MEM_UTIL, INFER_VENV, CPU_PIN
#                            optional, forwarded to start_servers.sh
#   # ---- topology ----
#   NUM_GPUS                 default 1
#   SERVERS_PER_GPU          default 4 (GPU_MEM_UTIL auto = 0.85/SERVERS_PER_GPU)
#   BASE_PORT                default 8050
#   CLIENTS_PER_SERVER       Habitat processes per server (default 1; the server
#                            micro-batches concurrent sessions)
#   READY_TIMEOUT_S          max wait for all servers (default 1800)
#   # ---- EVT-Bench driver ----
#   EVT_BENCH_REPO           EVT-Bench checkout (default $HOME/EVT-Bench)
#   EVT_CONDA_ENV            conda env with habitat-lab (default evt_bench), or
#   EVT_PYTHON               explicit python interpreter (overrides EVT_CONDA_ENV)
#   TASK_VARIANTS            space-separated subset of "dt stt at" (default dt)
#   CHUNKS                   dataset shards = run.py --split-num (default 30;
#                            keep 30, see docs)
#   EVT_JAW_HFOV             jaw camera hfov override in deg (default empty =
#                            upstream 86); we use 120 for our checkpoints
#   EVT_JAW_HEIGHT           jaw camera mount height override in m (default empty)
#   OUTPUT_ROOT              results root (default $EVT_BENCH_REPO/exp_results/lightnav_<ts>)
#   LOG_DIR                  logs (default $OUTPUT_ROOT/logs)
#
# Example:
#   MODEL_PATH=/path/to/checkpoint \
#   NUM_GPUS=2 SERVERS_PER_GPU=4 EVT_JAW_HFOV=120 \
#   bash scripts/eval_evt_bench.sh
set -euo pipefail

TAG="[eval_evt_bench]"

# ── model / vocabulary ───────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the checkpoint dir}
ACTION_TOKENIZER_BUNDLE=${ACTION_TOKENIZER_BUNDLE:-}
# Not required for a checkpoint that ships its own decoder (eval_config.json plus
# action_tokenizer/, as the released checkpoints do): leave unset and lightnav-serve
# resolves the decoder from the checkpoint itself.
BACKEND=${BACKEND:-vllm_local}

# ── topology ─────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}
SERVERS_PER_GPU=${SERVERS_PER_GPU:-4}
BASE_PORT=${BASE_PORT:-8050}
CLIENTS_PER_SERVER=${CLIENTS_PER_SERVER:-1}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1800}
TOTAL_SERVERS=$((NUM_GPUS * SERVERS_PER_GPU))
PARALLEL=$((TOTAL_SERVERS * CLIENTS_PER_SERVER))

# ── EVT-Bench driver ─────────────────────────────────────────────────────────
EVT_BENCH_REPO=${EVT_BENCH_REPO:-$HOME/EVT-Bench}
EVT_CONDA_ENV=${EVT_CONDA_ENV:-evt_bench}
EVT_PYTHON=${EVT_PYTHON:-}
TASK_VARIANTS=${TASK_VARIANTS:-dt}
CHUNKS=${CHUNKS:-30}
EVT_JAW_HFOV=${EVT_JAW_HFOV:-}
EVT_JAW_HEIGHT=${EVT_JAW_HEIGHT:-}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=${OUTPUT_ROOT:-$EVT_BENCH_REPO/exp_results/lightnav_${TIMESTAMP}}
LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/logs}

# ── paths (this repo) ────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SH="$REPO_ROOT/scripts/start_servers.sh"
PATCH_PY="$REPO_ROOT/evt_bench/patch_task_config.py"
PID_FILE="$REPO_ROOT/.servers.pids"      # lines: "<pid> <port> <gpu>"
READY_DIR="$REPO_ROOT/.servers_ready"    # one port<PORT>.ready per server

# ── 0. sanity checks ─────────────────────────────────────────────────────────
[ -d "$MODEL_PATH" ] || { echo "$TAG ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
[ -z "$ACTION_TOKENIZER_BUNDLE" ] || [ -d "$ACTION_TOKENIZER_BUNDLE" ] || {
    echo "$TAG ERROR: ACTION_TOKENIZER_BUNDLE does not exist: $ACTION_TOKENIZER_BUNDLE" >&2; exit 1; }
[ -f "$START_SH" ] || { echo "$TAG ERROR: missing $START_SH" >&2; exit 1; }
[ -d "$EVT_BENCH_REPO" ] || {
    echo "$TAG ERROR: EVT_BENCH_REPO does not exist: $EVT_BENCH_REPO" >&2
    echo "$TAG clone https://github.com/wsakobe/TrackVLA and set EVT_BENCH_REPO" >&2
    exit 1; }
# run.py is executed with cwd = EVT-Bench root, so every path handed to it (and
# every path used after the cd below) must be absolute.
EVT_BENCH_REPO="$(cd "$EVT_BENCH_REPO" && pwd)"
[ -f "$EVT_BENCH_REPO/run.py" ] || { echo "$TAG ERROR: $EVT_BENCH_REPO/run.py not found" >&2; exit 1; }
[ -f "$EVT_BENCH_REPO/trackvla_client_agent.py" ] || {
    echo "$TAG ERROR: $EVT_BENCH_REPO/trackvla_client_agent.py not found" >&2
    echo "$TAG copy $REPO_ROOT/evt_bench/trackvla_client_agent.py next to run.py" >&2
    exit 1; }
grep -q "trackvla_client_agent" "$EVT_BENCH_REPO/run.py" || {
    echo "$TAG ERROR: run.py has no 'trackvla' branch" >&2
    echo "$TAG apply it with: (cd $EVT_BENCH_REPO && git apply $REPO_ROOT/evt_bench/run_py.patch)" >&2
    exit 1; }
for V in $TASK_VARIANTS; do
    CFG="$EVT_BENCH_REPO/habitat-lab/habitat/config/benchmark/nav/track/track_infer_${V}.yaml"
    [ -f "$CFG" ] || { echo "$TAG ERROR: task config not found: $CFG (TASK_VARIANTS must be from: dt stt at)" >&2; exit 1; }
done
[ "$CHUNKS" -eq 30 ] || echo "$TAG WARN: CHUNKS=$CHUNKS differs from EVT-Bench's 30 shards; numbers will not be comparable."

# Resolve the EVT-Bench interpreter. `conda activate` only prepends PATH, so
# always use the absolute $CONDA_PREFIX/bin/python afterwards.
if [ -z "$EVT_PYTHON" ]; then
    CONDA_SH="${CONDA_SH:-}"
    if [ -z "$CONDA_SH" ]; then
        for candidate in \
            "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}" \
            "$HOME/miniconda3/etc/profile.d/conda.sh" \
            "$HOME/anaconda3/etc/profile.d/conda.sh" \
            "$HOME/miniforge3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
            if [ -n "$candidate" ] && [ -f "$candidate" ]; then CONDA_SH="$candidate"; break; fi
        done
    fi
    [ -n "$CONDA_SH" ] || {
        echo "$TAG ERROR: cannot locate conda.sh; set CONDA_SH=/path/to/etc/profile.d/conda.sh or EVT_PYTHON=/path/to/python" >&2
        exit 1; }
    set +u   # conda's activate scripts reference unset variables
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$EVT_CONDA_ENV" || {
        echo "$TAG ERROR: conda activate $EVT_CONDA_ENV failed; available envs:" >&2
        conda env list >&2
        exit 1; }
    set -u
    EVT_PYTHON="$CONDA_PREFIX/bin/python"
    # Let habitat-sim find the conda env's own CUDA/GL libraries first, even if a
    # torch virtualenv left its libraries on LD_LIBRARY_PATH.
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
[ -x "$EVT_PYTHON" ] || { echo "$TAG ERROR: EVT_PYTHON is not executable: $EVT_PYTHON" >&2; exit 1; }
"$EVT_PYTHON" -c "import habitat, habitat_sim, magnum" 2>/dev/null || {
    echo "$TAG ERROR: $EVT_PYTHON cannot import habitat/habitat_sim/magnum (wrong env?)" >&2; exit 1; }
"$EVT_PYTHON" -c "import websocket" 2>/dev/null || {
    echo "$TAG ERROR: $EVT_PYTHON lacks websocket-client: $EVT_PYTHON -m pip install websocket-client" >&2; exit 1; }
if [ -n "$EVT_JAW_HFOV" ] || [ -n "$EVT_JAW_HEIGHT" ]; then
    "$EVT_PYTHON" -c "import yaml" 2>/dev/null || {
        echo "$TAG ERROR: $EVT_PYTHON lacks PyYAML (needed for the jaw camera patch)" >&2; exit 1; }
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"   # absolute: used after cd to the EVT root
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
echo "$TAG model=$MODEL_PATH K=$K H=$HORIZON backend=$BACKEND"
echo "$TAG servers=$TOTAL_SERVERS (gpus=$NUM_GPUS x $SERVERS_PER_GPU) parallel_clients=$PARALLEL chunks=$CHUNKS variants='$TASK_VARIANTS'"
echo "$TAG evt_repo=$EVT_BENCH_REPO python=$EVT_PYTHON"
echo "$TAG output=$OUTPUT_ROOT"

# ── 1. cleanup trap ──────────────────────────────────────────────────────────
cleanup() {
    echo
    echo "$TAG cleaning up servers..."
    if [ -f "$PID_FILE" ]; then
        while read -r pid _ _; do
            [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true
        done <"$PID_FILE"
        sleep 1
        while read -r pid _ _; do
            [ -n "${pid:-}" ] && kill -9 "$pid" 2>/dev/null || true
        done <"$PID_FILE"
    fi
    rm -f "$READY_DIR"/*.ready 2>/dev/null || true
    echo "$TAG done."
}
trap cleanup EXIT INT TERM

# ── 2. start servers ─────────────────────────────────────────────────────────
echo "$TAG starting $TOTAL_SERVERS servers..."
NUM_GPUS=$NUM_GPUS SERVERS_PER_GPU=$SERVERS_PER_GPU BASE_PORT=$BASE_PORT \
    MODEL_PATH="$MODEL_PATH" ACTION_TOKENIZER_BUNDLE="$ACTION_TOKENIZER_BUNDLE" \
    TASK=tracking BACKEND=$BACKEND \
    LOG_DIR="$LOG_DIR/servers" \
    bash "$START_SH"

# ── 3. wait for ready ────────────────────────────────────────────────────────
# `find` (not `ls *.ready | wc -l`): under pipefail a glob with no match fails.
echo "$TAG waiting for $TOTAL_SERVERS servers to be ready (timeout ${READY_TIMEOUT_S}s)..."
DEADLINE=$((SECONDS + READY_TIMEOUT_S))
LAST_REPORTED=-1
while :; do
    READY_COUNT=$(find "$READY_DIR" -maxdepth 1 -name '*.ready' 2>/dev/null | wc -l)
    if [ "$READY_COUNT" -ne "$LAST_REPORTED" ]; then
        echo "$TAG   $READY_COUNT/$TOTAL_SERVERS ready"
        LAST_REPORTED=$READY_COUNT
    fi
    if [ "$READY_COUNT" -ge "$TOTAL_SERVERS" ]; then
        echo "$TAG all $TOTAL_SERVERS servers ready."
        break
    fi
    # Fail fast when a server process has already exited without becoming ready.
    if [ -f "$PID_FILE" ]; then
        while read -r pid port _; do
            [ -n "${pid:-}" ] || continue
            if ! kill -0 "$pid" 2>/dev/null && [ ! -f "$READY_DIR/port${port}.ready" ]; then
                echo "$TAG ERROR: server on port $port (pid $pid) exited before becoming ready; log tail:" >&2
                tail -30 "$LOG_DIR"/servers/*port"${port}"*.log 2>/dev/null >&2 || true
                exit 1
            fi
        done <"$PID_FILE"
    fi
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
        echo "$TAG ERROR: only $READY_COUNT/$TOTAL_SERVERS ready after ${READY_TIMEOUT_S}s; server log tails:" >&2
        for f in "$LOG_DIR"/servers/*.log; do
            [ -f "$f" ] || continue
            echo "--- $f ---" >&2
            tail -15 "$f" >&2 || true
        done
        exit 1
    fi
    sleep 3
done

# ── 4. run EVT-Bench chunks ──────────────────────────────────────────────────
cd "$EVT_BENCH_REPO"
FAILED_CHUNKS=0
for V in $TASK_VARIANTS; do
    TASK_CONFIG="habitat-lab/habitat/config/benchmark/nav/track/track_infer_${V}.yaml"
    if [ -n "$EVT_JAW_HFOV" ] || [ -n "$EVT_JAW_HEIGHT" ]; then
        # run.py drops Hydra overrides for the trackvla branch, so write a patched
        # copy of the task config and pass its absolute path instead.
        PATCHED="$OUTPUT_ROOT/track_infer_${V}_hfov${EVT_JAW_HFOV:-def}_h${EVT_JAW_HEIGHT:-def}.yaml"
        patch_args=()
        [ -n "$EVT_JAW_HFOV" ] && patch_args+=(--hfov "$EVT_JAW_HFOV")
        [ -n "$EVT_JAW_HEIGHT" ] && patch_args+=(--height "$EVT_JAW_HEIGHT")
        "$EVT_PYTHON" "$PATCH_PY" "$EVT_BENCH_REPO/$TASK_CONFIG" "$PATCHED" "${patch_args[@]}"
        [ -s "$PATCHED" ] || { echo "$TAG ERROR: failed to write $PATCHED" >&2; exit 1; }
        TASK_CONFIG="$PATCHED"
    fi
    SAVE_PATH="$OUTPUT_ROOT/$V"
    mkdir -p "$SAVE_PATH"
    echo "$TAG variant=$V config=$TASK_CONFIG save=$SAVE_PATH"

    IDX=0
    while [ "$IDX" -lt "$CHUNKS" ]; do
        # one wave: up to PARALLEL chunks, chunk -> server IDX % TOTAL_SERVERS
        pids=()
        i=0
        while [ "$i" -lt "$PARALLEL" ] && [ "$IDX" -lt "$CHUNKS" ]; do
            SRV=$((IDX % TOTAL_SERVERS))
            PORT=$((BASE_PORT + SRV))
            GPU=$((SRV / SERVERS_PER_GPU))   # render on the same GPU as the server
            CHUNK_LOG="$LOG_DIR/eval_${V}_chunk_${IDX}.log"
            echo "$TAG   chunk $IDX -> ws://localhost:$PORT (gpu $GPU)  log=$CHUNK_LOG"
            CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="habitat-lab" "$EVT_PYTHON" run.py \
                --split-num "$CHUNKS" \
                --split-id "$IDX" \
                --exp-config "$TASK_CONFIG" \
                --run-type eval \
                --model-name trackvla \
                --model-path "ws://localhost:$PORT" \
                --save-path "$SAVE_PATH" \
                >"$CHUNK_LOG" 2>&1 &
            pids+=("$!:$IDX")
            # NOTE: not `((IDX++))`: it returns 1 when IDX was 0 and kills `set -e`.
            IDX=$((IDX + 1))
            i=$((i + 1))
        done
        for entry in "${pids[@]}"; do
            pid=${entry%%:*}
            chunk=${entry##*:}
            if ! wait "$pid"; then
                echo "$TAG WARN: chunk $chunk failed (see $LOG_DIR/eval_${V}_chunk_${chunk}.log)" >&2
                FAILED_CHUNKS=$((FAILED_CHUNKS + 1))
            fi
        done
    done

    # ── 5. aggregate (must run from the EVT-Bench root: reads track_episode_step/
    #      relatively and writes following_info.json into cwd) ──────────────────
    N_RESULTS=$(find "$SAVE_PATH" -name '*.json' ! -name '*_info.json' 2>/dev/null | wc -l)
    if [ "$N_RESULTS" -eq 0 ]; then
        echo "$TAG WARN: no finished episodes under $SAVE_PATH; skipping analyze_results.py" >&2
        continue
    fi
    echo "$TAG variant=$V: $N_RESULTS episodes; aggregating..."
    "$EVT_PYTHON" analyze_results.py --path "$SAVE_PATH" | tee "$OUTPUT_ROOT/metrics_${V}.txt"
done

echo "$TAG all variants finished. results under $OUTPUT_ROOT"
if [ "$FAILED_CHUNKS" -gt 0 ]; then
    echo "$TAG WARN: $FAILED_CHUNKS chunk(s) failed; metrics above are incomplete." >&2
    exit 1
fi
