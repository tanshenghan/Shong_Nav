#!/usr/bin/env bash
# Container entrypoint for the lightnav WebSocket server.
#
#   1) Resolve LD_LIBRARY_PATH so the torch wheels find their own bundled nvJitLink
#      over any older system libnvJitLink.so.12. Without it `import torch` can fail with
#      `undefined symbol: __nvJitLinkGetErrorLogSize_12_9` (or the matching minor).
#      (Same fix as scripts/start_servers.sh, applied here for every container start.)
#   2) Build the server argument list from environment variables (set via
#      `docker run -e ...`) so the image needs no rebuild to retarget a checkpoint.
#
# Any extra arguments passed to `docker run` are appended verbatim.
set -euo pipefail

VENV="${VIRTUAL_ENV:-/app/.venv}"
NV_LIBS=$(echo "$VENV"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr ' ' ':')
if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="$NV_LIBS:${LD_LIBRARY_PATH:-}"
fi

: "${MODEL_PATH:?MODEL_PATH must be set (checkpoint mount path)}"

args=(
    lightnav-serve
    --model_path "$MODEL_PATH"
    --task "${TASK:-tracking}"
    --backend "${BACKEND:-vllm_local}"
    --gpu_memory_utilization "${GPU_MEM_UTIL:-0.85}"
    --max_new_tokens "${MAX_NEW_TOKENS:-8}"
    --host "${HOST:-0.0.0.0}"
    --port "${PORT:-8050}"
    --max_batch_size "${MAX_BATCH_SIZE:-8}"
    --max_wait_ms "${MAX_WAIT_MS:-8}"
)

# Action decoder: resolved from the checkpoint itself (eval_config.json +
# action_tokenizer/); set ACTION_TOKENIZER_BUNDLE only to override it.
if [ -n "${ACTION_TOKENIZER_BUNDLE:-}" ]; then
    args+=(--action_tokenizer_bundle "$ACTION_TOKENIZER_BUNDLE")
fi

if [ -n "${NUM_HISTORY_FRAMES:-}" ]; then
    args+=(--num_history_frames "$NUM_HISTORY_FRAMES")
fi

if [ -n "${POOL_SPATIAL:-}" ]; then
    args+=(--pool_spatial "$POOL_SPATIAL")
fi
if [ -n "${ASPECT_MODE:-}" ]; then
    args+=(--aspect_mode "$ASPECT_MODE")
fi

# Optional episode recording (docs/VISUALIZATION.md): RECORD_DIR turns it on; the
# other knobs default inside the server. Mount RECORD_DIR as a volume and render the
# episodes afterwards with `lightnav-render <RECORD_DIR>`.
if [ -n "${RECORD_DIR:-}" ]; then
    args+=(--record_dir "$RECORD_DIR")
    if [ -n "${RECORD_FPS:-}" ]; then
        args+=(--record_fps "$RECORD_FPS")
    fi
    if [ -n "${RECORD_TIMELINE:-}" ]; then
        args+=(--record_timeline "$RECORD_TIMELINE")
    fi
    case "${RECORD_IMAGES:-1}" in
        0|false|no|off) args+=(--no_record_images) ;;
    esac
fi
# Camera parameters of the CLIENT camera; only the trajectory overlay of the rendered
# video uses them.
if [ -n "${CAM_HFOV_DEG:-}" ]; then
    args+=(--cam_hfov_deg "$CAM_HFOV_DEG")
fi
if [ -n "${CAM_HEIGHT:-}" ]; then
    args+=(--cam_height "$CAM_HEIGHT")
fi
if [ -n "${TRAJ_FORWARD_OFFSET:-}" ]; then
    args+=(--traj_forward_offset "$TRAJ_FORWARD_OFFSET")
fi
if [ -n "${WAYPOINT_DT_S:-}" ]; then
    args+=(--waypoint_dt_s "$WAYPOINT_DT_S")
fi

exec "${args[@]}" "$@"
