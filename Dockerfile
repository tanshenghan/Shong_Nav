# lightnav WebSocket inference server: optional GPU image for the in-process
# vLLM backend (`--backend vllm_local`).
#
# The model checkpoint and the trajectory vocabulary are NOT baked in; mount them at
# runtime (e.g. `-v /path/to/models:/models`). Only code + dependencies live here, so
# the image is independent of checkpoint iterations.
#
# Verified stack (see pyproject.toml / README): vllm==0.19.1, transformers==5.8.0,
# torch 2.10.0, python 3.11. `pip`/`uv` resolve the cu12.8 torch wheel from PyPI; the base
# image below only supplies the NVIDIA container plumbing, so the two need not match. On
# sm_103 parts (B300 / B30Z) the cu12.8 wheel cannot JIT for the device and the `hf`
# backend fails -- build with the cu12.9 index there (see the README installation notes).
#
# Build:
#   docker build -t lightnav:latest .
# Run (flat trajectory vocabulary):
#   docker run --gpus all -p 8050:8050 -v /path/to/models:/models \
#       -e MODEL_PATH=/models/hf_ckpt -e TRAJ_VOCAB_PATH=/models/traj_vocab \
#       -e K=4096 -e HORIZON=10 lightnav:latest
# Run (RVQ action tokenizer bundle):
#   docker run --gpus all -p 8050:8050 -v /path/to/models:/models \
#       -e MODEL_PATH=/models/hf_ckpt -e ACTION_TOKENIZER_BUNDLE=/models/action_tokenizer \
#       lightnav:latest

# BASE: nvidia/cuda *-base* (not -devel, not -runtime). The cu12.x torch/vllm wheels
# bundle the entire CUDA userspace themselves (cudart, cublas, cudnn, nccl, nvrtc,
# nvjitlink, ... arrive via the `nvidia-*-cu12` pip wheels), so a -devel/-runtime base
# would ship a second, unused copy (several GB). `-base` is just the NVIDIA container
# plumbing (NVIDIA_DRIVER_CAPABILITIES + CUDA_HOME).
#
# Triton (bundled with torch) ships its own ptxas + CUDA headers and JIT-compiles its
# launcher stub with the host C compiler at runtime; `build-essential` below covers
# that. If a JIT still fails on a missing cuda.h/ptxas on a new GPU generation, add
# the `nvidia-cuda-nvcc-cu12` wheel rather than switching to a -devel base.
ARG CUDA_IMAGE=nvidia/cuda:12.9.1-base-ubuntu22.04
FROM ${CUDA_IMAGE}

# uv brings its own pinned CPython 3.11 -- no system python needed.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# build-essential: Triton's runtime launcher-stub compile needs a host C compiler.
# ca-certificates: TLS for package downloads. The `video` extra uses headless
# OpenCV + imageio/ffmpeg wheels, so no libgl1 is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Two layers for cache hygiene:
#   1. dependencies only (--no-install-project), keyed on pyproject.toml + README.md.
#      This is the heavy layer (torch, vllm, nvidia-*-cu12 wheels); editing src/ does
#      not invalidate it.
#   2. the project itself, keyed on src/. Cheap.
# --mount=type=cache keeps uv's download/wheel cache on the BuildKit builder across
# builds so the multi-GB CUDA/torch/vllm download happens once.
# No lock file is committed; pass `--frozen` here if you add one (`uv lock`).
COPY pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra vllm --extra video --python 3.11 --no-install-project

COPY src ./src
# This sync installs the project itself: the entrypoint execs the `lightnav-serve`
# console script declared in pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra vllm --extra video --python 3.11

# Non-root runtime user. No chown of /app: the runtime only reads the venv (root-owned
# files are world-readable/executable) and writes nothing here -- logs go to stdout,
# PYTHONDONTWRITEBYTECODE=1 blocks .pyc writes, and torch/vllm caches land in the
# user's $HOME.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Thread caps: small per-step CPU ops (video patchify, embed assembly) thrash when
# fanned across all cores. Read straight from env by the native math libs at process
# init; tune to the container's CPU limit.
ENV OMP_NUM_THREADS=32 \
    MKL_NUM_THREADS=32 \
    OPENBLAS_NUM_THREADS=32 \
    NUMEXPR_NUM_THREADS=32 \
    # vllm_local extracts the in-process ViT from the engine; this requires the model
    # to live in-process, not in a worker subprocess.
    VLLM_ENABLE_V1_MULTIPROCESSING=0

# Server config is driven by env (see docker/entrypoint.sh) so the image needs no
# rebuild to retarget a checkpoint or change K / horizon. Defaults below; override
# per run with `docker run -e ...`. Leave NUM_HISTORY_FRAMES unset to honour the
# checkpoint's eval_config.json.
ENV HOST=0.0.0.0 \
    PORT=8050 \
    TASK=tracking \
    BACKEND=vllm_local \
    K=256 \
    HORIZON=10 \
    GPU_MEM_UTIL=0.85 \
    MAX_NEW_TOKENS=8 \
    MAX_BATCH_SIZE=8 \
    MAX_WAIT_MS=8 \
    MODEL_PATH=/models/hf_ckpt \
    TRAJ_VOCAB_PATH=/models/traj_vocab

EXPOSE 8050

COPY --chown=appuser:appuser docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
