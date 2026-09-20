# Development

```bash
pip install -e ".[test]"   # pytest + pytest-asyncio; not part of the runtime extras
make check                 # ruff check .
make test                  # CPU test suite: pytest -m "not gpu"
```

Tests under `tests/` are CPU-only (fake engines, synthetic tokens) and cover token decoding,
the sample builders and SlowFast layout, the micro-batch scheduler, the wire protocol, the
Habitat client loop, the EVT-Bench client's response parsing and the visualisation module
(video-encoding tests skip without the `video` extra).

Notes:

- **Frame convention.** `observe()` and the server take HWC `uint8` RGB frames at any
  resolution; they are resized to the checkpoint's `video_size` and normalised to `[-1, 1]`
  internally. Pointing pixels are reported in the *client's* frame size.
- **`hf` attention.** Defaults to `sdpa`; override with `LIGHTNAV_ATTN=flash_attention_2`.
- **vLLM version.** `inference/vllm_utils.py` patches vLLM internals specific to
  **vLLM 0.19.x**; a runtime guard refuses other versions (`LIGHTNAV_SKIP_VERSION_GUARD=1`
  bypasses it and is unsupported).
- **CUTLASS DSL.** `nvidia-cutlass-dsl` is pinned in the `vllm` extra. vLLM 0.19.1 asks
  only for `>=4.4.0.dev1`, which lets pip resolve a dev build that dropped
  `cutlass.cute.core.ThrMma`; every `quack-kernels` release up to 0.6.4 still imports that
  symbol, so an unpinned install fails with `AttributeError: module 'cutlass.cute.core' has
  no attribute 'ThrMma'` while the engine starts.
- **GPU architecture.** The stock cu12.8 torch wheel cannot JIT for `sm_103`
  (B300 / B30Z): the `hf` backend fails in transformers' Qwen3-VL vision tower with
  `nvrtc: error: invalid value for --gpu-architecture`. Use the cu12.9 wheels there (see
  the README installation notes); `vllm_local` works either way.

---

## GPU smoke test

The CPU suite cannot exercise the model itself. On a machine with a GPU and a checkpoint,
`scripts/smoke_gpu.sh` runs every real-hardware path once at minimal scale and prints a
PASS / FAIL / SKIP table:

```bash
MODEL_PATH=checkpoints/LightNav-0 \
    HABITAT_CONFIG=habitat_server/configs/vlnce_r2r.yaml EVT_BENCH_REPO=$HOME/EVT-Bench \
    bash scripts/smoke_gpu.sh
```

`ACTION_TOKENIZER_BUNDLE` is only needed for a checkpoint that does not ship its own
decoder.

Steps: `lightnav-predict` with the `hf` and `vllm_local` backends; `lightnav-serve` +
`lightnav-ws-client` with `--record_dir`, then `lightnav-render`; one Habitat episode via
`scripts/eval_habitat.sh` (skipped without `HABITAT_CONFIG`); one EVT-Bench shard against
the running server (skipped without `EVT_BENCH_REPO`). Logs and outputs land in
`output/smoke_<timestamp>/`.
