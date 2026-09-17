# Frozen original LightNav + fresh FM expert — 2026-09-16

User choices: initialize from the original `/mnt/cfs/Models/Light`; reinitialize
the FM expert; train three complete full-data epochs on GPU 0–7. No prior BetaNav
checkpoint or Adam state is loaded.

## Model and supervision

- `training_mode: fm_only`; frozen VLM in eval/no_grad, BF16 weights/compute.
- Train only context projection and FM expert: 36,517,891 parameters, FP32 master weights.
- Prompt-only VLM inputs; no teacher-forced answers, CE/LM-head projection, or RVQ codebooks/targets.
- Original 30-step continuous SE(2) deltas and valid masks; `loss = 2 × FM`.
- Preserve terminal/tail masks. No STOP predictor is trained in this experiment.
- Empty-supervision ranks retain zero-gradient expert hooks for correct DDP reduction;
  globally empty updates skip AdamW and weight decay.

## Capacity measurements

Real episode 4112, t=152, 82 historical RGB frames, 1,394 prompt tokens, 30 valid
future actions. Full pretrained weights, forward/backward/Adam included. Repeated
cached CPU input; these measurements exclude the production data pipeline.

| GPUs | Per-GPU batch | Global batch | Peak allocated GiB/card | Peak reserved GiB/card | Samples/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 16.06 | 18.50 | 5.16 |
| 1 | 16 | 16 | 23.35 | 28.76 | 5.24 |
| 1 | 32 | 32 | 37.92 | 48.69 | 5.24 |
| 1 | 48 | 48 | 52.50 | 68.62 | 5.16 |
| 1 | 64 | 64 | 67.07 | 88.54 | 5.18 |
| 8 | 64 | 512 | 66.97 | 88.31 | 39.34 |

Reports: [single GPU](profile_long_1gpu.json), [eight GPUs](profile_long_8gpu_b64.json).
GPU telemetry is in the corresponding `.gpu.csv` files. Larger batches gave little
additional compute throughput; batch 64 uses substantial GPU memory while preserving
space for runtime buffers and differences in instruction/history length. Reserved
memory includes reusable CUDA allocator cache; it is not all live activation memory.

## Training

Config: `configs/betanav/r2r_fm_only.yaml`. Per-GPU batch=64, accumulation=1,
global batch=512, CPU workers=4/rank, prefetch=1. Other FM optimization settings
unchanged: LR=1e-4, warmup=1,000 updates, cosine floor=0.1, clip=1,
AdamW betas=[0.9,0.95], weight decay=0.01. No validation.

631,244 frame anchors / 10,819 episodes; sampler pads four duplicate frames per
epoch across eight ranks. Each rank sees 78,906 samples, so each epoch has 1,233
updates (last global batch=464). Three epochs: **3,699 updates**, 1,893,744 sample
presentations, including 12 sampler padding duplicates.

Output: `/mnt/cfs/Models/beta_nav/r2r_fm_only_frozen_lightnav_20260916`.
Every 1,000 updates save full model/FM Adam/RNG/data cursor asynchronously and keep
the latest three. Step 3,000 additionally gets a permanent model-only archive.
Final step 3,699 saves full state. Original experiment remains separate.

Launch directory is recorded in [launch_dir.txt](launch_dir.txt); it contains the
frozen training config, source hashes, supervisor/launcher PIDs, `train.log`, and
5-second `gpu.csv`. The output directory's `metrics.jsonl` records actual training.

Validation before launch: full CPU suite 784 passed, 5 skipped, 2 GPU tests deselected;
an additional two-rank FM-only async checkpoint/ZeRO exact-resume test passed.
`make check` passed its syntax-compile fallback (ruff unavailable). Eight-GPU full
model capacity test completed four finite updates and exited normally.

Formal training verified through step 10: loss=5.45433, raw FM=2.72716, CE=0,
VLM LR/gradient norm=0. At this sampled update: 6.949 seconds, 73.68 samples/s,
maximum per-rank data wait=0.000224 seconds. Startup step 1 included 27.28 seconds
of loader initialization and is not a steady-state speed measurement. One live
GPU sample showed 85–89% utilization and 52.3–78.6 GiB reported memory usage;
these vary with the random batch's history lengths. No OOM/nonfinite/error appeared
during the startup observation. [Verified configuration and metrics](launch_verification.json).
