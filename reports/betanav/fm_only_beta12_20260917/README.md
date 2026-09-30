# FM-only, Beta(1,2), eight GPUs — 2026-09-17

User-requested experiment: increase high-noise training exposure using
`tau ~ Beta(1,2)` and train with `loss = 1.0 × FM` on GPUs 0–7.

## Initialization and data

- Original frozen VLM: `/mnt/cfs/Models/Light` (eval/no_grad, BF16).
- Fresh FM expert and context projection; no previous BetaNav weights or Adam state.
- Same full R2R LeRobot dataset and normalization manifest as the completed uniform run.
- Continuous 30-step local SE(2) targets and valid masks, prompt-only conditioning.
- No RVQ supervision or validation split. FM-only still has no learned STOP branch.

## Sampling

`X_tau = (1-tau) * noise + tau * GT`: smaller tau means a larger noise coefficient.
One independent scalar time per action chunk, shared over its 30 action steps.
`Beta(1,2)` has mean tau=1/3 and P(tau <= 0.3)=0.51. Gaussian action noise is
sampled separately. Samples use each rank's Torch RNG, which the existing full
checkpoints restore. Sampling settings enter both model metadata and the strict
resume contract. Legacy missing fields mean uniform sampling.

GPU sampler checks: 100,000 samples per GPU under BF16 autocast, FP32 sampled times.
All eight GPUs passed distribution, finite-range and exact RNG replay checks.
Observed high-noise fraction: 50.81–51.15%; see [gpu_sampling_check.json](gpu_sampling_check.json).

## Training configuration

Config: `configs/betanav/r2r_fm_only_beta12.yaml`.

| Setting | Value |
|---|---|
| Time distribution | Beta(1,2) |
| FM loss weight | 1.0 |
| Per-rank / global batch | 64 / 512 |
| Gradient accumulation | 1 |
| Epochs / target updates | 3 / 3,699 |
| Updates per epoch | 1,233; final global batch=464 |
| FM LR / warmup | 1e-4 / 1,000 updates |
| LR schedule | cosine, floor ratio=0.1 |
| Optimizer | DDP + ZeRO-1 AdamW, betas=(0.9,0.95), weight decay=0.01 |
| Expert | 6 layers, width 512, 8 heads, FFN 2048 |
| CPU workers / prefetch | 4 per GPU / 1 |
| Complete checkpoints | every 1,000 steps and at completion; keep last 3 |
| Permanent model archive | every 3,000 steps |

Output: `/mnt/cfs/Models/beta_nav/r2r_fm_only_beta12_frozen_lightnav_20260917`.
At normal completion, the final full checkpoint is `checkpoints/step-00003699`,
with a permanent model-only archive at `model_checkpoints/step-00003000`.
The full states include FM Adam/RNG/data cursor. Model-only archives are not exact resumes.

The launch directory is recorded in [launch_dir.txt](launch_dir.txt). It holds the
immutable launch config, source hashes, PIDs, training log and five-second GPU
telemetry. Actual losses are recorded in the output directory's `metrics.jsonl`.
Supervisor status is recorded in the launch directory's `launch.json`.

Verification before launch: 822 CPU tests passed, 7 skipped, 2 GPU tests deselected;
`make check` passed its syntax-compile fallback (ruff unavailable); `git diff --check`
passed. Existing batch=64 full-model and eight-GPU capacity measurements are in
`reports/betanav/fm_only_20260916/`.

Comparison note: this run changes both the time-sampling distribution and the
loss multiplier relative to the completed 2026-09-16 uniform/2×FM run. Training
loss values therefore should not be treated as an isolated sampler comparison.

Startup verified through step 10: total loss=FM loss=2.95106, CE=0, VLM gradient
norm=0. Sampled update time=6.988 seconds, throughput=73.27 samples/s, maximum
data wait=0.000199 seconds. All eight workers were alive; live GPU utilization
sampled at 86–95%. No nonfinite loss, OOM or training error appeared during this
startup observation. See [launch_verification.json](launch_verification.json).
