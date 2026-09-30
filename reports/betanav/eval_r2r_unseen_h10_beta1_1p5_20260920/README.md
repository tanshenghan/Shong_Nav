# Beta(1,1.5), FM horizon 10: R2R val_unseen evaluation — 2026-09-20

Full evaluation requested by the user. STOP protocol explicitly selected: frozen original LightNav RVQ STOP; movement from the trained FM expert.

## Model and benchmark

- Base model: `/mnt/cfs/Models/Light`.
- Checkpoint: `/mnt/cfs/Models/beta_nav/r2r_fm_only_h10_beta1_1p5_frozen_lightnav_20260918/checkpoints/step-00003699` (step 3,699; FM horizon 10; training Beta(1,1.5)).
- Backend: HF, Torch 2.10.0+cu129. FM Euler integration: 10 iterations; deterministic per-episode/step seed 42.
- R2R VLN-CE val_unseen: all 1,839 episodes across 11 scenes. Scene geometry and NDTW GT were checked locally; no downloads.
- Eight disjoint shards on GPU 0–7, one simulator and one model client per card. Shards 0–6 have 229 episodes each; shard 7 has 236.
- Reference Habitat camera: 480×270, HFOV 120°, camera height 0.88 m; checkpoint preprocessing uses 448×256 with its SlowFast history.
- Max episode budget 500, final-budget forced STOP enabled; success distance 3 m. No privileged oracle STOP.
- `fm_only_stop_mode=rvq`, `fm_stop_mode=baseline`: the frozen VLM uses its original RVQ prompt for STOP and a separate continuous prompt for FM. Existing baseline rules also stop on malformed RVQ output, zero RVQ trajectory or near-zero FM velocity.
- Metrics: SR, oracle success, SPL, NDTW, navigation error and episode lengths. Oracle success is a reported metric, not an action-selection policy.

## Outputs

- Evaluation directory: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/outputs/betanav_eval/20260920T011359Z_h10_beta1_1p5_r2r_unseen`.
- Merged final results: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/outputs/betanav_eval/20260920T011359Z_h10_beta1_1p5_r2r_unseen/results.jsonl` and `/mnt/pfs/tanshenghan/vln4project/LightNav-0/outputs/betanav_eval/20260920T011359Z_h10_beta1_1p5_r2r_unseen/summary.json`.
- Per-shard live results: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/outputs/betanav_eval/20260920T011359Z_h10_beta1_1p5_r2r_unseen/shard_0/results.jsonl` through `shard_7/results.jsonl`.
- Logs: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/outputs/betanav_eval/20260920T011359Z_h10_beta1_1p5_r2r_unseen/logs`.
- Launch metadata/supervisor: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/runs/habitat_eval_launch/20260920T011359Z_h10_beta1_1p5_r2r_unseen` (`launch.json`, `eval.log`, `supervisor.log`).
- Supervisor verifies the merged episode keys against the source split, with no missing, extra or duplicate episode. Successful coverage is saved to `coverage.json`; a partial run is not marked complete.
- The detached supervisor records the exit status and remains active after this conversation turn; the server/container must stay online.

## Preflight

- Habitat and client environments import successfully, independent Python 3.9 / Python 3.11 environments.
- One real 20-step FM smoke episode completed successfully, and each generated waypoint array was 10×3.
- The smoke used fixed-budget stopping before the user selected RVQ STOP. It is kept separately and never merged into formal evaluation.
- Smoke output: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/outputs/betanav_eval/20260920T011359Z_h10_beta1_1p5_r2r_unseen_smoke20`.
- Formal RVQ STOP startup will be checked through actual episode results. No training/source code changes were needed for evaluation.

## Verified formal startup

At 2026-09-20T01:19:26.624172+00:00, all eight servers and eight clients were alive. Every client loaded the selected step-3,699 checkpoint and used the user-selected RVQ STOP protocol. 24 formal episodes had completed, with no observed traceback or CUDA OOM. The run is still in progress; these early records are not the full-split result. See `startup_verification.json`.
