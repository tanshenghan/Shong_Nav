# Beta(1,3) FM-only — delayed eight-GPU training

User requested a new version eight hours after 2026-09-17 09:08:44 UTC.
Earliest dispatch: **2026-09-17 17:08:44 UTC / 2026-09-18 01:08:44 Beijing**.
If Beta(1,2) is still running or any GPU is busy, the queue waits, checking every
30 seconds. It never stops the current training. This is a detached server-side
process, not a Codex automation; the server/container must remain running.

- Frozen original LightNav: `/mnt/cfs/Models/Light`.
- Fresh context projection and FM expert; no resume from Beta(1,2).
- `loss = 1.0 * FM`; no CE/RVQ targets or VLM updates.
- `tau ~ Beta(1,3)`, `X_tau=(1-tau)*noise+tau*GT`.
- Mean tau=0.25; noise coefficient >=0.7 has probability 65.7%.
- GPUs 0–7, batch=64 per GPU, global batch=512, accumulation=1.
- Full R2R train data, three epochs / 3,699 updates, no validation.
- LR=1e-4, warmup=1,000, ZeRO-1 AdamW and BF16, unchanged from Beta(1,2).
- Full checkpoints every 1,000 steps and at completion, newest three retained.
- Permanent model-only checkpoint every 3,000 steps.

Config: `configs/betanav/r2r_fm_only_beta13.yaml`.
Output: `/mnt/cfs/Models/beta_nav/r2r_fm_only_beta13_frozen_lightnav_20260918`.
Final full checkpoint is expected at `checkpoints/step-00003699` after completion.

Prepared launch directory: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/runs/betanav_launch/20260917T091717Z_fm_only_beta13_scheduled_8gpu`.
The frozen `train_config.yaml` and `supervisor.py` are guarded with SHA256 hashes.
`schedule.json` fixes the due time, required GPUs, and predecessor run.
`schedule_state.json` records the queue PID, state and last check; `launch.json`
records the actual training status. `train.log` and `gpu.csv` appear at dispatch;
training metrics will be in the output directory's `metrics.jsonl`.
The queue holds an exclusive file lock and records dispatch intent before starting
its supervisor, preventing duplicate launches. A failed or ambiguous dispatch is
not retried automatically. No eight-hour foreground wait is required.

Read-only readiness check:

```bash
.venv-train/bin/python -m lightnav.training.scheduled_launch --schedule /mnt/pfs/tanshenghan/vln4project/LightNav-0/runs/betanav_launch/20260917T091717Z_fm_only_beta13_scheduled_8gpu/schedule.json --check
```

See `schedule_verification.json` for the verified background process and state.

Verified host queue PID: 464849, state `waiting_for_time`. 837 CPU tests passed, 7 skipped, 2 GPU tests deselected. `make check` passed the syntax-compile fallback (ruff unavailable).
The restricted process view initially hid the live queue. A second launch attempt was rejected by the exclusive lock; its BlockingIOError is recorded in queue.log. Host-side verification confirmed the original queue remained alive and continued updating state; no duplicate job was scheduled.
