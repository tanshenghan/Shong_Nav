# FM-only: ten future actions, three Beta samplers — 2026-09-18

User requested FM output `10×3` and separate eight-GPU training runs with Beta(1,1.5), Beta(1,2), and Beta(1,3).

## Trajectory contract

- Training config `fm_horizon: 10` is propagated to both the LeRobot reader and `BetaModelConfig.horizon`.
- Validate the original stored 30-step targets, then take the first ten consecutive local SE(2) increments and valid-mask entries.
- Target/action shape `[B,10,3]`; axes are forward metres, left metres, counterclockwise radians. Composed local waypoints also have shape `[B,10,3]`.
- These are transitions t→t+1 through t+9→t+10, with no interpolation, stride or RVQ reconstruction.
- Incomplete episode tails keep their real mask. No frames are removed or source Parquet files modified.
- `num_flow_steps: 10` remains the independent number of Euler integration iterations.
- Resume identity includes `fm_horizon`; missing legacy values mean 30, and 30-step states cannot be resumed as 10-step models.

## Common settings

- Original frozen VLM `/mnt/cfs/Models/Light`; fresh FM context projection/expert for every experiment. No warm-start from another Beta model.
- FM-only, no CE/RVQ, `lambda_fm: 1.0`, seed 42.
- GPUs 0–7, batch 64 per GPU, global batch 512, accumulation 1.
- Full R2R data: 631,244 frames, three epochs, 1,233 updates per epoch / 3,699 total.
- Same normalization manifest and SlowFast image history as previous experiments.
- AdamW ZeRO-1, BF16, FM LR=1e-4, warmup=1,000, cosine floor=0.1.
- Full checkpoint every 1,000 updates and at completion; newest three retained. Permanent model-only archive every 3,000.
- Approximately ten hours per model is an estimate, not a wall-time cutoff. The epoch target determines completion.

## Sequential launch order

### 1. Beta(1,1.5)

- Config: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/configs/betanav/r2r_fm_only_h10_beta1_1p5.yaml`
- Output (created at actual training start): `/mnt/cfs/Models/beta_nav/r2r_fm_only_h10_beta1_1p5_frozen_lightnav_20260918`
- Launch directory: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/runs/betanav_launch/20260918T072810Z_fm_only_h10_beta1_1p5_8gpu`
- Queue PID at registration: 980355
- Initial state: `waiting_for_gpus`

### 2. Beta(1,2)

- Config: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/configs/betanav/r2r_fm_only_h10_beta12.yaml`
- Output (created at actual training start): `/mnt/cfs/Models/beta_nav/r2r_fm_only_h10_beta12_frozen_lightnav_20260918`
- Launch directory: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/runs/betanav_launch/20260918T072810Z_fm_only_h10_beta12_8gpu`
- Queue PID at registration: 980356
- Initial state: `waiting_for_previous_run`

### 3. Beta(1,3)

- Config: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/configs/betanav/r2r_fm_only_h10_beta13.yaml`
- Output (created at actual training start): `/mnt/cfs/Models/beta_nav/r2r_fm_only_h10_beta13_frozen_lightnav_20260918`
- Launch directory: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/runs/betanav_launch/20260918T072810Z_fm_only_h10_beta13_8gpu`
- Queue PID at registration: 980357
- Initial state: `waiting_for_previous_run`

Every queue process has been verified alive in the host process view. The first waits for the currently running JanusVLN GPU jobs; the other two wait for their predecessor to exit, then for all eight GPUs to be free. No existing GPU task was interrupted.

The queue checks every 30 seconds. It is a detached server process, so the server/container must remain online. It records status in each launch directory's `schedule_state.json`; training status goes in `launch.json`. `train.log` and `gpu.csv` appear after actual dispatch. Training metrics/checkpoints appear under the respective output directory. The expected final full checkpoint is `checkpoints/step-00003699`, and the permanent model archive is `model_checkpoints/step-00003000`.

## Validation

- 854 CPU tests passed, 7 skipped, 2 GPU tests deselected.
- `make check` passed its syntax-compile fallback (ruff unavailable); `git diff --check` passed.
- Real LeRobot samples at dataset indices 0, 30 and 35 produced valid-step counts 10, 5 and 0. Targets exactly matched the original arrays' first ten entries.
- A tiny random CPU Qwen3-VL plus expert, constructed through the real training model builder, completed FM backward and sampling with `[3,10,3]` output; VLM gradients stayed absent. This is interface validation, not a pretrained GPU training or navigation evaluation.
- All three frozen configs were checked to differ only in beta parameter and output directory.
- Full GPU startup remains pending because all eight cards are occupied by another job.

Artifacts: [runs.json](runs.json), [queue_verification.json](queue_verification.json), [real_data_cpu_check.json](real_data_cpu_check.json).

## Completed results (2026-09-20)

All three runs exited successfully at step 3,699 / epoch 3, with complete final checkpoints and horizon 10. GPU 0–7 are idle at this check.

- Beta(1,1.5): last 20 logged FM losses mean=0.505322; elapsed=10.02 h; checkpoint `/mnt/cfs/Models/beta_nav/r2r_fm_only_h10_beta1_1p5_frozen_lightnav_20260918/checkpoints/step-00003699`.
- Beta(1,2): last 20 logged FM losses mean=0.557028; elapsed=7.65 h; checkpoint `/mnt/cfs/Models/beta_nav/r2r_fm_only_h10_beta12_frozen_lightnav_20260918/checkpoints/step-00003699`.
- Beta(1,3): last 20 logged FM losses mean=0.614633; elapsed=7.54 h; checkpoint `/mnt/cfs/Models/beta_nav/r2r_fm_only_h10_beta13_frozen_lightnav_20260918/checkpoints/step-00003699`.

Final plot and verified summary: `/mnt/pfs/tanshenghan/vln4project/LightNav-0/reports/betanav/fm_h10_final_20260920T010919Z`. Different time distributions prevent ranking navigation quality by training loss alone.
