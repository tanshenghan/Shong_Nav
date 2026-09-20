# Model assets and parameters

Checkpoint layout, the action decoder, `eval_config.json` and every server / CLI parameter.

## Checkpoint

A Hugging Face directory (or a directory containing `hf_ckpt/`) with `config.json`,
`model*.safetensors`, `tokenizer*`, `processor_config.json`, and ideally
`eval_config.json`. The architecture must be stock `Qwen3VLForConditionalGeneration`; the
trajectory / action tokens are ordinary rows of the embedding table.

## Action decoder

An **RVQ action-tokenizer bundle**: a directory with `manifest.json`, the per-level
codebook `.npy` files and the `jacobian_weights` file it names. The released checkpoints
ship it as `action_tokenizer/`, referenced from `eval_config.json`, so it is found without
flags; `--action_tokenizer_bundle` overrides it.

## `eval_config.json` (processing parameters)

If present next to or above the checkpoint (the search walks up four parents, so
`hf_ckpt/`, the `global_step_*` directory and the run root are all covered), it supplies the
processing parameters that must match training and cannot be guessed:

```json
{
  "version": 1,
  "common": {"video_size": [224, 320], "pool_enable": true, "pool_spatial": 2,
             "pool_mode": "avg", "pool_stage": "pre_vit"},
  "tasks": {
    "trackvla": {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "slowfast_tiers": null,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "/path/to/action_tokenizer"}},
    "vlnce":    {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "action_tokenizer"}}
  }
}
```

| Key | Meaning |
|---|---|
| `common.video_size` | model input frame size `[H, W]`; every frame is resized to it |
| `common.pool_*` | spatial pooling of vision tokens (`pool_stage` `pre_vit` or `post_vit`) |
| `tasks.<task>.num_history_frames` | history window (frames) fed to the model per step |
| `tasks.<task>.predict_horizon` | trajectory horizon `H` (rows per chunk) |
| `tasks.<task>.video_fps` | frame rate the checkpoint was trained at (send frames at this rate or faster; waypoint rows carry no time base of their own, see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `tasks.<task>.slowfast_tiers` | optional multi-rate history layout (SlowFast checkpoints keep the whole episode) |
| `tasks.<task>.action_tokenizer` | decoder snapshot: `{"method": "rvq", "bundle_path"}`; relative paths resolve against the config's directory |

`tasks` is keyed by task: `trackvla` for tracking (`--task tracking`), `vlnce` for
navigation (`--task vln`, and the Habitat evaluation). Values are read automatically; the CLIs'
`--num_history_frames` (and `--pool_spatial`, where offered) override them. Without the file the code
falls back to conservative defaults (`num_history_frames=16`, no pooling,
`video_size=(224, 320)`) which will not match a trained checkpoint — treat a missing
`eval_config.json` as a warning sign. Full schema: `src/lightnav/eval_config.py`.


### Shipping the action decoder with the checkpoint

`tasks.<task>.action_tokenizer.bundle_path` may be a **relative path**; it is resolved
against the directory holding `eval_config.json` (falling back to the checkpoint
directory). The released checkpoints ship their decoder this way:

```
hf_ckpt/
  config.json, model-*.safetensors, tokenizer*, processor_config.json
  eval_config.json          # bundle_path: "action_tokenizer" (both tasks)
  action_tokenizer/         # manifest.json + codebook_l*.npy + jacobian_weights.npy (+ alpha)
```

With such a layout `lightnav-serve`, `lightnav-predict`, `lightnav-eval-habitat` and
`build_tracking_agent()` need no `--action_tokenizer_bundle`: the
decoder is taken from the task entry that matches the server task (`--task tracking` →
`trackvla`, `--task vln` → `vlnce`), then from the other tasks, then from the sibling
directories `action_tokenizer/<task>` or `action_tokenizer/`. The explicit flag always
wins.

## Server / CLI parameters

`lightnav-serve` flags (environment names are what `scripts/start_servers.sh` and
`docker/entrypoint.sh` read):

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--model_path` | `MODEL_PATH` | required | checkpoint directory |
| `--action_tokenizer_bundle` | `ACTION_TOKENIZER_BUNDLE` | from the checkpoint | RVQ bundle dir; overrides the decoder the checkpoint ships |
| `--task` | `TASK` | `tracking` | `tracking` (tracking prompt, `tasks.trackvla`) or `vln` (VLN prompt, `tasks.vlnce`) |
| `--backend` | `BACKEND` | `vllm_local` | `vllm_local` or `hf` |
| `--gpu_memory_utilization` | `GPU_MEM_UTIL` | `0.85` | fraction of GPU memory handed to the vLLM engine |
| `--max_batch_size` | `MAX_BATCH_SIZE` | `8` | max sessions per scheduler tick; also sizes vLLM `max_num_seqs`. `1` = strictly serial |
| `--max_wait_ms` | `MAX_WAIT_MS` | `8` | max wait to fill a batch; a lone request is flushed after 2 ms |
| `--num_history_frames` | `NUM_HISTORY_FRAMES` | checkpoint config | history-window override (normally leave unset) |
| `--pool_spatial` | `POOL_SPATIAL` | checkpoint config | spatial-pooling override |
| `--aspect_mode` | `ASPECT_MODE` | `stretch` | `stretch`: resize every frame to `video_size` (training behaviour); `keep`: per-session size with the camera's aspect ratio at the same pixel budget (4:3 → 288×384 for a 256×448 checkpoint) |
| `--max_new_tokens` | `MAX_NEW_TOKENS` | `8` | lower bound for the per-step decode cap (see below) |
| `--host` / `--port` | `HOST` / `PORT` | `0.0.0.0` / `8050` | bind address |
| `--ready_file` | – | none | file touched once the port is bound (used by the launchers) |
| `--no_warmup` | – | off | skip the synthetic warm-up inference before binding the port |
| `--record_dir` | `RECORD_DIR` | off | record every connection's episodes (frames + per-step records) for `lightnav-render` (see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `--record_fps`, `--record_timeline`, `--no_record_images` | `RECORD_FPS`, `RECORD_TIMELINE`, `RECORD_IMAGES` | `10`, `realtime`, images on | recording defaults written to the manifest / frame storage |
| `--cam_hfov_deg`, `--cam_height`, `--traj_forward_offset`, `--waypoint_dt_s` | `CAM_HFOV_DEG`, `CAM_HEIGHT`, `TRAJ_FORWARD_OFFSET`, `WAYPOINT_DT_S` | `90`, `0.5`, auto, `0.1` | client-camera geometry and HUD convention used only by the rendered overlay |

Engine-level environment variables (no flag):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLN_KV_CACHE_GIB` | auto | vLLM KV-cache size in GiB (auto-scales with `max_num_seqs`, floor 2 GiB) |
| `VLN_VLLM_ENFORCE_EAGER` | `0` | `1` disables CUDA-graph capture (faster start, slower steps) |
| `LIGHTNAV_ATTN` | `sdpa` | attention implementation for the `hf` backend (e.g. `flash_attention_2`) |
| `VLN_EVAL_TEMPERATURE` / `TOP_P` / `TOP_K` | greedy | sampling knobs for experiments; benchmark numbers assume they are unset |

**Decode token budget.** A step emits a *grounding prefix* followed by the action token(s):
one `<act_l{d}_*>` per RVQ level. The server
probes the tokenizer for the grounding families it knows and caps generation at
`max(--max_new_tokens, prefix + action_tokens)`, which saves the decode step otherwise spent
reaching `eos` without ever truncating the action.

| checkpoint family | grounding prefix | tokens |
|---|---|---|
| tracking, legacy | `<tpos_k>` | 1 |
| tracking + grid pointing | `<opos_k>` | 1 |
| navigation + dual pointing | `<apos_A><opos_O>` | 2 |

**GPU sizing.** The 4B checkpoint needs ~10 GB for weights in bf16 plus the vLLM KV cache;
one 24 GB GPU comfortably serves one engine with `--gpu_memory_utilization 0.85`. When
several servers share a GPU, `scripts/start_servers.sh` divides the utilisation by
`SERVERS_PER_GPU`.

Note what `--gpu_memory_utilization` does *not* do here. The engine passes an explicit
`kv_cache_memory_bytes` (`VLN_KV_CACHE_GIB`, floor 2 GiB) and therefore skips vLLM's
profile run, so real usage is roughly weights + that KV cache + CUDA graphs (~15 GB for
the released 4B checkpoint) whatever the fraction says. The fraction still acts as the
ceiling vLLM checks against free memory at start-up, so lower it when the GPU is shared
with another job, and raise `VLN_KV_CACHE_GIB` — not the fraction — when you want a
bigger KV cache.

---

## Habitat BetaNav FM evaluation

`lightnav-eval-habitat` normally executes the trajectory decoded from the token/RVQ
branch. To evaluate a trained BetaNav continuous expert instead, select it explicitly:

```bash
lightnav-eval-habitat \
  --model_path /path/to/base-lightnav-checkpoint \
  --backend hf --action_output fm \
  --beta_checkpoint /path/to/betanav-run
```

`--beta_checkpoint` accepts a `model.pt`, a committed `step-*` directory, or a training
run directory. For a run directory, the newest committed `latest_model.json` /
`latest.json` pointer is resolved and recorded in `summary.json`. `--fm_seed` defaults
to 42 and deterministically keys FM noise by scene, episode and policy step;
`--fm_steps` optionally overrides the integration-step count stored in `beta_config`.
With `--action_output fm`, the executed motion is the FM trajectory. The token branch remains active only
for its trained STOP decision and RVQ diagnostics. BetaNav inference is HF eager only;
`--backend vllm_local --action_output fm` is rejected instead of falling back.

To compare the same trained BetaNav token branch against FM, use
`--backend hf --action_output token --beta_checkpoint /path/to/step-00029589`
with the same base `--model_path`. This loads the complete BetaNav checkpoint and
executes its decoded RVQ trajectory, skipping FM context projection and sampling.
It preserves baseline RVQ STOP/decode-failure semantics and requires
`--fm_stop_mode baseline`; FM seeds and integration steps do not affect this path.
Without `--beta_checkpoint`, token output still uses the base LightNav weights.

Diagnostic subset/STOP analysis is available through `--episode_manifest FILE`, whose
entries contain exact `scene_id` + `habitat_episode_id` pairs. `--fm_stop_mode baseline`
reproduces the released evaluator; `decode_continue` keeps executing FM after malformed
RVQ text; `explicit_gate` additionally permits only the canonical RVQ STOP tuple to end
an episode and suppresses accidental zero-speed FM commands. `--oracle_stop_distance 3`
uses privileged goal distance and is an analysis upper bound, never a deployable result.
Every output episode records `stop_source` and per-policy-event counts.

The official R2R `test` split has no public goals/GT and therefore cannot produce local
SR/SPL/NDTW/NE. Use `val_unseen` for quantitative local evaluation; `test` is only for
generating trajectories for the external benchmark server.

---

## Input aspect ratio (`--aspect_mode`)

The checkpoints are trained on frames resized to a fixed `common.video_size` (256×448 for
the released checkpoint, ≈16:9). By default (`stretch`) every client frame is resized to
that size regardless of its aspect ratio, exactly as in training — a 4:3 camera therefore
arrives horizontally squeezed.

`--aspect_mode keep` (env `ASPECT_MODE=keep`; also `lightnav-predict --aspect_mode`,
`lightnav-eval-habitat --aspect_mode`, `build_tracking_agent(aspect_mode=)`) instead picks,
per session from its first frame, the size with the **source aspect ratio** whose area is
closest to `video_size` and whose sides are multiples of 32 (patch × merge; and of the pre-ViT
pooling factors when the checkpoint pools before the ViT). The vision-token count per frame
stays at the training budget, so history length and sequence length are unchanged:

| camera | `stretch` | `keep` |
|---|---|---|
| 16:9 (480×270, 1920×1080) | 256×448 | 256×448 (identical) |
| 4:3 (640×480) | 256×448 (squeezed) | 288×384 |
| 1:1 | 256×448 | 352×352 |

Frames within one session must share one size (the first frame decides; `reset` starts
over). `keep` avoids geometric distortion but changes the token grid the model sees, which
the released checkpoint was not trained on — validate on your robot before relying on it.
Full native resolution (feeding the camera's own pixel count) is not supported.
## R2R collection script

For one actual closed-loop LightNav navigation with fresh Habitat observations,
raw pointing/RVQ predictions, continuous control and a prediction video, see
[data_pipeline/R2R/docs/LIGHTNAV_ROLLOUT.md](../data_pipeline/R2R/docs/LIGHTNAV_ROLLOUT.md#运行).
This is a model rollout, separate from the expert-GT collection below.

`data_pipeline/R2R/collect_r2r.py` runs in a separate Habitat-Sim Python 3.9 environment.
It renders expert RGB/depth/poses and exports RVQ-10 / FM-30, apos-10 and final-destination opos labels
without LightNav model inference. All collection and validation parameters are documented in
[data_pipeline/R2R/docs/R2R_COLLECTION.md](../data_pipeline/R2R/docs/R2R_COLLECTION.md#collection-command-parameters).
The former `scripts/collect_beta_nav.py` entry point forwards to the same collector.

`data_pipeline/R2R/annotate_r2r.py` adds ten-step apos and fixed final-destination opos.
The current default allows spatial opos only in the episode's final half; the
camera projection and depth visibility checks still apply inside that window.
It projects the recorded terminal ground point with saved camera poses and depth,
using CPU only and no API or semantic assets. New generation supports only the
final terminal-ground definition; the old API/semantic generator is archived.
All remaining flags are documented in
[data_pipeline/R2R/docs/POINTING.md](../data_pipeline/R2R/docs/POINTING.md#命令).

`data_pipeline/R2R/run_lerobot.sh` collects and exports self-contained LeRobotDataset
v3.0 data. `export_lerobot.py` accepts `dataset`, `--output-dir`, `--repo-id` and
repeatable `--episode-id`; `validate_lerobot.py` accepts `dataset`, optional
`--source` and `--report`. Setup, feature schema, loss masks, logical frame time
and all command parameters are documented in
[LEROBOT_FORMAT.md](../data_pipeline/R2R/docs/LEROBOT_FORMAT.md#运行与加载).

`data_pipeline/R2R/run_r2r_train.sh` builds all local R2R train instructions with
resumable shards and a standard LeRobot v3 merge. Workers, GPU allocation, shard
size, retries, pilot limits, source paths and coverage checks are documented in
[FULL_R2R_TRAIN.md](../data_pipeline/R2R/docs/FULL_R2R_TRAIN.md#参数).
The collector's `--episode-ids-file` bypasses route sampling with an exact JSON ID list.

`data_pipeline/R2R/run_rxr.sh` uses the same collector, GT definitions and LeRobot exporter for RxR.
The shared builder and collector accept `--dataset r2r|rxr`, `--split`, `--role` and `--languages`;
RxR defaults to the complete train guide source with all languages. Source splits and roles
use separate output directories. Parameters and examples are documented in
[RXR_LEROBOT.md](../data_pipeline/R2R/docs/RXR_LEROBOT.md#命令).

`data_pipeline/R2R/export_pointing_grid.py` adds explicit original-image pixel indices
and 48x27 grid cells to an existing pointing dataset without inference. Its dataset,
output and episode-selection arguments are documented in
[POINTING.md](../data_pipeline/R2R/docs/POINTING.md#二维像素与具体网格).

`visual/render_episode_gt.py` renders a saved episode with its instruction, future
ground-truth path and valid apos/opos, or a single-frame PNG with pixel/grid-cell
annotations and a magnified detail. All inputs, playback and rendering flags are
documented in [data_pipeline/R2R/docs/VISUALIZATION.md](../data_pipeline/R2R/docs/VISUALIZATION.md#运行).
For image-backed BetaNav LeRobot v3, `--episode-id` selects the original source episode
and `--scene` optionally disambiguates its Scan. `--hide-occluded`, `--number-steps`,
and `--preview-frame` control ground overlays and the poster. `--rgb-only --scale 1`
keeps the MP4 at the native 470×280 resolution without instruction or top-view panels. See the
[direct Parquet workflow](../data_pipeline/R2R/docs/VISUALIZATION.md#直接读取-lerobot-parquet).

`src/lightnav/dataloader/Train_Lerobot_DataLoader.py` provides the Beta training reader and
a CPU inspection command. Its dataset, model path, batch/worker options, episode
and sample indices, optional tiny-model check, and report arguments are documented
in [TRAIN_LEROBOT_DATALOADER.md](TRAIN_LEROBOT_DATALOADER.md#检查命令)。
The inspection command also accepts `--fm-horizon` (integer 1–30, default 30) and
`--training-mode ce_fm|fm_only` (default ce_fm). The former takes consecutive future
target steps and their masks from the stored 30-step arrays.

`train_betanav.py` launches BetaNav training with Accelerate/DDP and optional
ZeRO-1 optimizer state sharding. All CLI options (`--config`, `--output-dir`,
`--data-manifest`, `--resume`, `--expand-training-data`, `--override-max-steps`,
`--dry-run`, `--cpu`, `--tiny-model`, `--stop-after`),
training YAML settings, launcher configuration, and the scene-split preparation
command (`--output`, `--seed`, `--val-fraction`, `--scale-floor`, `--normalization-from`) are documented
in [TRAIN_BETANAV.md](TRAIN_BETANAV.md).
The R2R YAML uses all episodes, disables validation, and stops at global step `29589`.
`checkpoint_mode: dual`, `save_every: 1000`, and `keep_last: 3` retain the newest three
complete training states; `model_save_every: 3000` also retains model-only archives permanently.
Immutable CPU model/Adam snapshots are written in background threads using per-rank optimizer shards.
Complete states restore Adam/RNG and the data cursor; permanent model archives omit that training state.
`--override-max-steps` explicitly permits a changed stopping limit on resume and adjusts the remaining
cosine schedule without resetting the data cursor or relaxing other optimizer/batch constraints.

`training_mode: fm_only` freezes the complete original VLM, removes token/RVQ supervision and
trains only the context projection and action expert. `ce_fm` remains the default.
`num_epochs: 3` derives the update limit from the actual distributed loader (including tail batches);
omit it to use `max_steps`. It cannot be combined with `--override-max-steps`.
See `configs/betanav/r2r_fm_only.yaml` for the separate three-epoch experiment.

`fm_horizon` selects the continuous action length for both the training reader and
expert (integer 1–30; legacy default 30). Set `fm_horizon: 10` for `[B,10,3]` targets
and predictions. `num_flow_steps` independently controls integration iterations.
The horizon is part of the strict resume contract; old missing values mean 30.
The three `configs/betanav/r2r_fm_only_h10_beta*.yaml` configurations train fresh
ten-step experts with Beta(1,1.5), Beta(1,2), and Beta(1,3), respectively.

`flow_time_distribution` selects `uniform` (default) or `beta` for training times.
`flow_time_alpha` and `flow_time_beta` must be finite and positive (both default to 1.0).
Our interpolation is `(1-tau)*noise + tau*GT`, so Beta(1,2) favors noisier inputs.
The separate `configs/betanav/r2r_fm_only_beta12.yaml` combines Beta(1,2) with
`lambda_fm: 1.0`. Sampling settings are recorded in model and resume metadata;
explicitly supplied flow times bypass random sampling.

`configs/betanav/r2r_fm_only_beta13.yaml` keeps the same eight-GPU settings with
Beta(1,3) and a separate fresh experiment directory.
`python -m lightnav.training.scheduled_launch --schedule PATH` runs a one-shot
server queue for a prepared training supervisor. `--check` only checks readiness.
The JSON contract, GPU availability checks and process lifetime requirement are
documented in [TRAIN_BETANAV.md](TRAIN_BETANAV.md#beta13-与延迟启动).

`python -m lightnav.training.profile_batch` measures complete training updates
with real frames and full pretrained weights, without writing checkpoints.
Its `--config`, `--report`, `--batches`, `--per-rank-batch`, `--warmup`, `--steps`
and `--history-frame` options, plus measured batch recommendations, are documented
in [TRAIN_BETANAV.md](TRAIN_BETANAV.md#batch-与显存实测2026-09-13).
`--no-accumulation` tests each candidate as a complete batch with accumulation=1;
the effective batch varies by candidate and `--per-rank-batch` is ignored.

Performance measurement adds `--count-flops`, `--peak-bf16-tflops`, `--trace-dir`
and `--telemetry` to the batch profiler. `python -m lightnav.training.performance`
measures a dense BF16 GEMM baseline (`--report`, `--device`, `--sizes`, `--iterations`).
All options, training performance log fields and GPU monitoring commands are in
[TRAINING_PERFORMANCE.md](TRAINING_PERFORMANCE.md).


### FM-only checkpoints

With an `fm_only` checkpoint, `--backend hf --action_output fm --beta_checkpoint PATH`
automatically uses the training continuous-trajectory prompt and
`encode_context()` / `sample_actions()`. No RVQ decoder, LM generation or learned
STOP is used. Use `--fm_stop_mode baseline` (default), `--max_steps 500` and keep
forced STOP enabled; oracle STOP is rejected in this fixed-budget protocol.
The 500th decision is forced STOP. If a predicted velocity would implicitly STOP
Habitat, the policy tries later predicted waypoints, then a small deterministic
turn; this is recorded as `fm_only_zero_command_suppressed`. Results record
`training_mode=fm_only` and `stop_protocol=max_steps`. SR/SPL use the final position
at the budget, while oracle success reflects whether the goal was ever reached.
Comparisons with token-controlled STOP must state this protocol difference.


For FM-only checkpoints, `--fm_only_stop_mode rvq` enables frozen LightNav RVQ
STOP with the previous `--fm_stop_mode baseline` rules: canonical STOP, near-zero
RVQ trajectories and malformed-token fallback stop the episode. A near-zero FM
velocity also stops, and the maximum-step forced STOP remains enabled. RVQ controls
only stopping; movement uses the trained FM-only expert. The frozen VLM uses its
original RVQ prompt for STOP, then a separate continuous-trajectory prefill for FM,
matching each branch's training prompt. Results record `stop_protocol=rvq`.
`--fm_only_stop_mode max_steps` is the default and preserves the pure-FM fixed-budget
protocol described above. The frozen STOP weights differ from step-29589's CE-trained
STOP branch even though the decision rules are aligned.

### VLNVerse trajectory preparation

`python -m data_pipeline.vlnverse.cli.plan` is the separate A* curved-route branch.
Required flags are `--episode-parquet`, `--occupancy-npz`, `--config`, and a new
`--output-dir`; `--metadata-jsonl` optionally supplies episode metadata. The JSON
configuration specifies ordered source checkpoint indices, optional expected source
hash/episode ID, `planning` parameters and `motion` parameters. The example
`data_pipeline/vlnverse/configs/planned_0055.json` retains a 0.30 m footprint,
0.50 m source-route corridor, endpoint poses and 15 Hz synthetic sampling.
`time_law=ramp_cruise` uses quintic velocity ramps over `ramp_fraction=0.1` of the
duration at each end; `global_septic` remains optional. Both certify continuous
linear/angular derivative bounds. This format is `vlnverse_planned_motion_v1`,
with no official-step correspondence. See
[planned route contracts](../data_pipeline/vlnverse/docs/PLANNED_ROUTES.md).

`python -m data_pipeline.vlnverse.replay.render_parallel` requires `--scene-usd`,
`--poses-npz`, and a new `--output-dir`; optional `--workers` (4), `--threads`
(12 per process), and `--samples` (8) control CPU resources. It renders disjoint
ranges with the same full pose array, checks shard identities, then publishes one
complete render manifest. Other camera settings use the Blender defaults.

`python -m data_pipeline.vlnverse.replay.compare_planned` requires
`--previous-retimed-dir`, `--previous-images`, `--planned-dir`, `--planned-images`,
and a new `--output-dir`. It produces `planned.mp4` and `before_after.mp4`, fixed
at 15 FPS, with full Instructions and the next 30 poses. New input timestamps must
match 15 Hz in the default `--playback-mode reference_time`. Explicit
`--playback-mode pose_frames` permits other strictly increasing input timestamps,
shows each stored pose once at 15 FPS, and labels the changed playback timing.
The compact example `configs/planned_0055_compact.json` uses 4.5 Hz motion samples:
200 poses spanning 44.22 synthetic seconds produce a 13.333-second video.
Reference timestamps and HUD velocities remain synthetic motion values; future
30-pose horizons use actual sample timestamps and are no longer fixed at two seconds.
The previous synthetic timeline is normalized to the new duration;
the report records old omitted frames and does not promise equal spatial progress.
`python -m data_pipeline.vlnverse.replay.plot_route` requires `--planned-dir`
and a new `--output` PNG path for the overhead route review.

`python -m data_pipeline.vlnverse.cli.retime` converts official stepwise poses to
smooth reference motion for a mobile robot that can turn in place. It requires
`--episode-parquet`, `--occupancy-npz` and a new `--output-dir`; optional
`--config`, `--episode-id` and `--metadata-jsonl` specify retiming settings and
source metadata. All flags, units and output contracts are documented in
[the VLNVerse data contract](../data_pipeline/vlnverse/docs/DATA_CONTRACT.md#命令参数).
The [default config](../data_pipeline/vlnverse/configs/retime.json) uses
`sample_mode="per_source_step"` and `poses_per_step=6` (allowed: 5, 6, 7, 8),
including both endpoints of every nonduplicate source transition and deduplicating
shared endpoints. Global intervals may vary: use `timestamp_s` for dt. Point count does
not set motion duration. `sample_hz=30` only applies when explicitly selecting
`sample_mode="uniform_time"`. The config also controls the circular footprint and
margin, and linear/angular derivative limits. Output metadata uses `schema_version=2`,
`format="vlnverse_smooth_motion_v2"`; default `quality.sample_hz` is null.
`sampling.uniform_timestamps` reports actual spacing, and
`quality.continuous_maxima` checks full-curve analytic derivative peaks independently
of the saved pose count.
Outputs use synthetic retimed seconds and planar collision checks.
`scene.official_map.load_official_map` now calibrates the supported official
coordinate-header freemap format against the PNG, scene bounds and room vertices.
Camera-only visual preview tools are implemented; physical replay, actual-state
and depth capture, training labels and LeRobot export remain later stages in
[the pipeline design](../data_pipeline/vlnverse/docs/DESIGN.md).
For either renderer, `--scene-usd` should select the official navigation asset
`start_result_navigation.usd` when present, matching the evaluator's priority.
The base `kujiale_0018.usda` retains doors removed from the navigation scene and
must not silently replace the verified navigation asset.

`python -m data_pipeline.vlnverse.replay.isaac_render` renders each input pose with
an independent headless Isaac Sim 4.5 camera. Required flags are `--scene-usd`,
`--poses-npz` and a new `--output-dir`. Optional flags are `--camera-height-m`
(1.25 m above input Z), `--hfov-deg` (90), `--width` (640), `--height` (480),
`--warmup-frames` (30), and `--subframes` (2). This is a chosen preview camera,
not recovered official extrinsics; it outputs PNGs and a completion manifest,
without a robot or physical rollout. Scene loading uses bounded 300-second waits;
capture uses a 180-second wait. `--help` works without importing Isaac.
Input conventions, failure checks and the Isaac launcher command are in
[the replay README](../data_pipeline/vlnverse/replay/README.md#独立-isaac-相机预览).

`python -m data_pipeline.vlnverse.replay.blender_render` is the explicit CPU visual
preview path, using the independent `data_pipeline/.venv/blender-preview/` environment
(Python 3.11; install with `pip install -r data_pipeline/vlnverse/requirements-preview-cpu.txt`).
The [requirements file](../data_pipeline/vlnverse/requirements-preview-cpu.txt)
pins bpy, NumPy, usd-core and Pillow. Required arguments are `--scene-usd`,
`--poses-npz` and a new `--output-dir`; optional arguments are `--camera-height-m`
(1.25 above input Z), `--hfov-deg` (90), `--width` (480), `--height` (360),
`--camera-pitch-down-deg` (0), `--future-steps` (0, disabled), `--samples` (8),
`--threads` (16), and `--max-frames` (unset; limit diagnostic captures).
`--frame-start` (0) and `--frame-stop` (exclusive; default input length) select a
nonempty render shard. Output PNG names retain global pose indices; the manifest
records the actual range. A partial range is not a completed full-episode render.
For the instruction/path review, set `--camera-pitch-down-deg 20` on both renders,
`--future-steps 0` on the original, and `--future-steps 30` on the smoothed sequence.
The horizon means the next 30 stored smoothed poses (`i+1..i+30`), not 30 source
actions or a fixed number of seconds; use only remaining poses near the end, with
no endpoint padding. Ground raycasts require an upward surface near source Z,
either in the USD `floor` group or a `wall`-group horizontal cap whose entire mesh
thickness and source-Z deviation are at most 2 mm, with a separate floor-only ray
confirming identified floor within 8 cm below. Furniture and vertical walls fail.
Ground raycasts place 3D markers in the scene; scene occlusion,
the field of view and near clipping can hide them. Annotated PNGs are review images,
not clean training RGB. The previous previews used the default level camera.
It approximates selected MDL color/texture, roughness and glass features with shared
preview lighting. It is neither an Isaac-equivalent renderer nor a physics rollout.
The selected scene's world transforms and all 23 original/111 smoothed poses have
been verified. The historical unannotated preview videos decode completely (603 frames,
20 FPS, 30.15 seconds); their 23 source-anchor RGB pairs match pixel for pixel. This completes
only the selected CPU preview. The Isaac GPU path remains unvalidated. See the
[render report](../data_pipeline/vlnverse/reports/preview_render_20260919.json).

`python -m data_pipeline.vlnverse.replay.compare` accepts required `--retimed-dir`,
`--original-images`, `--smooth-images`, and new `--output-dir`, plus `--fps` (20)
and `--playback-mode` (`reference_time`, or `pose_frames`). Additional flags are
`--overlay-style` (`scene` by default, or `lightnav`) and `--future-steps` (30,
positive integer; future smoothed poses for the `lightnav` ribbon). The compare
flag is separate from the Blender renderer's same-named 3D overlay flag.
Optional `--official-rgb-npy PATH` replaces the left view with the published RGB
array from the source Parquet's own episode directory. For
`EPISODE/data/chunk-000/episode_000000.parquet`, the required array path is
`EPISODE/videos/chunk-000/observation.images.rgb/rgb.npy`. It must be uint8
`[source_pose_count, height, width, 3]`; directory identity, shape, count and a
content hash are checked/recorded. Frames preserve their aspect ratio and are
letterboxed without cropping; resizing is spatial only. Both Blender image
directories and manifests remain required for the current input checks.
With this flag, output names are `official_rgb.mp4`, `smoothed_{K}poses.mp4` and
`official_vs_smoothed.mp4`; the default names below remain unchanged otherwise.
The manifest records `official_rgb`, `left_view_source` and
`matched_left_right_camera=false`; its existing `camera` field describes the
Blender preview, not recovered official calibration. Footer labels distinguish
official RGB from the smoothed Blender preview: their cameras and renderers are
not equivalent. Source frames are held at source anchors on the chosen playback
clock; this does not recover real action timing or make the preview training RGB.
It currently reads Blender-style numbered PNGs and `render_manifest.json`, emitting
`original_gt.mp4`, `smoothed_{K}poses.mp4` (`K=poses_per_step`, default 6),
`side_by_side.mp4` and a comparison manifest.
All videos display the complete official `instruction` from the retimed manifest
at the top. The side-by-side video uses one full-width `Instructions` banner;
single videos also have banners. Text wraps without truncation, with banner height
expanding as needed. The scene-style smoothed footer reports the remaining cyan future-pose
count (up to 30, reaching 0 at the final pose). The manifest records the instruction,
banner layout and path annotation settings. These videos are for review, not clean
training RGB. The historical `kujiale_0018_2_2` instruction/path version is complete: all 23 original and 111
smoothed views rendered, and all three videos decode to 111 frames at 15 FPS
(7.4 seconds). Individual videos are 480×710; side-by-side is 960×656, including
the full instruction banner. Official instruction text and every future-pose
window were verified; the final frame has no future markers and its rendered RGB
matches the unannotated source pose exactly. See the
[instruction/path report](../data_pipeline/vlnverse/reports/preview_instruction_path_20260919.json).
Those historical outputs remain available:
[original GT](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/original_gt.mp4),
[smoothed with future path](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/smoothed_6poses.mp4),
and [side-by-side](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/side_by_side.mp4).
The recommended preview uses `--playback-mode pose_frames --fps 15`: each smoothed
pose appears exactly once, giving 111 frames and 7.4 seconds in the selected episode.
The original side holds the preceding source pose until its next matching anchor;
both sides have the same video frame count and duration. This changes only the
playback clock; the nonuniform `timestamp_s` in the retimed NPZ is unchanged.
The playback no longer preserves reference dt, so its apparent motion must not be
claimed to retain the reference speed, acceleration or jerk limits.
The default `reference_time` mode holds views until their next synthetic timestamp;
original poses use matching source-anchor times. The historical 20 FPS, 30.15-second
videos remain available. Neither mode recovers official elapsed time or interpolates
images. Commands, outputs and playback reports are in
[the preview workflow](../data_pipeline/vlnverse/replay/README.md).

For the optional shared LightNav visual style, render both sides with
`--camera-pitch-down-deg 0 --future-steps 0`, then run the compare command through
`PYTHONPATH=src data_pipeline/.venv/lerobot/bin/python` with
`--overlay-style lightnav --future-steps 30 --playback-mode pose_frames --fps 15`.
The existing environment contains OpenCV and imageio. This mode directly calls
`lightnav.viz.render.draw_traj_ribbon` and `draw_scifi_hud`, and rejects pitched
cameras or input RGB with a 3D path already drawn. The original side has no ribbon
and no velocity estimate. The smoothed ribbon uses future world positions
transformed once into the current body frame, without a cumulative sum; the
projection-only yaw column is zero to bypass action-specific yaw-only heuristics.
Actual yaw remains in the trajectory NPZ. HUD velocity uses saved analytic
synthetic-reference derivatives rotated into body coordinates, not a default
0.1-second or video-FPS estimate; HUD step/FPS identify the video frame/playback
clock. Full official Instructions stay in the top banner; HUD labels identify
the original and smoothed sides.
The shared blue ribbon uses an automatic visual forward offset (about 1.667 m
for the 480×360, 1.25 m, 90° camera) and width fraction 0.25. It has no depth
occlusion and does not depict exact ground placement. This is review graphics,
not training RGB. The comparison manifest records shared-function provenance
and these visual/time conventions. The selected `kujiale_0172_51_0` episode has
23 source poses, 111 smoothed poses, four ±15° in-place turns and a synthetic
reference duration of about 34.435 seconds. Its full preview is complete: all three
videos decode to 111 frames, 15 FPS and 7.4 seconds; single videos are 480×686 and
side-by-side is 960×626. The 23 original and 111 smoothed base images are all unique
within each sequence, and all 23 corresponding source-anchor RGB pairs match exactly.
Every future window and the full instruction text were verified; 341 related tests
and Ruff passed. See the [LightNav preview report](../data_pipeline/vlnverse/reports/preview_lightnav_viz_0172_20260919.json).
The recommended outputs are
[LightNav side-by-side](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/side_by_side.mp4),
[smoothed with ribbon/HUD](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/smoothed_6poses.mp4),
and [original GT](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/original_gt.mp4).
