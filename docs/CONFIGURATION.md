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
[data_pipeline/docs/LIGHTNAV_ROLLOUT.md](../data_pipeline/docs/LIGHTNAV_ROLLOUT.md#运行).
This is a model rollout, separate from the expert-GT collection below.

`data_pipeline/collect_r2r.py` runs in a separate Habitat-Sim Python 3.9 environment.
It renders expert RGB/depth/poses and exports RVQ-10 / FM-30, apos-10 and final-destination opos labels
without LightNav model inference. All collection and validation parameters are documented in
[data_pipeline/docs/R2R_COLLECTION.md](../data_pipeline/docs/R2R_COLLECTION.md#collection-command-parameters).
The former `scripts/collect_beta_nav.py` entry point forwards to the same collector.

`data_pipeline/annotate_r2r.py` adds ten-step apos and fixed final-destination opos.
The current default allows spatial opos only in the episode's final half; the
camera projection and depth visibility checks still apply inside that window.
It projects the recorded terminal ground point with saved camera poses and depth,
using CPU only and no API or semantic assets. New generation supports only the
final terminal-ground definition; the old API/semantic generator is archived.
All remaining flags are documented in
[data_pipeline/docs/POINTING.md](../data_pipeline/docs/POINTING.md#命令).

`data_pipeline/run_lerobot.sh` collects and exports self-contained LeRobotDataset
v3.0 data. `export_lerobot.py` accepts `dataset`, `--output-dir`, `--repo-id` and
repeatable `--episode-id`; `validate_lerobot.py` accepts `dataset`, optional
`--source` and `--report`. Setup, feature schema, loss masks, logical frame time
and all command parameters are documented in
[LEROBOT_FORMAT.md](../data_pipeline/docs/LEROBOT_FORMAT.md#运行与加载).

`data_pipeline/run_r2r_train.sh` builds all local R2R train instructions with
resumable shards and a standard LeRobot v3 merge. Workers, GPU allocation, shard
size, retries, pilot limits, source paths and coverage checks are documented in
[FULL_R2R_TRAIN.md](../data_pipeline/docs/FULL_R2R_TRAIN.md#参数).
The collector's `--episode-ids-file` bypasses route sampling with an exact JSON ID list.

`data_pipeline/run_rxr.sh` uses the same collector, GT definitions and LeRobot exporter for RxR.
The shared builder and collector accept `--dataset r2r|rxr`, `--split`, `--role` and `--languages`;
RxR defaults to the complete train guide source with all languages. Source splits and roles
use separate output directories. Parameters and examples are documented in
[RXR_LEROBOT.md](../data_pipeline/docs/RXR_LEROBOT.md#命令).

`data_pipeline/export_pointing_grid.py` adds explicit original-image pixel indices
and 48x27 grid cells to an existing pointing dataset without inference. Its dataset,
output and episode-selection arguments are documented in
[POINTING.md](../data_pipeline/docs/POINTING.md#二维像素与具体网格).

`visual/render_episode_gt.py` renders a saved episode with its instruction, future
ground-truth path and valid apos/opos, or a single-frame PNG with pixel/grid-cell
annotations and a magnified detail. All inputs, playback and rendering flags are
documented in [data_pipeline/docs/VISUALIZATION.md](../data_pipeline/docs/VISUALIZATION.md#运行).
For image-backed BetaNav LeRobot v3, `--episode-id` selects the original source episode
and `--scene` optionally disambiguates its Scan. `--hide-occluded`, `--number-steps`,
and `--preview-frame` control ground overlays and the poster. `--rgb-only --scale 1`
keeps the MP4 at the native 470×280 resolution without instruction or top-view panels. See the
[direct Parquet workflow](../data_pipeline/docs/VISUALIZATION.md#直接读取-lerobot-parquet).

`src/lightnav/dataloader/Train_Lerobot_DataLoader.py` provides the Beta training reader and
a CPU inspection command. Its dataset, model path, batch/worker options, episode
and sample indices, optional tiny-model check, and report arguments are documented
in [TRAIN_LEROBOT_DATALOADER.md](TRAIN_LEROBOT_DATALOADER.md#检查命令)。

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
