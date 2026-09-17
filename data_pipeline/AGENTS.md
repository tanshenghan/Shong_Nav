# Data pipeline working rules

The user designated this directory as the permanent home for data-pipeline work.

- Keep collection, conversion, labeling, sampling and validation implementations,
  pipeline documentation, future configuration, tests and small reports here.
- Update `README.md` whenever the generation paradigm or primary entry point changes.
  Document command parameters in `docs/`; link them from the repository's
  `docs/CONFIGURATION.md` index.
- Keep raw datasets, RGB/depth collections and codebook assets under the configured
  data root (currently `/mnt/cfs/data/beta_nav`). Local environments belong in the
  ignored `.venv/` directory. Do not commit environment packages or large media.
- Preserve source/label provenance, coordinate frames, physical units, sampling
  definitions, independent loss masks and codebook versions in generated manifests.
- Current agreed paradigm: R2R route/actions, start state and instruction; fresh
  LightNav-configured Habitat-Sim RGB/depth/body/camera poses; 470x280, HFOV 120°,
  height 0.88 m, pitch 0°; RVQ horizon 10 and FM horizon 30.
- Current pointing definition: apos targets ground position min(t+10, N-1),
  falling back backwards along the future path to the nearest visible point.
  Always retain target_world_point, including the terminal 3D target in the last
  ten frames. No visible candidate means pixel supervision is unknown/masked.
  Never output apos turn/STOP markers. Historical configs without tail_policy
  retain their original masked-tail behavior when validating existing data.
- Save explicit original-image pixel_rc and 48x27 grid row/col/cell_id for valid
  apos/opos points. Rows/cols are zero-based; spatial IDs are 1+row*48+col.
  Missing spatial locations stay null in JSON and -1 in integer arrays, including
  the opos=0 not-visible class. Never invent a grid cell for an offscreen target.
- Current pointing tokens use only 0..1296. Unknown apos/opos use token_id=0 and
  an explicit <apos_0>/<opos_0> string, keeping state=unknown and loss_valid=false.
  Confirmed not-visible opos also uses 0 but keeps loss_valid=true. Preserve this
  distinction; zero placeholders do not create valid pixels or change loss masks.
  Record unknown_token_id=0 in pointing_config. Missing metadata means historical
  token_id=-1/null; support that only for read-only validation. Keep missing
  spatial indices and future-frame indices separate from pointing token IDs.
- Default opos_mode=terminal_ground: opos is the final destination, exactly the
  recorded terminal world_from_body[:3,3], fixed across all frames. Project using
  each saved camera pose, intrinsics and depth; never backtrack or replace the
  destination with another visible point. Always save target_world_point and
  target_frame_id, even when no spatial pixel is visible. This mode requires no
  API, object recognition, semantic assets or semantic replay. New collection and
  offline annotation both produce this label. Missing historical opos_mode means
  the old doubao semantics; do not reinterpret historical labels.
- Default opos_visibility_window=last_half: for N recorded frames, only allow
  spatial opos labels at t >= floor(N/2), using zero-based inclusive frame IDs.
  Earlier frames supervise <opos_0>, state=not_visible, loss_valid=true, with
  reason=before_opos_visibility_window; this is a temporal label restriction,
  not a claim of physical occlusion. Keep the same terminal 3D target throughout.
  Allowed frames still require actual projection/depth visibility. Save the
  policy and opos_visibility_start_frame; missing historical policy means all.
- Final generation supports only terminal_ground + terminal. The obsolete Doubao
  API, last-eight-frame search and semantic-object generation code was archived
  in archive/before_final_cleanup_20260912.tar.gz, then removed from active code.
  Keep historical metadata/search/object-mask validation read-only. Do not revive
  the old generation switches unless the user explicitly requests that behavior.
- Local config.py supplies LLM_DB_BASE_URL, LLM_DB_API_KEY and LLM_DB_MODEL.
  Keep it ignored by Git and excluded from provenance copies/hashes and logs.
  The final pipeline never reads this file. Exclude it from release archives too.
- releases/ holds versioned code snapshots and file hashes for manual review.
  Update the version for later releases rather than rewriting a frozen snapshot.
- Do not export forward/left/right action labels in new training samples/frames.
  Internal expert action replay remains until the user requests continuous resampling.
- Use a new annotation overlay for old data. Preserve historical data/provenance.
- Do not treat record FPS as physical action timing, EOF as STOP, unknown object
  labels as not-visible labels, or decoded RVQ trajectories as clean FM ground truth.
- LeRobot export is implemented in export_lerobot.py, with masked statistics in
  lerobot_stats.py and independent readback validation in validate_lerobot.py.
  run_lerobot.sh collects, exports, verifies, renders review videos and removes
  only its own scratch capture. Use the separate dataset environment initialized
  by setup_lerobot.sh; never install its dependencies into Habitat or inference.
  Use official LeRobot 0.4.4 / v3.0 image-backed storage: embedded lossless RGB PNG
  and float32 camera-Z metre depth plus boolean validity in Parquet. Final datasets
  must load without the source capture, including all RVQ assets and provenance.
  Use frame_index as logical time, fps=1 and timestamp=frame_index
  for synthetic alignment, retaining physical_dt_s=null. Each frame's structured
  action includes opos, apos, RVQ and the complete FM-30 waypoints plus SE(2)
  increments, with independent masks. Represent them as action.* feature columns;
  a single next-step motion vector does not replace this complete supervision.
- Maintain CPU geometry tests in `tests/` within this directory. `make test` from
  the repository root includes them. Use dry runs for entry-point/path-only changes;
  regenerating GPU datasets is unnecessary for a directory reorganization.
- By explicit user convention, Beta training reads live in
  `src/lightnav/dataloader/Train_Lerobot_DataLoader.py`, with tests in
  `tests/test_train_lerobot_loader.py` (both relative to the repository root).
  The reader returns `BetaTrainBatch` from selected LeRobot Parquet columns with
  checkpoint SlowFast sampling. Training is independent of this directory: setup
  is `scripts/setup_betanav_train.sh`, runtime is `.venv-train`, docs live in
  `docs/`, reports in `reports/betanav/`, and split preparation in
  `src/lightnav/training/prepare_data.py` (all paths relative to repository root).
  Preserve historical environments/reports here. Do not mix the export environment's older huggingface-hub
  with Transformers 5.8. Do not load model weights or CUDA objects in workers.
