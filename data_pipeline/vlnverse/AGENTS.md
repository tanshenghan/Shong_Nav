# VLNVerse pipeline working rules

- Read [README.md](README.md), [docs/DESIGN.md](docs/DESIGN.md) and
  [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) before changing a stage.
- Keep VLNVerse generation code, configs, docs, CPU tests and small reports in this
  subtree. Keep R2R/RxR under the sibling `R2R/` directory. Training readers and
  runtime continue to follow the root repository conventions.
- `source/` reads official data; `scene/` owns frame/map conversions and collision
  geometry; `motion/` owns path/time processing; `replay/` owns simulation capture;
  `labels/` owns targets and masks; `export/` owns dataset writing. CLI modules are
  thin wrappers. Do not make CPU motion code depend on Isaac or model weights.
- This robot may turn in place. The original retiming branch preserves source in-place turns and yaw continuity.
  The user-authorized `planning/` branch may replace those turns with a certified
  curved route inside reviewed ordered checkpoint corridors; its motion and output
  contracts are separate. See [docs/PLANNED_ROUTES.md](docs/PLANNED_ROUTES.md).
  Do not smooth a centerline through obstacles or treat endpoint-only collision
  checks as a swept-footprint guarantee. Unknown and out-of-map space is blocked.
- Require explicit map units, origin, axes, resolution and robot footprint. Never
  infer a raw occupancy image's coordinate transform or free-space convention
  silently. The planar circular-footprint check is not a 3D dynamics guarantee.
- Retimed timestamps are synthetic seconds, not source measured timestamps. Keep
  source mapping, input/config hashes, units and quality limitations in outputs.
- Default sampling is `sample_mode="per_source_step"`, with `poses_per_step=6`
  including both endpoints of each nonduplicate source transition. Accept 5/6/7/8
  points and deduplicate shared endpoints. Merged straight/rotation runs retain
  smooth timing across source anchors. Global sample intervals are not guaranteed
  constant; `sampling.uniform_timestamps` reports the observed timestamp spacing.
- `sample_hz` only applies to explicit `sample_mode="uniform_time"`. In the default
  mode, `quality.sample_hz` is null and consumers must use `timestamp_s` for dt.
  Point count does not determine duration or imply faster physical motion.
- Validate full-primitive derivative peaks analytically; sparse sampled maxima
  alone cannot prove speed, acceleration and jerk limits for the whole curve.
- The septic progress curve comes from endpoint value/derivative constraints;
  it is not a fitted or learned model from five points. Continuous spatial curves
  are handled separately by `planning/` and `motion/planned.py`.
- New sampled poses require new RGB/depth observations from scene replay.
  Do not reuse duplicated source frames as if captured at the new camera poses.
- Future labels must define a time horizon in seconds and sample interval; do not
  assign a fixed dt to sparse pose indices or copy R2R's 10/30-step horizon.
  Simulation may resample the continuous reference at its own control period.
  FM must use derived actual/replayed poses, not RVQ reconstructions.
- `scene/official_map.py` supports only the validated coordinate-header freemap
  format, with PNG classes and paired room vertices checked. It is not a generic
  native-map importer or a certificate for complete 3D robot clearance.
- `replay/` now contains Blender CPU and Isaac camera-only preview renderers and
  comparison-video encoding. Blender shading approximates selected MDL fields;
  it is not Isaac-equivalent. Report actual run status; the Isaac GPU path has not
  yet been verified in this task. Do not describe visual previews as physical replay.
  The completed CPU preview covers only `kujiale_0018_2_2`; see
  [its render report](reports/preview_render_20260919.json).
- Use the official navigation scene `start_result_navigation.usd` when present,
  matching the evaluator's asset selection. The base scene can retain door meshes
  removed in the navigation scene; do not substitute it silently. Install the
  CPU preview environment from `requirements-preview-cpu.txt`, separate from
  training, inference and Isaac environments.
- Default `reference_time` comparison videos hold freshly rendered views on a shared synthetic clock;
  original poses use corresponding source-anchor times. Container FPS does not
  define pose sampling or measured physical time. Do not fabricate new viewpoints
  with optical flow or duplicate images as training observations.
  Explicit `pose_frames` playback shows each smooth pose once at video FPS; it
  changes only playback timing, not NPZ timestamps, and cannot claim to retain
  the reference velocity, acceleration or jerk limits.
- Physical control replay, actual-state/depth/contact capture, `labels/` and
  `export/` remain planned. Preserve this distinction from the preview tools.
- Add meaningful CPU coverage under `tests/`. Keep large data under the configured
  root (default `/mnt/cfs/data/beta_nav`); do not overwrite existing outputs or
  rewrite immutable provenance. Small generation reports belong in `reports/`.
