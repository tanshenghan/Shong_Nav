# Four randomly selected FM + StopHead samples

Selection seed: **20260928**. Sampling is without replacement, separately from
the completed evaluation's successes and failures. Source records and file
hashes are frozen in `selection.json`.

| Source outcome | Episode | Scene | Steps | Final distance |
|---|---|---|---:|---:|
| Success | 938 | oLBMNvg9in8 | 23 | 2.680 m |
| Success | 1091 | X7HyMhZNoso | 28 | 1.830 m |
| Failure | 1060 | QUCTc6BB5sX | 29 | 13.108 m |
| Failure | 522 | x8F5xyUWy9e | 39 | 11.643 m |

**No real sample MP4 has been generated in the development session.** The source
evaluation retained neither RGB frames nor full action traces. Re-recording
requires the model and Habitat resources. The actual recording entry point was
attempted and failed before model/GPU startup with `Permission denied` for:

`/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz`.

The development account also cannot read the original LightNav model and MP3D
scene directory. `local_attempt/status.json` records the failed attempt; it is
not a finished video run.

Use one GPU in the authorized platform task account:

```bash
exec bash /mnt/pfs/tanshenghan/vln4project/LightNav-0/scripts/record_betanav_four_samples.sh
```

The command loads the model once and re-records precisely these four episodes
with the original model, noise seed, scene identity and STOP settings. It writes
new MP4s with full instructions, trajectory and STOP diagnostics, raw frames,
and an explicit comparison with original episode metrics. The new frames are
re-runs, not a reconstruction of the original unrecorded observations.

Expected output: `outputs/betanav_videos/<UTC>_four_samples/videos/`.
See [the usage guide](../../../docs/REVIEW_BETANAV_VIDEOS.md).

Validation: full CPU suite **1507 passed, 9 skipped, 2 deselected**; all changed
video Python files pass Ruff. Repository-wide `make check` still reports the
same 25 pre-existing errors in unrelated older files. CPU tests include seeded
selection, exact source coverage, single model load for four mock episodes,
replay drift detection, full instruction/STOP overlays, and real short-video
encode/decode checks. These tests do not establish real Habitat GPU replay.
