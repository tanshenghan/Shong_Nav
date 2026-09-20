# Last4FM: BetaNav source snapshot

This branch contains the current LightNav/BetaNav code, configurations, tests and
technical documentation. It is a clean source snapshot with no inherited Git
history containing experiment artifacts.

## Main entry points

- Model: `src/lightnav/inference/beta_model.py`
- Training: `train_betanav.py`, `src/lightnav/training/`
- LeRobot reader: `src/lightnav/dataloader/Train_Lerobot_DataLoader.py`
- FM-only horizon-10 experiments: `configs/betanav/r2r_fm_only_h10_beta*.yaml`
- Training guide: [docs/TRAIN_BETANAV.md](docs/TRAIN_BETANAV.md)
- Model contract: [docs/BETA_MODEL.md](docs/BETA_MODEL.md)
- Habitat evaluation: `scripts/eval_habitat.sh`, [docs/EVAL_HABITAT.md](docs/EVAL_HABITAT.md)

The three horizon-10 configurations freeze the original LightNav VLM, train a fresh
FM expert with `1.0 * FM`, and use Beta(1, 1.5), Beta(1, 2), or Beta(1, 3) time
sampling. They predict ten future local SE(2) increments of shape `[B, 10, 3]`.
FM-only training has no CE/RVQ targets. Navigation evaluation can use frozen
LightNav's RVQ STOP decision while FM generates movement.

## External inputs

Datasets, model weights, optimizer/checkpoint state, dataset-specific manifests,
experiment reports, archives, videos, and scene meshes/textures are intentionally
not distributed. Paths in the experiment YAML files describe the original server;
set `dataset_root`, `model_path`, `data_manifest` and `output_dir` for your machine.
README illustrations and the MuJoCo scene assets are also omitted; the MuJoCo demo
requires its external scene assets before it can run. Some historical documents
link to local reports or manifests that are not part of this source snapshot.

Prepare a fresh all-training manifest from your own LeRobot data before training:

```bash
bash scripts/setup_betanav_train.sh
.venv-train/bin/python -m lightnav.training.prepare_data \
  /path/to/lerobot_dataset \
  --output configs/betanav/data/r2r_all_train_20260913.json \
  --seed 42 --val-fraction 0 --scale-floor 0.05 0.02 0.05
```

This recomputes normalization for a new run. Exact continuation of an existing
checkpoint additionally requires its original manifest/normalization and complete
resume state, supplied separately. See the training guide for launch and resume.

## Attribution

Based on [Light Origins' LightNav-0](https://github.com/lightorigins/LightNav-0).
Original source commit before this snapshot: `0e0e5cbf` plus working-tree changes.
Upstream documentation and attribution are retained. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component licenses.
