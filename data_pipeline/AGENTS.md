# Data pipeline working rules

- Organize collection, conversion, labeling, validation, configuration, documentation,
  tests and small reports by dataset under this directory.
- R2R and RxR share `R2R/`; start with [R2R/README.md](R2R/README.md) and follow
  [R2R/AGENTS.md](R2R/AGENTS.md) for their generation and label contracts.
- VLNVerse has separate source, scene, motion, replay, label and export stages in
  `vlnverse/`; follow [vlnverse/README.md](vlnverse/README.md) and its `AGENTS.md`.
- Preserve existing installed environments in the ignored `data_pipeline/.venv/`.
  The ignored parent `config.py` is inactive local account configuration; never
  read it into logs, provenance, release snapshots or hashes.
- Keep large raw/generated datasets under the configured data root, currently
  `/mnt/cfs/data/beta_nav`. Preserve existing data and immutable provenance.
- Keep historical reports, archives and release snapshots unchanged when moving them.
- The standalone GT renderer stays at `visual/render_episode_gt.py`; its R2R/RxR
  documentation, tests and small reports live under `R2R/`.
- Training readers stay in `src/lightnav/dataloader/`; training setup/runtime,
  configs, docs and reports follow the repository root's training convention.
- Update this index and active command paths when adding or reorganizing pipelines.
