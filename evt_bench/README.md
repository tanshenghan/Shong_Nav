# EVT-Bench integration files

Files to copy into your own [EVT-Bench / TrackVLA](https://github.com/wsakobe/TrackVLA)
checkout so its `run.py` can drive a running `lightnav-serve` server over WebSocket.

| File | Purpose |
|------|---------|
| `trackvla_client_agent.py` | Minimal WebSocket client agent (`evaluate_agent`, `TrackVLAClientAgent`). Copy next to `run.py`. Python 3.9, needs `websocket-client`. |
| `run_py.patch` | Unified diff adding the `--model-name trackvla` branch to upstream `run.py` (`git apply` / `patch -p1`). |
| `patch_task_config.py` | Writes a patched copy of `track_infer_*.yaml` with a different jaw camera hfov / height. Used by `scripts/eval_evt_bench.sh`; not copied into EVT-Bench. |

The end-to-end recipe (conda env, dataset layout, server launch, sharded run,
`analyze_results.py`) is in [docs/EVAL_EVT_BENCH.md](../docs/EVAL_EVT_BENCH.md).

EVT-Bench itself (habitat-lab fork, task configs, datasets, `analyze_results.py`)
is licensed CC BY-NC-SA 4.0 and is not redistributed here; the evaluation loop in
`trackvla_client_agent.py` is adapted from its driver, see `THIRD_PARTY_NOTICES.md`.
