"""Training split metadata and FM normalization must exclude held-out scenes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from lightnav.training.prepare_data import combine_statistics, make_manifest  # noqa: E402


def _stats(index, values):
    return {"episode_index": index, "length": 4,
            "stats/action.fm.clean_se2_diff/count": [len(values)],
            "stats/action.fm.clean_se2_diff/mean": values.mean(0).tolist(),
            "stats/action.fm.clean_se2_diff/std": values.std(0).tolist()}


def test_parallel_variance_matches_valid_raw_steps_and_preserves_zero():
    values = np.arange(21).reshape(7, 3) * .02
    rows = [_stats(0, values[:2]), _stats(1, values[2:])]
    rows.append({"stats/action.fm.clean_se2_diff/count": [0]})
    result = combine_statistics(rows, [.01, .02, .03])
    np.testing.assert_allclose(result["std"], values.std(0))
    np.testing.assert_allclose(result["mean"], values.mean(0))
    assert result["valid_steps"] == 7
    np.testing.assert_array_equal(np.zeros(3) / result["action_scale"], 0)


def test_scene_split_keeps_shared_routes_together_and_fits_training_only(tmp_path):
    meta = tmp_path / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    episodes = [{"episode_index": i, "scene_id": f"scene-{i // 2}", "frames": 4} for i in range(4)]
    (meta / "beta_nav.json").write_text(json.dumps({"episodes": episodes}))
    (meta / "info.json").write_text(json.dumps(dict(total_episodes=4, total_frames=16)))
    arrays = [np.full((3, 3), i * 10.) + np.arange(9).reshape(3, 3) for i in range(4)]
    pq.write_table(pa.Table.from_pylist([_stats(i, a) for i, a in enumerate(arrays)]),
                   meta / "episodes/chunk-000/file-000.parquet")
    manifest = make_manifest(tmp_path, val_fraction=.5)
    assert manifest["train_episodes"] in ([0, 1], [2, 3])
    assert not set(manifest["train_scenes"]) & set(manifest["val_scenes"])
    training = np.concatenate([arrays[i] for i in manifest["train_episodes"]])
    np.testing.assert_allclose(manifest["normalization"]["std"], training.std(0))
    assert manifest["train_frames"] == manifest["val_frames"] == 8
    all_training = make_manifest(tmp_path, val_fraction=0)
    assert all_training["split_by"] == "all"
    assert all_training["train_episodes"] == list(range(4))
    assert all_training["train_frames"] == 16 and all_training["val_frames"] == 0
    assert all_training["val_episodes"] == all_training["val_scenes"] == []
    np.testing.assert_allclose(all_training["normalization"]["std"], np.concatenate(arrays).std(0))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(manifest))
    continued = make_manifest(tmp_path, val_fraction=0, normalization_from=frozen)
    assert continued["normalization"] == manifest["normalization"]
    assert continued["normalization_source"]["path"] == str(frozen)
    from lightnav.training.runtime import read_data_manifest

    all_path = tmp_path / "all.json"
    all_path.write_text(json.dumps(continued))
    config = SimpleNamespace(dataset_root=str(tmp_path), data_manifest=str(all_path), eval_every=0)
    assert read_data_manifest(config)["train_frames"] == 16
    config.eval_every = 500
    with pytest.raises(ValueError, match="eval_every=0"):
        read_data_manifest(config)
