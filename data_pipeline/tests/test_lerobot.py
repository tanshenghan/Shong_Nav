"""Actual LeRobot round trips plus mask/statistics regressions without Habitat."""

import json
import shutil

import numpy as np
import pytest
from PIL import Image

from data_pipeline.collect_r2r import export_targets, sha256
from data_pipeline.export_lerobot import RGB_KEY, export_dataset
from data_pipeline.lerobot_stats import numeric_stats
from data_pipeline.pointing import PointingConfig
from lightnav.traj_vocab import load_rvq_bundle


def test_fm_statistics_exclude_padding_and_pool_future_steps():
    values = np.array([[[1., 2., 3.], [999., 999., 999.]], [[3., 4., 5.], [999., 999., 999.]]])
    stats = numeric_stats("action.fm.waypoints_ego", values, {"action.fm.valid_mask": [[True, False], [True, False]]})
    np.testing.assert_array_equal(stats["mean"], [2., 3., 4.])
    np.testing.assert_array_equal(stats["count"], [2])


def test_point_statistics_ignore_hidden_and_unknown_placeholders():
    stats = numeric_stats("action.opos.pixel", [[np.nan, np.nan], [4., 6.], [999., 999.]],
                          {"action.opos.token_id": [0, 49, 0]})
    np.testing.assert_array_equal(stats["mean"], [4., 6.])
    assert stats["count"].item() == 1
    empty = numeric_stats("action.opos.pixel", [[np.nan, np.nan]], {"action.opos.token_id": [0]})
    assert empty["count"].item() == 0
    assert all(np.isfinite(v).all() for v in empty.values())


def test_depth_statistics_use_valid_metres_only():
    stats = numeric_stats("observation.depth_m", np.array([[[0., 2.], [4., 10.]]]),
                          {"observation.depth_valid": [[[False, True], [True, False]]]})
    np.testing.assert_array_equal(stats["mean"], [3.])
    assert stats["count"].item() == 2


def synthetic_source(tmp_path):
    source = tmp_path / "raw"
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    manifest = dict(method="rvq", horizon=10, levels=[2, 2, 2], feature_dim=30,
                    representation="se2_diff", objective="ade_v1", feature_space="weighted_diff",
                    encode={"type": "ade", "heading_weight": .3},
                    codebook_files=[f"codebook_l{i}.npy" for i in range(3)],
                    jacobian_weights_file="weights.npy", alpha_file="alpha.json", stop={"l0": 0, "codes": [0, 0, 0]})
    for i in range(3):
        book = np.zeros((2, 30), np.float32)
        if i == 0:
            book[1].reshape(10, 3)[:, 0] = .25
        np.save(bundle_path / f"codebook_l{i}.npy", book)
    np.save(bundle_path / "weights.npy", np.ones(30, np.float32))
    (bundle_path / "alpha.json").write_text("{}")
    (bundle_path / "manifest.json").write_text(json.dumps(manifest))
    bundle = load_rvq_bundle(bundle_path, 10)
    K = np.array([[10., 0., 10.], [0., 10., 8.], [0., 0., 1.]])
    camera = dict(width=20, height=16, K=K.tolist())
    for ep_id, count in [("a_stationary", 1), ("b_moving", 12)]:
        folder = source / "episodes" / ep_id
        folder.mkdir(parents=True)
        frames = []
        for t in range(count):
            body = np.eye(4)
            body[2, 3] = -.25 * t
            pose = body @ np.diag([1., -1., -1., 1.])
            pose[1, 3] = .88
            rgb = np.full((16, 20, 3), t + 10, dtype=np.uint8)
            Image.fromarray(rgb).save(folder / f"{t}.png")
            depth = np.full((16, 20), 9., np.float32)
            np.savez(folder / f"{t}.npz", depth_m=depth, valid=np.ones((16, 20), bool))
            frames.append(dict(frame_id=t, rgb=f"{t}.png", depth=f"{t}.npz", world_from_body=body.tolist(),
                               world_from_camera_cv=pose.tolist(), world_from_depth_camera_cv=pose.tolist()))
        (folder / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
        result = export_targets(folder, frames, bundle, True, camera, PointingConfig())
        (folder / "episode.json").write_text(json.dumps(dict(result, complete=True, episode_id=ep_id,
            scene_id="synthetic", instruction="Stop in " + ep_id, camera=camera, pointing_config=PointingConfig().metadata())))
    (source / "manifest.json").write_text(json.dumps(dict(rvq_bundle=str(bundle_path),
        rvq_sha256={p.name: sha256(p) for p in bundle_path.iterdir()})))
    return source


def test_official_lerobot_export_is_self_contained_and_preserves_both_episodes(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    from data_pipeline.validate_lerobot import validate_lerobot

    source = synthetic_source(tmp_path)
    output = tmp_path / "lerobot"
    exported = export_dataset(source, output)
    assert len(exported["episodes"]) == 2
    assert exported["features"][RGB_KEY]["dtype"] == "image"
    assert (output / "meta/info.json").is_file()
    assert list((output / "data").rglob("*.parquet"))
    assert validate_lerobot(output, source)["frames"] == 13
    with pytest.raises(FileExistsError):
        export_dataset(source, output)
    shutil.rmtree(source)
    report = validate_lerobot(output)
    assert report["official_loader_all_frames"] and report["dataloader_batch"]
    assert report["rgb_depth_hashes_exact"] and report["finite_masked_stats"]


def test_missing_observations_rejected_before_output_is_created(tmp_path):
    source = synthetic_source(tmp_path)
    (source / "episodes/b_moving/0.png").unlink()
    output = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        export_dataset(source, output)
    assert not output.exists()
