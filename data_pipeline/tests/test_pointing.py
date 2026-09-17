"""Current pointing geometry, annotation, collection and grid-export checks."""

import json
import shutil

import numpy as np
import pytest
from PIL import Image

from data_pipeline.annotate_r2r import annotate_episode
from data_pipeline.collect_r2r import export_targets, sha256
from data_pipeline.export_pointing_grid import export_grid
from data_pipeline.pointing import PointingConfig, backproject_pixels, ground_point, project_world
from data_pipeline.validate_pointing import validate_pointing
from lightnav.traj_vocab import RVQBundle


@pytest.fixture
def camera():
    pose = np.diag([1., -1., -1., 1.])
    pose[1, 3] = .88
    return pose, np.array([[50., 0., 50.], [0., 50., 50.], [0., 0., 1.]])


def trajectory(n=20):
    p = np.zeros((n, 3))
    p[:, 2] = -np.arange(n) * .25
    return p


def test_apos_exact_ten_ground_point_and_height(camera):
    pose, K = camera
    p = ground_point(trajectory(), 0, pose, K, np.full((100, 100), 9.),
                     np.ones((100, 100), bool), PointingConfig())
    assert p["reason"] == "t_plus_10"
    assert p["world_point"] == [0., 0., -2.5]
    assert p["pixel"] == pytest.approx([50., 67.6])
    assert p["backtrack_m"] == 0


def test_apos_depth_occlusion_backtracks_to_closest_visible_on_path(camera):
    pose, K = camera
    cfg = PointingConfig(depth_tolerance_m=0., depth_relative_tolerance=0.)
    p = ground_point(trajectory(), 0, pose, K, np.full((100, 100), 1.8),
                     np.ones((100, 100), bool), cfg)
    assert p["reason"] == "visible_path_backtrack"
    assert 1.8 - cfg.path_spacing_m <= -p["world_point"][2] <= 1.8
    assert p["selected_frame_fraction"] < 10
    assert p["backtrack_m"] == pytest.approx(2.5 + p["world_point"][2])


@pytest.mark.parametrize("mode", ["invalid_depth", "behind_camera", "stationary"])
def test_apos_unknown_never_creates_turn_stop_or_edge_token(camera, mode):
    pose, K = camera
    points, valid, t = trajectory(), np.ones((100, 100), bool), 0
    if mode == "invalid_depth":
        valid[:] = False
    elif mode == "behind_camera":
        points[:, 2] *= -1
    elif mode == "stationary":
        points[:] = 0
    p = ground_point(points, t, pose, K, np.full((100, 100), 9.), valid, PointingConfig())
    assert p["token_id"] == 0 and not p["loss_valid"] and p["pixel"] is None


@pytest.mark.parametrize("n,t", [(20, 10), (20, 19), (5, 0), (1, 0)])
def test_apos_tail_keeps_terminal_world_target_even_without_a_pixel(camera, n, t):
    pose, K = camera
    positions = trajectory(n)
    pose[:3, 3] += positions[t]
    p = ground_point(positions, t, pose, K, np.full((100, 100), 9.),
                     np.ones((100, 100), bool), PointingConfig())
    assert p["target_frame_id"] == n - 1
    assert p["target_world_point"] == positions[-1].tolist()
    if t == n - 1:
        assert p["world_point"] is None and p["pixel"] is None
        assert not p["loss_valid"]
    else:
        assert p["world_point"] == positions[-1].tolist()
        assert p["reason"] == "terminal_ground_point" and p["loss_valid"]


def test_apos_tail_occlusion_still_uses_visible_path_backtrack(camera):
    pose, K = camera
    positions = trajectory()
    pose[:3, 3] += positions[10]
    p = ground_point(positions, 10, pose, K, np.full((100, 100), 1.8),
                     np.ones((100, 100), bool), PointingConfig())
    assert p["reason"] == "visible_path_backtrack"
    assert p["target_world_point"] == positions[-1].tolist()
    assert p["world_point"] != p["target_world_point"] and p["loss_valid"]


def test_historical_pointing_metadata_preserves_masked_tail(camera):
    pose, K = camera
    cfg = PointingConfig.from_metadata({"horizon": 10})
    p = ground_point(trajectory(), 10, pose, K, np.full((100, 100), 9.),
                     np.ones((100, 100), bool), cfg)
    assert p["reason"] == "missing_t_plus_10" and not p["loss_valid"]
    assert p["token_id"] == -1  # historical metadata still validates the old encoding
    assert "target_world_point" not in p


def test_projection_roundtrip_and_world_rotation(camera):
    pose, K = camera
    angle = .7
    transform = np.eye(4)
    transform[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]]
    transform[:3, 3] = [3, 7, 11]
    pose = transform @ pose
    pixels = np.array([[14.5, 77.5], [61.5, 36.5]])
    points = backproject_pixels(pixels, [2., 4.], pose, K)
    uv, z = project_world(points, pose, K)
    np.testing.assert_allclose(uv, pixels, atol=1e-10)
    np.testing.assert_allclose(z, [2, 4])


def test_annotation_overlay_preserves_source_and_exports_both_channels(tmp_path, camera):
    pose, K = camera
    source = tmp_path / "source"
    source.mkdir()
    frames, samples = [], []
    for i, position in enumerate(trajectory(12)):
        body = np.eye(4)
        body[:3, 3] = position
        cam = pose.copy()
        cam[:3, 3] += position
        Image.new("RGB", (100, 100)).save(source / f"{i}.png")
        np.savez_compressed(source / f"{i}.npz", depth_m=np.full((100, 100), 9.),
                            valid=np.ones((100, 100), bool))
        frames.append(dict(frame_id=i, rgb=f"{i}.png", depth=f"{i}.npz",
                           world_from_body=body.tolist(), world_from_camera_cv=cam.tolist(),
                           world_from_depth_camera_cv=cam.tolist(),
                           next_expert_action=1))
        samples.append(dict(frame_id=i, next_expert_action=1, rvq_tokens="test",
                            history_start=0, history_end_inclusive=i))
    for name, records in [("frames", frames), ("samples", samples)]:
        (source / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (source / "episode.json").write_text(json.dumps(dict(complete=True, frames=12, episode_id="x",
        scene_id="synthetic", instruction="synthetic goal", camera={"K": K.tolist(), "width": 100, "height": 100})))
    (source / "manifest.json").write_text('{"format":"synthetic_test_only"}')
    np.savez_compressed(source / "targets.npz", fm_clean_se2_diff=np.zeros((12, 30, 3)))
    original = (source / "samples.jsonl").read_bytes()
    destination = tmp_path / "overlay"
    result = annotate_episode(source, destination, PointingConfig())
    (destination / "manifest.json").write_text(json.dumps({
        "format": "beta_nav_pointing_v1",
        "source_dataset": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
        "pointing_config": PointingConfig().metadata(), "episodes": ["x"]}))
    assert result["apos_valid"] == 8 and result["opos_valid"] == 12
    assert (source / "samples.jsonl").read_bytes() == original
    rows = [json.loads(s) for s in (destination / "samples.jsonl").read_text().splitlines()]
    assert all("next_expert_action" not in r for r in rows)
    assert all(r["opos_tokens"] is not None for r in rows)
    assert all(r["apos_tokens"] == f"<apos_{r['apos']['token_id']}>" for r in rows)
    assert rows[-1]["apos_tokens"] == "<apos_0>" and not rows[-1]["apos"]["loss_valid"]
    with np.load(destination / "targets.npz") as data:
        assert data["opos_valid_mask"].all()
        assert (data["opos_token_id"] >= 0).all()
        np.testing.assert_array_equal(data["apos_target_frame_id"][-10:], np.full(10, 11))
        np.testing.assert_allclose(data["apos_target_world"][-10:], np.tile(trajectory(12)[-1], (10, 1)))
    # Validate expects run/episodes/<id>; use the same output in a real overlay layout.
    run = tmp_path / "run"
    (run / "episodes").mkdir(parents=True)
    (destination / "manifest.json").replace(run / "manifest.json")
    destination.rename(run / "episodes/x")
    assert validate_pointing(run)["apos_valid"] == 8
    # Enrich a historical overlay without grid fields, without touching its labels.
    legacy = tmp_path / "legacy"
    shutil.copytree(run, legacy)
    old_episode = legacy / "episodes/x"
    old_meta = json.loads((old_episode / "episode.json").read_text())
    old_meta.pop("pointing_grid")
    (old_episode / "episode.json").write_text(json.dumps(old_meta))
    old_rows = [json.loads(s) for s in (old_episode / "samples.jsonl").read_text().splitlines()]
    for row in old_rows:
        for channel in ("apos", "opos"):
            row[channel].pop("grid")
            row[channel].pop("pixel_rc")
    (old_episode / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in old_rows))
    with np.load(old_episode / "targets.npz") as data:
        old_arrays = {k: data[k] for k in data.files if not k.endswith(("_grid_rc", "_grid_id", "_pixel_rc"))}
    np.savez_compressed(old_episode / "targets.npz", **old_arrays)
    old_hash = sha256(old_episode / "samples.jsonl")
    enriched = tmp_path / "grid_export"
    assert export_grid(legacy, enriched, "x")["apos_valid"] == 8
    assert sha256(old_episode / "samples.jsonl") == old_hash
    assert (enriched / "episodes/x/pointing_pixels.csv").is_file()
    assert not (enriched / "provenance/config.py").exists()
    with np.load(enriched / "episodes/x/targets.npz") as new_arrays:
        for name, values in old_arrays.items():
            np.testing.assert_array_equal(new_arrays[name], values)
        assert (new_arrays["opos_grid_id"][new_arrays["opos_token_id"] == 0] == -1).all()
    with pytest.raises(FileExistsError):
        export_grid(legacy, enriched, "x")
    book = np.zeros((2, 30), np.float32)
    book[1].reshape(10, 3)[:, 0] = .25
    bundle = RVQBundle(path=tmp_path, manifest={"stop": {"codes": [0, 0, 0]}}, levels=[2, 2, 2],
        horizon=10, representation="se2_diff", objective="ade_v1",
        encode={"type": "ade", "heading_weight": .3}, feature_space="weighted_diff",
        codebooks=[book, np.zeros_like(book), np.zeros_like(book)], jacobian_weights=np.ones(30),
        alpha={}, stop_l0=0)
    # Check the collector's revised export contract with actual RVQ computation.
    export_targets(source, frames, bundle, True, {"K": K}, PointingConfig())
    emitted = [json.loads(s) for s in (source / "samples.jsonl").read_text().splitlines()]
    assert len(emitted) == 12 and emitted[-1]["explicit_stop"]
    assert sum(s["apos"]["loss_valid"] for s in emitted) == 8
    assert all("next_expert_action" not in s for s in emitted)
    assert all(s["apos_tokens"] == f"<apos_{s['apos']['token_id']}>" for s in emitted)
    assert all(0 <= s["apos"]["token_id"] <= 1296 for s in emitted)
    assert emitted[-1]["apos_tokens"] == "<apos_0>" and not emitted[-1]["apos"]["loss_valid"]
    assert all(s["opos"]["target_frame_id"] == 11 for s in emitted)
    assert all(s["opos_tokens"] == "<opos_0>" for s in emitted[:6])
    assert all(s["opos"]["reason"] == "before_opos_visibility_window" for s in emitted[:6])
    assert emitted[-1]["opos"]["state"] == "not_visible"
    assert emitted[-1]["opos_tokens"] == "<opos_0>"
    with np.load(source / "targets.npz") as data:
        np.testing.assert_array_equal(data["opos_target_frame_id"], np.full(12, 11))
        np.testing.assert_allclose(data["opos_target_world"], np.tile(trajectory(12)[-1], (12, 1)))
        assert (data["opos_grid_id"][data["opos_token_id"] == 0] == -1).all()
