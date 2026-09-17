"""Final destination annotation, CPU entry point and historical read compatibility."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_pipeline.annotate_r2r import annotate_episode, provenance_code_paths
from data_pipeline.pointing import PointingConfig, terminal_ground_point
from data_pipeline.validate_pointing import validate_pointing


def make_source(tmp_path, *, moving=True, step_m=.25):
    source = tmp_path / "source" / "episodes" / "test"
    source.mkdir(parents=True)
    (source.parents[1] / "manifest.json").write_text("{}")
    frames = []
    K = np.array([[50., 0, 50.], [0, 50., 50.], [0, 0, 1.]])
    for i in range(12):
        body = np.eye(4)
        body[2, 3] = -i * step_m if moving else 0.
        camera = body @ np.diag([1., -1., -1., 1.])
        camera[1, 3] = .88
        Image.new("RGB", (100, 100)).save(source / f"{i}.png")
        np.savez(source / f"{i}.npz", depth_m=np.full((100, 100), 2.), valid=np.ones((100, 100), bool))
        frames.append(dict(frame_id=i, rgb=f"{i}.png", depth=f"{i}.npz",
                           world_from_body=body.tolist(), world_from_camera_cv=camera.tolist()))
    (source / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
    (source / "samples.jsonl").write_text("\n".join(json.dumps(dict(frame_id=i, history_end_inclusive=i))
                                                       for i in range(12)))
    (source / "episode.json").write_text(json.dumps(dict(complete=True, frames=12, episode_id="test",
        scene_id="synthetic", instruction="Stop just inside the doorway.",
        camera=dict(width=100, height=100, K=K.tolist()))))
    np.savez(source / "targets.npz", fm_clean_se2_diff=np.zeros((12, 30, 3)))
    return source, frames, K



def test_opos_keeps_the_same_endpoint_when_occluded_or_beneath_camera(tmp_path):
    source, _, _ = make_source(tmp_path)

    folder = tmp_path / "labels"
    result = annotate_episode(source, folder, PointingConfig(opos_visibility_window="all"))
    assert result["opos_bound"] and result["opos_search"] is None
    assert not result["object_bound"] and result["opos_visible"] > 0
    assert not (folder / "detections").exists()
    rows = [json.loads(s) for s in (folder / "samples.jsonl").read_text().splitlines()]
    assert all(r["opos"]["target_world_point"] == [0., 0., -2.75] for r in rows)
    assert all(r["opos"]["target_frame_id"] == 11 for r in rows)
    assert rows[0]["apos"]["target_frame_id"] == 10
    assert rows[0]["opos"]["state"] == "not_visible"  # occluded by depth at 2 m
    assert rows[-1]["opos"]["state"] == "not_visible"  # beneath the terminal camera
    assert rows[-1]["opos"]["pixel"] is None
    with np.load(folder / "targets.npz") as data:
        np.testing.assert_array_equal(data["opos_target_frame_id"], np.full(12, 11))
        np.testing.assert_array_equal(data["opos_target_world"], np.tile([0., 0., -2.75], (12, 1)))



def test_default_annotation_cli_needs_neither_api_config_nor_semantic_assets(tmp_path, monkeypatch):
    import data_pipeline.annotate_r2r as annotate

    source, _, _ = make_source(tmp_path, step_m=.5)
    output = tmp_path / "default_run"

    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.name == "config.py":
            pytest.fail("final annotation must not read local API configuration")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr("sys.argv", ["annotate_r2r.py", str(source.parents[1]),
                        "--output-dir", str(output)])
    annotate.main()
    report = json.loads((output / "manifest.json").read_text())
    assert report["opos_mode"] == "terminal_ground" and not report["api_requested"]
    assert report["pointing_config"]["opos_visibility_window"] == "last_half"
    assert validate_pointing(output)["opos_visible"] == 3
    folder = output / "episodes/test"
    assert json.loads((folder / "episode.json").read_text())["opos_visibility_start_frame"] == 6
    rows = [json.loads(s) for s in (folder / "samples.jsonl").read_text().splitlines()]
    assert all(r["opos_tokens"] == "<opos_0>" and r["opos"]["loss_valid"] for r in rows[:6])
    assert all(r["opos"]["reason"] == "before_opos_visibility_window" for r in rows[:6])
    assert [r["frame_id"] for r in rows if r["opos"]["state"] == "point"] == [7, 8, 9]
    assert rows[-1]["opos"]["reason"] == "stop_location_outside_view"


@pytest.mark.parametrize("config", [PointingConfig(opos_mode="doubao"), PointingConfig(tail_policy="mask"),
                                   PointingConfig(unknown_token_id=-1)])
def test_generation_rejects_retired_definitions_before_creating_output(tmp_path, config):
    source, _, _ = make_source(tmp_path)
    output = tmp_path / "retired"
    with pytest.raises(ValueError, match="final GT generation requires"):
        annotate_episode(source, output, config)
    assert not output.exists()


def test_provenance_excludes_credentials_and_retired_modules(tmp_path):
    for name in ("config.py", "doubao.py", "annotate_r2r.py", "pointing.py"):
        (tmp_path / name).write_text("placeholder")
    assert [p.name for p in provenance_code_paths(tmp_path)] == ["annotate_r2r.py", "pointing.py"]
    historical = PointingConfig.from_metadata({})
    assert historical.tail_policy == "mask" and historical.opos_mode == "doubao"
    assert historical.opos_search_frames == 1
    assert historical.unknown_token_id == -1
    assert historical.opos_visibility_window == "all"
    assert PointingConfig().metadata()["opos_visibility_window"] == "last_half"
    assert PointingConfig().metadata()["unknown_token_id"] == 0
    assert "opos_search_frames" not in PointingConfig().metadata()


def test_opos_invalid_depth_keeps_endpoint_and_masks_pixel(tmp_path):
    _, frames, K = make_source(tmp_path)
    positions = np.asarray([f["world_from_body"] for f in frames])[:, :3, 3]
    point = terminal_ground_point(positions, frames[0]["world_from_camera_cv"], K,
                                  np.full((100, 100), 3.), np.zeros((100, 100), bool),
                                  PointingConfig(opos_visibility_window="all"), frame_index=0)
    assert point["state"] == "unknown" and not point["loss_valid"]
    assert point["target_world_point"] == positions[-1].tolist()
    assert point["pixel"] is None and point["token_id"] == 0


def test_masked_zero_opos_is_serialized_and_validator_keeps_it_distinct(tmp_path, monkeypatch):
    import data_pipeline.annotate_r2r as annotate

    source, _, _ = make_source(tmp_path, step_m=.5)
    # A goal in the frustum with invalid depth is unknown; later offscreen goals
    # remain known not-visible. Both carry token 0, with different loss masks.
    np.savez(source / "8.npz", depth_m=np.full((100, 100), 3.), valid=np.zeros((100, 100), bool))
    output = tmp_path / "zero_labels"
    monkeypatch.setattr("sys.argv", ["annotate_r2r.py", str(source.parents[1]), "--output-dir", str(output)])
    annotate.main()
    folder = output / "episodes/test"
    rows = [json.loads(s) for s in (folder / "samples.jsonl").read_text().splitlines()]
    assert rows[8]["opos_tokens"] == rows[-1]["opos_tokens"] == "<opos_0>"
    assert rows[8]["apos_tokens"] == "<apos_0>"
    assert not rows[8]["opos"]["loss_valid"] and rows[-1]["opos"]["loss_valid"]
    with np.load(folder / "targets.npz") as data:
        for channel in ("apos", "opos"):
            assert (data[channel + "_token_id"] >= 0).all()
            assert np.isnan(data[channel + "_pixel"][8]).all()
            assert (data[channel + "_grid_rc"][8] == -1).all()
    assert validate_pointing(output)["frames"] == 12
    # Historical overlays retain their original negative sentinel and null text.
    historical = tmp_path / "historical_labels"
    shutil.copytree(output, historical)
    old_folder = historical / "episodes/test"
    for path in (historical / "manifest.json", old_folder / "episode.json"):
        metadata = json.loads(path.read_text())
        metadata["pointing_config"].pop("unknown_token_id")
        path.write_text(json.dumps(metadata))
    old_rows = json.loads(json.dumps(rows))
    for row in old_rows:
        for channel in ("apos", "opos"):
            if row[channel]["state"] == "unknown":
                row[channel]["token_id"] = -1
                row[channel + "_tokens"] = None
    (old_folder / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in old_rows))
    with np.load(old_folder / "targets.npz") as data:
        old_arrays = {k: data[k] for k in data.files}
    for channel in ("apos", "opos"):
        old_arrays[channel + "_token_id"][~old_arrays[channel + "_valid_mask"]] = -1
    np.savez(old_folder / "targets.npz", **old_arrays)
    assert validate_pointing(historical)["frames"] == 12
    # Reject data corruption that turns unknown depth into a positive class label.
    rows[8]["opos"]["loss_valid"] = True
    (folder / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(AssertionError):
        validate_pointing(output)
