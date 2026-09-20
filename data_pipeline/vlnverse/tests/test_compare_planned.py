"""Reference and pose playback keep distinct clocks and preserve old turn intervals."""

import hashlib
import json

import numpy as np
import pytest

from data_pipeline.vlnverse.replay.compare_planned import (
    load_trajectory,
    main,
    normalized_reference_schedule,
    validate_comparison_identity,
    validate_render_manifest,
)


def test_normalized_time_alignment_preserves_in_place_turn_samples():
    # Poses 1..3 occupy the same XY but have changing yaw. A path-distance
    # schedule would collapse this interval; the reference-time schedule keeps it.
    old_positions = np.array([[0, 0], [1, 0], [1, 0], [1, 0], [2, 0], [3, 0]])
    mapped, indices = normalized_reference_schedule([0, .2, .4, .6, .8, 1.], np.arange(31)/15)
    np.testing.assert_allclose(mapped, np.arange(31)/30)
    assert set(indices) == set(range(6))
    assert all(np.count_nonzero(indices == i) >= 5 for i in (1, 2, 3))
    np.testing.assert_array_equal(old_positions[1], old_positions[3])
    assert indices[0] == 0 and indices[-1] == 5
    assert (np.diff(indices) >= 0).all()


def test_sparse_video_schedule_exposes_omitted_previous_samples():
    _, indices = normalized_reference_schedule(np.arange(31)/30, np.arange(4)/15)
    np.testing.assert_array_equal(indices, [0, 10, 20, 30])
    assert len(np.flatnonzero(np.bincount(indices, minlength=31) == 0)) == 27


def test_pose_frames_alignment_uses_saved_nonuniform_time_not_video_index():
    mapped, indices = normalized_reference_schedule(
        [0, 1, 2, 3, 4], [0, .3, .9, 1.2], playback_mode="pose_frames")
    np.testing.assert_allclose(mapped, [0, 1, 3, 4])
    np.testing.assert_array_equal(indices, [0, 1, 3, 4])


@pytest.mark.parametrize("times", [[0], [0, 0], [0, 2, 1], [1, 2], [0, np.nan]])
def test_pose_frames_still_rejects_invalid_reference_clocks(times):
    with pytest.raises(ValueError, match="reference timestamps"):
        normalized_reference_schedule([0, 1], times, playback_mode="pose_frames")


def test_unknown_playback_mode_is_rejected():
    with pytest.raises(ValueError, match="playback_mode"):
        normalized_reference_schedule([0, 1], [0, 1/15], playback_mode="unknown")


@pytest.mark.parametrize("old,new", [
    ([0, 1], [0, 1/30]), ([0, 1], [0, .1, .2]), ([0, 1], [0]),
    ([1, 2], [0, 1/15]), ([0, 0], [0, 1/15]), ([0, np.nan], [0, 1/15]),
])
def test_invalid_or_non_15hz_clocks_are_rejected(old, new):
    with pytest.raises(ValueError):
        normalized_reference_schedule(old, new)


def _save_trajectory(path, count, duration):
    positions = np.column_stack([np.linspace(0, 1, count), np.zeros((count, 2))])
    values = dict(positions_xyz=positions, yaw_rad=np.zeros(count), timestamp_s=np.linspace(0, duration, count),
                  velocity_xyz=np.zeros((count, 3)), yaw_rate_rad_s=np.zeros(count))
    np.savez(path, **values)
    return values


def test_trajectory_rejects_nonfinite_reference_derivatives(tmp_path):
    path = tmp_path / "trajectory.npz"
    values = _save_trajectory(path, 3, 2/15)
    values["velocity_xyz"][1, 0] = np.nan
    np.savez(path, **values)
    with pytest.raises(ValueError, match="velocity_xyz"):
        load_trajectory(path, planned=True)


def test_pose_frames_loads_4_5_hz_without_rewriting_reference_timestamps(tmp_path):
    path = tmp_path / "trajectory.npz"
    expected = _save_trajectory(path, 200, 199/4.5)
    with pytest.raises(ValueError, match="15 Hz"):
        load_trajectory(path, planned=True)
    loaded = load_trajectory(path, planned=True, playback_mode="pose_frames")
    np.testing.assert_array_equal(loaded["timestamp_s"], expected["timestamp_s"])
    assert loaded["timestamp_s"][-1] > (len(loaded["timestamp_s"])-1)/15


def _metadata():
    return dict(episode_id="scene_1_0", instruction="Walk forward to the doorway.",
                source={"source_sha256": "official_source"}, quality={"valid": True}, config={})


@pytest.mark.parametrize("field,value", [("episode_id", "wrong_episode"), ("instruction", "Another instruction"),
                                         ("quality", {"valid": False}), ("source", {"source_sha256": "wrong"})])
def test_episode_instruction_and_source_identity_must_match(field, value):
    previous, planned = _metadata(), _metadata()
    planned[field] = value
    with pytest.raises(ValueError):
        validate_comparison_identity(previous, planned)


def _render_manifest(trajectory, count):
    return dict(source_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(), input_frames=count,
                rendered_frames=count, camera=dict(width=320, height=240, hfov_deg=90,
                height_above_source_z_m=1.25, pitch_down_deg=0), future_path={"enabled": False},
                scene_usd="/scene/start_result_navigation.usd", renderer="Blender CPU test")


@pytest.mark.parametrize("field,value", [("source_sha256", "wrong"), ("rendered_frames", 1),
                                         ("future_path", {"enabled": True})])
def test_mismatched_partial_or_preannotated_renders_are_rejected(tmp_path, field, value):
    path = tmp_path / "trajectory.npz"
    _save_trajectory(path, 3, 2/15)
    manifest = _render_manifest(path, 3)
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_render_manifest(manifest, path, 3)


def test_camera_pitch_cannot_be_silently_ignored(tmp_path):
    path = tmp_path / "trajectory.npz"
    _save_trajectory(path, 3, 2/15)
    manifest = _render_manifest(path, 3)
    manifest["camera"]["pitch_down_deg"] = 20
    with pytest.raises(ValueError, match="horizontal"):
        validate_render_manifest(manifest, path, 3)


@pytest.mark.parametrize("playback_mode,count,hz", [("reference_time", 7, 15), ("pose_frames", 37, 4.5)])
def test_cli_writes_planned_and_comparison_with_exact_clock_and_future_indices(tmp_path, playback_mode, count, hz):
    av = pytest.importorskip("av")
    pytest.importorskip("cv2")
    pytest.importorskip("lightnav.viz.render")
    from PIL import Image

    previous, planned = tmp_path / "previous", tmp_path / "planned"
    before, after = tmp_path / "before_images", tmp_path / "after_images"
    for path in (previous, planned, before, after):
        path.mkdir()
    _save_trajectory(previous / "trajectory.npz", 3, 1.)
    _save_trajectory(planned / "trajectory.npz", count, (count-1)/hz)
    np.savez(previous / "source_poses.npz", positions_xyz=[[0, 0, 0], [1, 0, 0]], yaw_rad=[0, 0])
    metadata = _metadata()
    metadata["source_poses_sha256"] = hashlib.sha256((previous / "source_poses.npz").read_bytes()).hexdigest()
    (previous / "manifest.json").write_text(json.dumps(metadata))
    (planned / "manifest.json").write_text(json.dumps(_metadata()))
    for directory, trajectory_dir, frame_count in ((before, previous, 3), (after, planned, count)):
        (directory / "render_manifest.json").write_text(json.dumps(_render_manifest(trajectory_dir / "trajectory.npz", frame_count)))
        for index in range(frame_count):
            Image.new("RGB", (320, 240), ((40 + index*10) % 256, 80, 120)).save(directory / f"{index:06d}.png")
    output = tmp_path / "videos"
    arguments = ["--previous-retimed-dir", str(previous), "--previous-images", str(before),
                 "--planned-dir", str(planned), "--planned-images", str(after), "--output-dir", str(output)]
    if playback_mode != "reference_time":
        arguments.extend(["--playback-mode", playback_mode])
    main(arguments)
    report = json.loads((output / "comparison_manifest.json").read_text())
    assert report["video_frames"] == count and report["video_fps"] == 15
    assert report["video_duration_s"] == count/15
    assert report["terminal_frame_hold_s"] == 1/15
    assert report["playback_mode"] == playback_mode
    assert report["planned_reference_timing_preserved"] is (playback_mode == "reference_time")
    assert report["previous_reference_timing_preserved"] is False
    assert report["previous_omitted_pose_indices"] == []
    assert report["frames"][0]["planned_future_indices"] == list(range(1, min(count, 31)))
    assert report["frames"][-1]["planned_future_indices"] == []
    assert [row["planned_pose_index"] for row in report["frames"]] == list(range(count))
    for index, row in enumerate(report["frames"]):
        assert row["planned_reference_timestamp_s"] == pytest.approx(index/hz)
        assert row["playback_timestamp_s"] == pytest.approx(index/15)
        assert row["planned_future_horizon_seconds"] == pytest.approx(min(30, count-1-index)/hz)
        assert row["future_horizon_seconds"] == row["planned_future_horizon_seconds"]
    assert report["frames"][-1]["planned_future_horizon_seconds"] == 0
    if playback_mode == "pose_frames":
        assert report["planned_full_future_horizon_s"] is None
        assert report["frames"][0]["planned_future_horizon_seconds"] == pytest.approx(30/4.5)
        assert report["frames"][-1]["planned_reference_timestamp_s"] != report["frames"][-1]["playback_timestamp_s"]
        assert "accelerated preview" in report["planned_clock"]
        assert "neither panel" in report["shared_viz"]["hud_velocity"]
    else:
        assert report["planned_full_future_horizon_s"] == 2
    assert " ".join(report["instruction_layout"]["planned"]["lines"]) == _metadata()["instruction"]
    for filename in ("planned.mp4", "before_after.mp4"):
        with av.open(str(output / filename)) as video:
            frames = list(video.decode(video=0))
        assert len(frames) == count
        np.testing.assert_allclose([float(frame.pts * frame.time_base) for frame in frames], np.arange(count)/15)
    with pytest.raises(FileExistsError):
        main(arguments)
