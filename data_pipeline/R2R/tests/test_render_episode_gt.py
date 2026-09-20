"""Validate recorded-world GT projection, label alignment, and video frame counts."""

import json

import numpy as np
import pytest
from PIL import Image, ImageFont

from visual.render_episode_gt import (
    FrameRenderer,
    clip_segment,
    future_ground_path,
    load_episode,
    mark_point,
    project_path,
    render_grid_image,
    render_video,
    top_view_points,
)


def test_float32_image_boundary_is_reported_without_clamping_a_point():
    image = Image.new("RGB", (470, 280))
    before = image.tobytes()
    label = {"state": "point", "loss_valid": True, "pixel": [470., 123.9]}
    assert mark_point(image, label, "opos", "pink", 1, ImageFont.load_default()) == "boundary_not_drawn"
    assert image.tobytes() == before
    assert label["pixel"] == [470., 123.9]
    label["pixel"] = [471., 123.9]
    with pytest.raises(ValueError, match="outside the image"):
        mark_point(image, label, "opos", "pink", 1, ImageFont.load_default())
    label["pixel"] = [469.99, 123.9]
    assert mark_point(image, label, "opos", "pink", 1, ImageFont.load_default()) == "point"


def test_future_ground_path_uses_real_steps_without_terminal_padding():
    positions = np.arange(45).reshape(15, 3)
    np.testing.assert_array_equal(future_ground_path(positions, 2, 10), positions[2:13])
    np.testing.assert_array_equal(future_ground_path(positions, 12, 10), positions[12:])
    assert len(future_ground_path(positions, 14, 10)) == 1


def test_camera_segment_clips_near_plane_and_frame_boundary():
    K = np.array([[50., 0, 50], [0, 50., 50], [0, 0, 1.]])
    assert clip_segment([0, 0, -2], [0, 1, -1], K, 100, 100) is None
    pixels, depths = clip_segment([-1, 0, -.1], [1, 0, 2], K, 100, 100)
    assert np.isfinite(pixels).all() and (depths > 0).all()
    assert (pixels >= 0).all() and (pixels < 100).all()
    assert pixels[0, 0] == pytest.approx(0)
    assert pixels[1].tolist() == pytest.approx([75, 50])


def test_world_pose_and_depth_occlusion_are_applied_to_the_gt_line():
    K = np.array([[50., 0, 50], [0, 50., 50], [0, 0, 1.]])
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, 3] = [3, .88, 7]
    path = np.array([[3, 0, 6], [3, 0, 4]])
    depth = np.full((100, 100), 10.)
    depth[:75] = .5
    valid = np.ones_like(depth, dtype=bool)
    segments = project_path(path, pose, K, depth, valid, tolerance=0, relative=0)
    assert len(segments) == 1
    pixels, visible = segments[0]
    np.testing.assert_allclose(pixels[0], [50, 94])
    np.testing.assert_allclose(pixels[-1], [50, 50 + .88 / 3 * 50])
    assert visible[0] and not visible[-1]
    assert 0 < visible.sum() < len(visible)


def test_top_view_rotates_world_route_into_current_body_frame():
    pose = np.array([[0., 0., 1., 5.], [0., 1., 0., 2.],
                     [-1., 0., 0., 7.], [0., 0., 0., 1.]])
    # Body forward points world -X; body right points world -Z.
    path = np.array([[5., 2., 7.], [4., 2., 7.], [4., 2., 6.]])
    pixels = top_view_points(path, pose, center=[100, 100], radius=50, extent_m=2)
    np.testing.assert_allclose(pixels, [[100, 100], [100, 75], [125, 75]])


@pytest.fixture
def episode(tmp_path):
    source, labels = tmp_path / "source", tmp_path / "labels"
    source.mkdir()
    labels.mkdir()
    K = [[135., 0, 235.], [0, 135., 140.], [0, 0, 1.]]
    frames, samples = [], []
    for t in range(4):
        pose = np.eye(4)
        pose[2, 3] = -t * .5
        camera = np.diag([1., -1., -1., 1.])
        camera[:3, 3] = [0, .88, -t * .5]
        Image.new("RGB", (470, 280), (80, 70, 60)).save(source / f"{t}.png")
        np.savez_compressed(source / f"{t}.npz", depth_m=np.full((280, 470), 10.),
                            valid=np.ones((280, 470), dtype=bool))
        frames.append(dict(frame_id=t, rgb=f"{t}.png", depth=f"{t}.npz",
                           world_from_body=pose.tolist(), world_from_camera_cv=camera.tolist()))
        samples.append(dict(frame_id=t, array_row=t,
                            apos=dict(state="point", loss_valid=True, pixel=[235, 220]),
                            opos=dict(state="unknown", loss_valid=False, pixel=None)))
    (source / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
    (source / "samples.jsonl").write_text("\n".join(json.dumps(dict(frame_id=t)) for t in range(4)))
    (labels / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
    metadata = dict(complete=True, episode_id="synthetic", frames=4,
                    instruction="Walk to the door. Stop just inside the door.",
                    camera=dict(width=470, height=280, K=K))
    (source / "episode.json").write_text(json.dumps(metadata))
    (labels / "episode.json").write_text(json.dumps(dict(metadata, source_episode=str(source))))
    return source, labels


def test_overlay_reads_original_observations_and_only_marks_valid_points(episode):
    source, labels = episode
    data = load_episode(source, annotations=labels)
    assert data["source"] == source and data["labels"] == labels
    renderer = FrameRenderer(data)
    first, stats = renderer.render(0)
    assert stats["apos"] == "point" and stats["opos"] == "unknown"
    assert stats["future_steps"] == 3 and stats["visible_line_parts"] > 0
    assert first.size[0] == 1220 and first.size[1] > 560
    assert all(size % 2 == 0 for size in first.size)
    _, last = renderer.render(3)
    assert last["future_steps"] == 0
    assert last["visible_line_parts"] == last["hidden_line_parts"] == 0
    data["samples"][1]["opos"] = dict(state="point", loss_valid=True, pixel=[180, 150])
    assert renderer.render(1)[1]["opos"] == "point"
    data["samples"][1]["apos"]["loss_valid"] = False
    assert renderer.render(1)[1]["apos"] != "point"


def test_misaligned_annotations_fail_instead_of_drawing_on_the_wrong_frame(episode):
    source, labels = episode
    rows = [json.loads(s) for s in (labels / "samples.jsonl").read_text().splitlines()]
    rows[1]["frame_id"] = 2
    (labels / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in rows))
    with pytest.raises(ValueError, match="matching contiguous IDs"):
        load_episode(source, annotations=labels)


def test_grid_figure_marks_the_original_pixel_cell_and_keeps_missing_opos_empty(episode, tmp_path):
    _, labels = episode
    data = load_episode(labels)
    output = tmp_path / "grid.png"
    result = render_grid_image(data, output, frame_id=0)
    assert result["pixel_rc"] == [220, 235]
    assert result["grid"] == dict(row=21, col=24, cell_id=1033)
    with Image.open(output) as image:
        assert image.width > 940 and image.height > 560
        image.verify()
    missing = render_grid_image(data, tmp_path / "missing.png", frame_id=1, channel="opos")
    assert missing["grid"] is None and missing["pixel"] is None
    with pytest.raises(FileExistsError):
        render_grid_image(data, output)


def test_grid_figure_rejects_mismatched_saved_cell(episode, tmp_path):
    _, labels = episode
    data = load_episode(labels)
    data["samples"][0]["apos"]["grid"] = dict(row=0, col=0, cell_id=1)
    with pytest.raises(ValueError, match="does not match"):
        render_grid_image(data, tmp_path / "wrong.png")


@pytest.mark.video
def test_mp4_contains_every_frame_and_refuses_overwrite(episode, tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    _, labels = episode
    output = tmp_path / "gt.mp4"
    result = render_video(load_episode(labels), output, fps=2, scale=1)
    assert result["frames"] == 4 and result["duration_seconds"] == pytest.approx(2., abs=.1)
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["frame_stats"][-1]["future_steps"] == 0
    assert output.with_suffix(".png").is_file()
    with pytest.raises(FileExistsError):
        render_video(load_episode(labels), output)
