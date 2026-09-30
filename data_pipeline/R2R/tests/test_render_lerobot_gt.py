"""Self-contained LeRobot reads, frame identity and honest GT overlays."""
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from visual.render_episode_gt import (
    FrameRenderer,
    load_lerobot_episode,
    read_observation,
    render_video,
)


@pytest.fixture
def lerobot_episode(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("ijson")
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    K = np.array([[135., 0, 235.], [0, 135., 140.], [0, 0, 1.]])
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 2, 3] = -np.arange(4)*.5
    rows, hashes = [], []
    for t in range(4):
        camera = np.diag([1., -1., -1., 1.])
        camera[:3, 3] = [0, .88, -t*.5]
        image = Image.new("RGB", (470, 280), (50+t, 70, 90))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        depth = np.full((280, 470), 10., dtype=np.float32)
        valid = np.ones((280, 470), dtype=bool)
        ids = t+np.arange(1, 31)
        mask = ids < 4
        bounded = np.minimum(ids, 3)
        ego = np.c_[(bounded-t)*.5, np.zeros((30, 2))]
        row = {"frame_index": t, "index": t, "episode_index": 0,
               "source.episode_id": "515", "source.scene_id": "mp3d/scan/scan.glb",
               "observation.images.rgb": {"bytes": encoded.getvalue(), "path": None},
               "observation.depth_m": depth.tolist(), "observation.depth_valid": valid.tolist(),
               "observation.world_from_body": poses[t].tolist(),
               "observation.world_from_camera_cv": camera.tolist(),
               "action.fm.waypoints_ego": ego.tolist(), "action.fm.valid_mask": mask.tolist(),
               "action.fm.future_frame_ids": np.where(mask, ids, -1).tolist(),
               "action.fm.vertical_offset_m": np.zeros(30).tolist()}
        for channel in ("apos", "opos"):
            is_point = channel == "apos" and t < 2
            world = poses[-1, :3, 3]
            camera_point = np.linalg.solve(camera, np.r_[world, 1])[:3]
            uvz = K @ camera_point
            pixel = (uvz[:2]/uvz[2]).tolist() if is_point else [float("nan")]*2
            values = {"state": "point" if is_point else "not_visible", "reason": "fixture",
                      "valid_mask": [True], "token_id": [100 if is_point else 0],
                      "pixel": pixel, "world": world.tolist() if is_point else [float("nan")]*3,
                      "pixel_rc": [int(pixel[1]), int(pixel[0])] if is_point else [-1, -1],
                      "grid_rc": [2, 3] if is_point else [-1, -1], "grid_id": [100 if is_point else -1],
                      "target_world": world.tolist(), "target_frame_id": [3]}
            row.update({f"action.{channel}.{key}": value for key, value in values.items()})
        rows.append(row)
        arrays = {"observation.images.rgb": np.asarray(image), "observation.depth_m": depth,
                  "observation.depth_valid": valid}
        hashes.append({key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()})
    metadata = {"episode_index": 0, "episode_id": "515", "scene_id": "mp3d/scan/scan.glb",
                "frames": 4, "instruction": "Walk to the door.",
                "camera": {"width": 470, "height": 280, "K": K.tolist()},
                "observation_hashes": hashes, "source_episode": "/deleted/scratch/never-read"}
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "video_path": None,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {name: {} for name in rows[0]}}))
    (root / "meta/beta_nav.json").write_text(json.dumps({"episodes": [metadata]}))
    index = {"episode_index": 0, "tasks": [metadata["instruction"]], "length": 4,
             "data/chunk_index": 0, "data/file_index": 0, "dataset_from_index": 0, "dataset_to_index": 4}
    pq.write_table(pa.Table.from_pylist([index]), root / "meta/episodes/chunk-000/file-000.parquet")

    def write():
        for part in range(2):
            pq.write_table(pa.Table.from_pylist(rows[part*2:part*2+2]),
                           root / f"data/chunk-000/file-{part:03d}.parquet")

    write()
    return root, rows, write


def test_lerobot_episode_spans_files_and_does_not_depend_on_scratch(lerobot_episode):
    root, _, _ = lerobot_episode
    episode = load_lerobot_episode(root, "515", scene="scan")
    assert len(episode["frames"]) == 4
    assert len(episode["provenance"]["parquet_files"]) == 2
    rgb, depth, valid = read_observation(episode, 2)
    assert np.asarray(rgb)[0, 0].tolist() == [52, 70, 90]
    assert depth.dtype == np.float32 and valid.all()
    assert episode["samples"][0]["opos"]["loss_valid"]
    assert episode["samples"][0]["opos"]["pixel"] is None
    _, stats = FrameRenderer(episode, hide_occluded=True, number_steps=True).render(3)
    assert stats["future_steps"] == 0
    assert stats["visible_line_parts"] == stats["hidden_line_parts"] == 0


@pytest.mark.parametrize("corruption, message", [
    ("frame", "complete and contiguous"), ("future", "future mask/IDs"),
    ("pixel", "pixel/world projection mismatch"), ("source", "Parquet/provenance mismatch")])
def test_rejects_misaligned_lerobot_labels(lerobot_episode, corruption, message):
    root, rows, write = lerobot_episode
    if corruption == "frame":
        rows[1]["frame_index"] = 0
    elif corruption == "future":
        rows[0]["action.fm.future_frame_ids"][0] = 3
    elif corruption == "pixel":
        rows[0]["action.apos.pixel"][0] += 20
    else:
        rows[0]["source.episode_id"] = "different"
    write()
    with pytest.raises(ValueError, match=message):
        load_lerobot_episode(root, "515")


def test_embedded_depth_is_checked_against_provenance(lerobot_episode):
    root, rows, write = lerobot_episode
    rows[0]["observation.depth_m"][0][0] = 4.
    write()
    episode = load_lerobot_episode(root, "515")
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        read_observation(episode, 0)


def test_cached_metadata_preserves_identity_and_observation_checks(lerobot_episode):
    root, rows, write = lerobot_episode
    metadata = json.loads((root / "meta/beta_nav.json").read_text())["episodes"][0]
    original = json.dumps(metadata, sort_keys=True)
    episode = load_lerobot_episode(root, "515", scene="scan", metadata=metadata)
    assert json.dumps(metadata, sort_keys=True) == original
    assert read_observation(episode, 0)[1].dtype == np.float32
    with pytest.raises(ValueError, match="cached metadata episode/scene mismatch"):
        load_lerobot_episode(root, "different", metadata=metadata)
    with pytest.raises(ValueError, match="cached metadata episode/scene mismatch"):
        load_lerobot_episode(root, "515", scene="different", metadata=metadata)
    rows[0]["observation.depth_m"][0][0] = 4.
    write()
    episode = load_lerobot_episode(root, "515", metadata=metadata)
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        read_observation(episode, 0)


@pytest.mark.video
def test_self_contained_video_renders_every_frame_with_provenance(lerobot_episode, tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    root, _, _ = lerobot_episode
    path = tmp_path / "preview.mp4"
    result = render_video(load_lerobot_episode(root, "515"), path, scale=1, fps=2,
                          preview_frame=1, hide_occluded=True, number_steps=True, rgb_only=True)
    assert result["frames"] == 4 and result["duration_seconds"] == pytest.approx(2.)
    assert result["size"] == [470, 280]
    report = json.loads(path.with_suffix(".json").read_text())
    assert report["preview_frame"] == 1
    assert report["layout"] == "rgb_only" and report["top_view"] is None
    assert report["input_provenance"]["episode_index"] == 0
    assert report["frame_stats"][-1]["future_frame_ids"] == []
    assert all(frame["hidden_line_parts"] == 0 for frame in report["frame_stats"])
