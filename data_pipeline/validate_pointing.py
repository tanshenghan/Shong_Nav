"""Read-only validation of the pointing overlay, including its source data alignment."""

import json
from pathlib import Path

import numpy as np

from data_pipeline.collect_r2r import sha256
from data_pipeline.pointing import (
    PointingConfig,
    attach_pixel_grid,
    fixed_world_point,
    ground_point,
    pixel_grid_arrays,
    pixel_grid_metadata,
    pointing_token,
    project_world,
    terminal_ground_point,
)
from lightnav.vln_utils import encode_point_pixel


def validate_pointing(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    source = Path(manifest["source_dataset"])
    assert sha256(source / "manifest.json") == manifest["source_manifest_sha256"]
    config = PointingConfig.from_metadata(manifest["pointing_config"])
    totals = dict(episodes=0, frames=0, apos_valid=0, opos_visible=0, opos_not_visible=0,
                  opos_temporally_hidden=0)
    for folder in sorted((root / "episodes").iterdir()):
        episode = json.loads((folder / "episode.json").read_text())
        assert episode["complete"]
        original = Path(episode["source_episode"])
        for name, digest in episode["source_hashes"].items():
            assert sha256(original / name) == digest
        if "source_annotation" in episode:
            for name, digest in episode["source_annotation_hashes"].items():
                assert sha256(Path(episode["source_annotation"]) / name) == digest
        frames = [json.loads(s) for s in (original / "frames.jsonl").read_text().splitlines()]
        rows = [json.loads(s) for s in (folder / "samples.jsonl").read_text().splitlines()]
        binding = json.loads((folder / "object_binding.json").read_text())
        K = np.asarray(episode["camera"]["K"])
        poses = np.asarray([f["world_from_body"] for f in frames])
        stop_location = binding.get("target_type") == "stop_location"
        if stop_location:
            assert binding["bound"] and binding["target_frame_id"] == len(frames) - 1
            np.testing.assert_array_equal(binding["world_point"], poses[-1, :3, 3])
        if episode.get("opos_search") is not None:
            # Read old search records without importing the retired API pipeline.
            search = json.loads((folder / "opos_search.json").read_text())
            assert search == episode["opos_search"]
            candidates = list(range(max(0, len(frames) - config.opos_search_frames), len(frames)))
            assert search["candidate_frame_ids"] == candidates
            attempts = search["attempts"]
            assert [a["frame_id"] for a in attempts] == candidates[:len(attempts)]
            assert not any(a["bound"] for a in attempts[:-1])
            if binding["bound"]:
                assert attempts[-1]["bound"]
                assert search["selected_frame_id"] == binding["anchor_frame_id"] == attempts[-1]["frame_id"]
            else:
                assert len(attempts) == len(candidates) and search["selected_frame_id"] is None
        assert len(rows) == len(frames) == episode["frames"]
        if "opos_visibility_start_frame" in episode:
            assert episode["opos_visibility_start_frame"] == config.opos_visibility_start_frame(len(rows))
        with np.load(folder / "targets.npz", allow_pickle=False) as data:
            if stop_location and config.opos_mode == "terminal_ground":
                assert episode["opos_search"] is None
                np.testing.assert_array_equal(data["opos_target_frame_id"], np.full(len(rows), len(rows) - 1))
                np.testing.assert_array_equal(data["opos_target_world"], np.tile(poses[-1, :3, 3], (len(rows), 1)))
            if "pointing_grid" in episode:
                grid = episode["pointing_grid"]
                w, h = grid["image_size_wh"]
                assert grid == pixel_grid_metadata(w, h)
                for channel in ("apos", "opos"):
                    points = [attach_pixel_grid(r[channel], w, h) for r in rows]
                    assert points == [r[channel] for r in rows]
                    for name, expected in pixel_grid_arrays(points, channel).items():
                        assert data[name].dtype == np.int32
                        np.testing.assert_array_equal(data[name], expected)
            if config.tail_policy == "terminal":
                indices = np.minimum(np.arange(len(rows)) + config.horizon, len(rows) - 1)
                np.testing.assert_array_equal(data["apos_target_frame_id"], indices)
                np.testing.assert_allclose(data["apos_target_world"], poses[indices, :3, 3], atol=1e-8)
            for channel in ("apos", "opos"):
                assert data[channel + "_pixel"].shape == (len(rows), 2)
                assert data[channel + "_world"].shape == (len(rows), 3)
                assert data[channel + "_valid_mask"].dtype == bool
            for t, (row, frame) in enumerate(zip(rows, frames)):
                assert row["frame_id"] == row["array_row"] == t
                assert row["history_end_inclusive"] <= t
                assert "next_expert_action" not in row
                assert Path(row["current_rgb"]) == (original / frame["rgb"]).resolve()
                with np.load(original / frame["depth"], allow_pickle=False) as depth:
                    expected = ground_point(poses[:, :3, 3], t, frame["world_from_camera_cv"], K,
                                            depth["depth_m"], depth["valid"], config)
                    if "pointing_grid" in episode:
                        h, w = depth["depth_m"].shape
                        assert episode["pointing_grid"]["image_size_wh"] == [w, h]
                        expected = attach_pixel_grid(expected, w, h)
                    assert expected == row["apos"]
                    h, w = depth["depth_m"].shape
                    if stop_location:
                        if config.opos_mode == "terminal_ground":
                            expected_opos = terminal_ground_point(poses[:, :3, 3], frame["world_from_camera_cv"],
                                                                 K, depth["depth_m"], depth["valid"], config,
                                                                 frame_index=t)
                        else:
                            expected_opos = fixed_world_point(binding["world_point"], frame["world_from_camera_cv"],
                                                              K, depth["depth_m"], depth["valid"], config)
                        if "pointing_grid" in episode:
                            expected_opos = attach_pixel_grid(expected_opos, w, h)
                        assert expected_opos == row["opos"]
                    for channel in ("apos", "opos"):
                        point = row[channel]
                        tid, valid = point["token_id"], point["loss_valid"]
                        assert tid == int(data[channel + "_token_id"][t])
                        assert valid == bool(data[channel + "_valid_mask"][t])
                        assert row[channel + "_tokens"] == pointing_token(channel, point)
                        if point["state"] == "point":
                            assert valid and 1 <= tid <= 1296
                            uv = np.asarray(point["pixel"])
                            assert 0 <= uv[0] < w and 0 <= uv[1] < h
                            assert tid == encode_point_pixel(*uv, w, h)
                            np.testing.assert_allclose(data[channel + "_pixel"][t], uv, atol=1e-4)
                            np.testing.assert_allclose(data[channel + "_world"][t], point["world_point"], atol=1e-4)
                            projected, z = project_world([point["world_point"]], frame["world_from_camera_cv"], K)
                            np.testing.assert_allclose(projected[0], uv, atol=1e-4)
                            col, r = np.floor(uv).astype(int)
                            assert depth["valid"][r, col]
                            observed = depth["depth_m"][r, col]
                            assert 0 < z[0] <= observed + config.depth_tolerance_m + config.depth_relative_tolerance * observed
                            if channel == "opos":
                                assert binding["bound"]
                                if not stop_location:
                                    with np.load(folder / "object_masks" / f"{t:06d}.npz") as mask:
                                        assert mask["visible_mask"][r, col]
                                totals["opos_visible"] += 1
                            else:
                                totals["apos_valid"] += 1
                        elif point["state"] == "not_visible":
                            assert channel == "opos" and tid == 0 and valid and binding["bound"]
                            if not stop_location:
                                with np.load(folder / "object_masks" / f"{t:06d}.npz") as mask:
                                    assert not mask["visible_mask"].any()
                            totals["opos_not_visible"] += 1
                            if point["reason"] == "before_opos_visibility_window":
                                totals["opos_temporally_hidden"] += 1
                        else:
                            assert point["state"] == "unknown" and tid == config.unknown_token_id and not valid
                        if point["state"] != "point":
                            assert point["pixel"] is None and point["world_point"] is None
                            assert np.isnan(data[channel + "_pixel"][t]).all()
                            assert np.isnan(data[channel + "_world"][t]).all()
        totals["episodes"] += 1
        totals["frames"] += len(rows)
    assert totals["episodes"] == len(manifest["episodes"])
    return {**totals, "checked": "source hashes, frame alignment, masks, temporal windows, "
                                "projections and visible instance pixels"}
