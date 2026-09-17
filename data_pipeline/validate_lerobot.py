#!/usr/bin/env python3
"""Read a LeRobotDataset and verify embedded observations, all GT, geometry and hashes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data_pipeline.collect_r2r import sha256, write_json  # noqa: E402
from data_pipeline.export_lerobot import (  # noqa: E402
    ARRAY_FEATURES,
    RGB_KEY,
    array_digest,
    read_source,
)
from data_pipeline.geometry import future_targets  # noqa: E402
from data_pipeline.pointing import (  # noqa: E402
    PointingConfig,
    attach_pixel_grid,
    ground_point,
    pointing_token,
    terminal_ground_point,
)
from lightnav.traj_vocab import load_rvq_bundle  # noqa: E402


def validate_lerobot(root, source=None):
    import datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    metadata = json.loads((root / "meta/beta_nav.json").read_text())
    dataset = LeRobotDataset(metadata["repo_id"], root=root, video_backend="pyav")
    assert dataset.meta.info["codebase_version"] == "v3.0" and dataset.fps == 1
    assert len(dataset) == sum(e["frames"] for e in metadata["episodes"])
    assert dataset.num_episodes == len(metadata["episodes"])
    for key, feature in metadata["features"].items():
        saved = dataset.features[key]
        assert saved["dtype"] == feature["dtype"] and list(saved["shape"]) == list(feature["shape"])
    for name, digest in metadata["rvq_assets"].items():
        assert sha256(root / "assets/action_tokenizer" / name) == digest
    bundle = load_rvq_bundle(root / "assets/action_tokenizer", horizon=10)
    raw = dataset.hf_dataset.with_format(None)
    embedded = raw.cast_column(RGB_KEY, datasets.Image(decode=False))
    offset, details = 0, []
    for ep in metadata["episodes"]:
        n = ep["frames"]
        rows = [raw[offset + t] for t in range(n)]
        arrays = {}
        for source_key, key in ARRAY_FEATURES.items():
            feature = metadata["features"][key]
            values = np.asarray([r[key] for r in rows], dtype=feature["dtype"])
            assert array_digest(values) == ep["array_hashes"][key], key
            arrays[source_key] = values
        if source is not None:
            original = read_source(Path(source) / "episodes" / ep["episode_id"])
            for key in ARRAY_FEATURES:
                np.testing.assert_array_equal(arrays[key], original["arrays"][key])
        future = future_targets(arrays["world_from_body"], 30)
        for key, expected in [("fm_waypoints_ego", "waypoints"), ("fm_clean_se2_diff", "se2_diff"),
                              ("fm_valid_mask", "valid"), ("future_frame_ids", "frame_ids"),
                              ("future_vertical_offset_m", "vertical_offset_m")]:
            np.testing.assert_allclose(arrays[key], future[expected], atol=1e-6)
        config = PointingConfig.from_metadata(ep["pointing_config"])
        visible, apos_visible = [], 0
        for t, row in enumerate(rows):
            assert row["frame_index"] == t and row["timestamp"] == float(t)
            assert row["index"] == offset + t and row["episode_index"] == ep["episode_index"]
            assert row["source.episode_id"] == ep["episode_id"] and row["source.scene_id"] == ep["scene_id"]
            # Exercise the actual official __getitem__, including RGB tensor conversion.
            loaded = dataset[offset + t]
            assert loaded["task"] == ep["instruction"]
            assert loaded[RGB_KEY].shape == (3, ep["camera"]["height"], ep["camera"]["width"])
            assert embedded[offset + t][RGB_KEY]["bytes"]
            for key, dtype in [(RGB_KEY, "uint8"), ("observation.depth_m", "float32"), ("observation.depth_valid", "bool")]:
                values = np.asarray(row[key], dtype=dtype)
                assert array_digest(values) == ep["observation_hashes"][t][key]
                decoded = loaded[key].numpy()
                if key == RGB_KEY:
                    decoded = np.rint(decoded.transpose(1, 2, 0) * 255).astype(np.uint8)
                np.testing.assert_array_equal(decoded, values)
            for key in ARRAY_FEATURES.values():
                np.testing.assert_allclose(loaded[key].numpy(), row[key], atol=2e-5, equal_nan=True)
            depth = np.asarray(row["observation.depth_m"], dtype=np.float32)
            valid = np.asarray(row["observation.depth_valid"], dtype=bool)
            pose = arrays["world_from_camera_cv"][t]
            positions = arrays["world_from_body"][:, :3, 3]
            K = np.asarray(ep["camera"]["K"])
            points = {"apos": ground_point(positions, t, pose, K, depth, valid, config),
                      "opos": terminal_ground_point(positions, pose, K, depth, valid, config, frame_index=t)}
            for channel, point in points.items():
                point = attach_pixel_grid(point, depth.shape[1], depth.shape[0])
                prefix = "action." + channel + "."
                assert row[prefix + "token"] == pointing_token(channel, point)
                assert row[prefix + "token_id"] == point["token_id"]
                assert row[prefix + "valid_mask"] == point["loss_valid"]
                assert row[prefix + "state"] == point["state"] and row[prefix + "reason"] == point["reason"]
                for key, source_key in (("pixel", "pixel"), ("world", "world_point"),
                                        ("target_world", "target_world_point")):
                    if point[source_key] is None:
                        assert np.isnan(row[prefix + key]).all()
                    else:
                        np.testing.assert_allclose(row[prefix + key], point[source_key], atol=2e-5)
                if point["state"] == "point":
                    if channel == "opos":
                        visible.append(t)
                    else:
                        apos_visible += 1
            codes = row["action.rvq.codes"]
            if row["action.rvq.valid_mask"]:
                assert row["action.rvq.tokens"] == "".join(f"<act_l{i}_{c}>" for i, c in enumerate(codes))
                if row["action.explicit_stop"]:
                    assert t == n - 1 and row["action.stop_loss_valid"] and bundle.is_stop(codes)
                else:
                    assert t + 10 < n and not bundle.is_stop(codes)
                    np.testing.assert_allclose(row["action.rvq.reconstruction"], bundle.decode_waypoints(codes), atol=1e-6)
            else:
                assert row["action.rvq.tokens"] == "" and np.all(np.asarray(codes) == -1)
        assert not arrays["fm_valid_mask"][-1].any() and rows[-1]["action.explicit_stop"]
        assert all(t >= ep["opos_visibility_start_frame"] for t in visible)
        details.append(dict(episode_id=ep["episode_id"], frames=n, opos_visibility_start_frame=ep["opos_visibility_start_frame"],
                            opos_visible_frame_ids=visible, apos_visible=apos_visible,
                            stop_rvq_codes=rows[-1]["action.rvq.codes"]))
        offset += n
    stats = json.loads((root / "meta/stats.json").read_text())
    assert all(np.isfinite(np.asarray(v)).all() for fields in stats.values() for v in fields.values())
    # A real training-style batch must load without needing any capture directory.
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)), num_workers=0)))
    assert batch["action.fm.waypoints_ego"].shape[1:] == (30, 3)
    return dict(complete=True, format="LeRobotDataset v3.0", frames=len(dataset), episodes=details,
                official_loader_all_frames=True, dataloader_batch=True, rgb_depth_hashes_exact=True,
                all_30_source_arrays_exact=True, geometry_recomputed=True, finite_masked_stats=True,
                source_comparison=source is not None, self_contained=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--source", type=Path, help="Optional original capture/overlay for additional exact comparison")
    parser.add_argument("--report", type=Path, help="NEW report file")
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error("report already exists")
    result = validate_lerobot(args.dataset, args.source)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
