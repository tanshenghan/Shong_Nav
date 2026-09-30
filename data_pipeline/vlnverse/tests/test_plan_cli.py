"""End-to-end provenance and publication contract for planned reference motion."""

import hashlib
import json

import numpy as np
import pytest

from data_pipeline.vlnverse.cli.plan import main


def test_planned_cli_preserves_source_and_publishes_distinct_contract(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "source.parquet"
    positions = [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]
    pq.write_table(pa.table({"observation.robot_position": positions,
                            "observation.robot_yaw": [0., 0., 0.], "frame_index": [0, 1, 2]}), path)
    before = path.read_bytes()
    grid = tmp_path / "grid.npz"
    np.savez(grid, occupied=np.zeros((80, 80), bool), resolution_m=.1, origin_xy_m=[-3., -3.])
    config = tmp_path / "config.json"
    settings = {"episode_id": "synthetic_planned_cli", "waypoint_indices": [0, 2],
                "source_sha256": hashlib.sha256(before).hexdigest(), "motion": {"sample_hz": 15.}}
    config.write_text(json.dumps(settings))
    output = tmp_path / "planned"
    args = ["--episode-parquet", str(path), "--occupancy-npz", str(grid),
            "--config", str(config), "--output-dir", str(output)]
    main(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["format"] == "vlnverse_planned_motion_v1"
    assert manifest["route_alignment"]["official_step_correspondence"] is False
    assert manifest["sampling"] == {"mode": "uniform_time", "sample_hz": 15., "uniform_timestamps": True}
    for name in ("trajectory", "source_poses", "planned_curve"):
        assert manifest[f"{name}_sha256"] == hashlib.sha256((output / f"{name}.npz").read_bytes()).hexdigest()
    with np.load(output / "trajectory.npz") as values:
        np.testing.assert_allclose(values["positions_xyz"][[0, -1]], np.array(positions)[[0, -1]], atol=1e-8)
        np.testing.assert_allclose(np.diff(values["timestamp_s"]), 1 / 15, atol=1e-10)
        assert "source_progress" not in values.files
    with np.load(output / "source_poses.npz") as values:
        np.testing.assert_array_equal(values["positions_xyz"], positions)
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        main(args)
    settings["source_sha256"] = "0" * 64
    config.write_text(json.dumps(settings))
    args[-1] = str(tmp_path / "mismatched")
    with pytest.raises(ValueError, match="different source episode"):
        main(args)
    assert not (tmp_path / "mismatched").exists()
