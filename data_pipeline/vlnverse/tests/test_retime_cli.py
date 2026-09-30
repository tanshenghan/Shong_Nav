"""Exercise official-format input, canonical map and self-contained output records."""

import json

import numpy as np
import pytest

from data_pipeline.vlnverse.cli.retime import main


@pytest.mark.parametrize(("last_x", "uniform_timestamps"), [(.5, True), (.75, False)])
def test_cli_emits_validated_motion_and_preserves_inputs(tmp_path, last_x, uniform_timestamps):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "source.parquet"
    table = pa.table({"observation.robot_position": ["[0,0,0]", "[0.25,0,0]", f"[{last_x},0,0]"],
        "observation.robot_yaw": [0., 0., 0.], "observation.step": [0, 1, 2], "timestamp": [0, 1, 2]})
    pq.write_table(table, path)
    before = path.read_bytes()
    grid = tmp_path / "grid.npz"
    np.savez(grid, occupied=np.zeros((40, 40), dtype=bool), resolution_m=.1, origin_xy_m=[-2., -2.])
    output = tmp_path / "retimed"
    args = ["--episode-parquet", str(path), "--occupancy-npz", str(grid),
            "--output-dir", str(output), "--episode-id", "synthetic_cli"]
    main(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["format"] == "vlnverse_smooth_motion_v2"
    assert manifest["sampling"] == {
        "mode": "per_source_step", "poses_per_source_step": 6, "sample_hz": None,
        "uniform_timestamps": uniform_timestamps, "shared_endpoints_stored_once": True,
    }
    assert manifest["quality"]["valid"] and not manifest["quality"]["simulator_replayed"]
    assert manifest["quality"]["sample_hz"] is None
    assert manifest["quality"]["sampling_mode"] == "per_source_step"
    assert manifest["quality"]["poses_per_source_step"] == 6
    assert "continuous_maxima" in manifest["quality"]
    assert manifest["time_basis"] == "synthetic_retimed_seconds"
    with np.load(output / "trajectory.npz") as motion:
        assert not np.allclose(np.diff(motion["timestamp_s"]), 1 / 30)
        assert (np.diff(motion["timestamp_s"]) > 0).all()
        np.testing.assert_allclose(motion["positions_xyz"][-1], [last_x, 0, 0])
        assert len(motion["timestamp_s"]) == 11
        np.testing.assert_allclose(motion["positions_xyz"][5], [.25, 0, 0], atol=1e-5)
        assert motion["velocity_xyz"][5, 0] > 0
    with np.load(output / "source_poses.npz") as source:
        np.testing.assert_array_equal(source["source_timestamps"], [0, 1, 2])
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        main(args)
    # An obstructed input cannot publish a successful manifest.
    np.savez(grid, occupied=np.ones((40, 40), dtype=bool), resolution_m=.1, origin_xy_m=[-2., -2.])
    args[args.index("--output-dir") + 1] = str(tmp_path / "blocked")
    with pytest.raises(ValueError, match="swept footprint"):
        main(args)
    assert not (tmp_path / "blocked").exists()


def test_cli_uniform_time_manifest_identifies_fixed_frequency(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({
        "observation.robot_position": [[0., 0., 0.], [.25, 0., 0.], [.5, 0., 0.]],
        "observation.robot_yaw": [0., 0., 0.], "frame_index": [0, 1, 2],
    }), path)
    grid = tmp_path / "grid.npz"
    np.savez(grid, occupied=np.zeros((40, 40), dtype=bool), resolution_m=.1, origin_xy_m=[-2., -2.])
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"sample_mode": "uniform_time", "sample_hz": 30.}))
    output = tmp_path / "retimed"
    main(["--episode-parquet", str(path), "--occupancy-npz", str(grid),
          "--output-dir", str(output), "--config", str(config)])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["sampling"] == {
        "mode": "uniform_time", "poses_per_source_step": None, "sample_hz": 30.,
        "uniform_timestamps": True, "shared_endpoints_stored_once": True,
    }
    assert manifest["quality"]["sample_hz"] == 30.
    with np.load(output / "trajectory.npz") as motion:
        assert len(motion["timestamp_s"]) == 91
        np.testing.assert_allclose(np.diff(motion["timestamp_s"]), 1 / 30)
