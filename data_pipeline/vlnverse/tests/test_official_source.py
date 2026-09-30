"""Source validation must reject corrupt ordering instead of manufacturing motion."""

import hashlib
import json

import numpy as np
import pytest

from data_pipeline.vlnverse.source.official import SourceEpisode, load_official_episode


def test_source_episode_keeps_yaw_wrap_and_source_time():
    episode = SourceEpisode(
        "sample", [[0, 0, 1], [0, 0, 1]], [3.1, -3.1], source_timestamps=[0, 1]
    )
    np.testing.assert_array_equal(episode.yaw_rad, [3.1, -3.1])
    np.testing.assert_array_equal(episode.source_timestamps, [0, 1])
    assert episode.provenance == {}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"episode_id": " "}, "episode_id"),
        ({"positions_xyz": np.empty((0, 3)), "yaw_rad": []}, "at least one"),
        ({"positions_xyz": [[0, 0]]}, "shape"),
        ({"positions_xyz": [[0, np.nan, 0]]}, "finite"),
        ({"yaw_rad": [np.inf]}, "finite"),
        ({"yaw_rad": [[0]]}, "shape"),
        ({"yaw_rad": [0, 1]}, "matching"),
        ({"source_timestamps": [0, 1]}, "matching"),
        ({"source_timestamps": [np.nan]}, "finite"),
    ],
)
def test_source_episode_rejects_bad_shapes_and_nonfinite_values(kwargs, message):
    arguments = {"episode_id": "sample", "positions_xyz": [[0, 0, 1]], "yaw_rad": [0]}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SourceEpisode(**arguments)


@pytest.fixture
def write_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "kujiale_0005" / "0_0" / "data" / "chunk-000" / "episode_000000.parquet"
    path.parent.mkdir(parents=True)

    def write(overrides=None, remove=()):
        data = {
            "observation.robot_position": ["[0, 0, 1.05]", "[0.25, 0, 1.05]", "[0.25, 0, 1.05]"],
            "observation.robot_yaw": [0.0, 0.0, np.pi / 12],
            "observation.robot_orientation": [
                "[1, 0, 0, 0]", "[1, 0, 0, 0]",
                json.dumps([np.cos(np.pi / 24), 0, 0, np.sin(np.pi / 24)]),
            ],
            "frame_index": [0, 1, 2],
            "observation.step": [0, 1, 2],
            "timestamp": [0.0, 1.0, 2.0],
            "episode_index": [0, 0, 0],
            "observation.action": [1, 3, 0],
        }
        data.update(overrides or {})
        for name in remove:
            del data[name]
        pq.write_table(pa.table(data), path)
        return path

    return write


def test_load_official_json_vectors_with_metadata_and_hashes(write_parquet):
    path = write_parquet()
    metadata = path.parents[2] / "meta" / "episodes.jsonl"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({
        "episode_index": 0, "tasks": ["Turn towards the window."], "finish_status": "success",
    }) + "\n", encoding="utf-8")
    episode = load_official_episode(path)
    assert episode.episode_id == "kujiale_0005/0_0"
    assert episode.instruction == "Turn towards the window."
    assert episode.provenance["finish_status"] == "success"
    assert episode.provenance["source_path"] == str(path.resolve())
    assert episode.provenance["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert episode.provenance["metadata_sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert episode.provenance["source_timestamp_semantics"] == "unverified_source_values_not_seconds"
    np.testing.assert_array_equal(episode.source_timestamps, [0, 1, 2])
    np.testing.assert_allclose(episode.positions_xyz[:, 0], [0, 0.25, 0.25])


def test_missing_metadata_does_not_imply_success(write_parquet):
    episode = load_official_episode(write_parquet(), episode_id="explicit")
    assert episode.episode_id == "explicit"
    assert episode.provenance["finish_status"] is None
    assert episode.instruction == ""


def test_official_metadata_uses_episode_id_and_instruction_text(write_parquet):
    path = write_parquet()
    metadata = path.parents[2] / "meta" / "episodes.jsonl"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({
        "episode_id": "0_0", "instruction_text": "Walk through the doorway.",
        "finish_status": "success", "fail_reason": None,
    }) + "\n", encoding="utf-8")
    episode = load_official_episode(path)
    assert episode.episode_id == "kujiale_0005/0_0"
    assert episode.instruction == "Walk through the doorway."
    assert episode.provenance["finish_status"] == "success"
    assert episode.provenance["source_metadata"]["episode_id"] == "0_0"


def test_rejects_single_unrelated_official_metadata_row(write_parquet, tmp_path):
    path = write_parquet()
    metadata = tmp_path / "episodes.jsonl"
    metadata.write_text('{"episode_id": "999_0", "finish_status": "success"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source episode identity"):
        load_official_episode(path, metadata_path=metadata)


def test_load_native_vectors_without_optional_quaternion_or_timestamp(write_parquet):
    path = write_parquet(
        {"observation.robot_position": [[0, 0, 1], [0.25, 0, 1], [0.25, 0, 1]]},
        remove=("frame_index", "observation.robot_orientation", "timestamp"),
    )
    episode = load_official_episode(path)
    assert episode.source_timestamps is None
    assert episode.provenance["source_order_columns"] == ["observation.step"]


@pytest.mark.parametrize("order", [[0, 2, 1], [0, 1, 1], [0, 1, 3], [0, 1, 1.5]])
def test_rejects_shuffled_duplicate_missing_or_fractional_frames(write_parquet, order):
    with pytest.raises(ValueError, match="original file order"):
        load_official_episode(write_parquet({"frame_index": order}))


@pytest.mark.parametrize(
    "remove",
    [("observation.robot_position",), ("observation.robot_yaw",), ("frame_index", "observation.step")],
)
def test_rejects_missing_pose_or_order_even_if_actions_exist(write_parquet, remove):
    with pytest.raises(ValueError, match="pose columns|verify pose order"):
        load_official_episode(write_parquet(remove=remove))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("observation.robot_position", "not JSON", "valid JSON"),
        ("observation.robot_position", "[1, 2]", "shape"),
        ("observation.robot_position", "[NaN, 0, 0]", "finite"),
        ("observation.robot_yaw", np.nan, "finite"),
        ("observation.robot_orientation", "[2, 0, 0, 0]", "unit"),
        ("observation.robot_orientation", "[0, 0, 0, 1]", "disagrees"),
        ("episode_index", 2, "filename"),
    ],
)
def test_rejects_bad_json_values_and_quaternions(write_parquet, column, value, message):
    with pytest.raises(ValueError, match=message):
        load_official_episode(write_parquet({column: [value] * 3}))


def test_quaternion_yaw_comparison_handles_wraparound(write_parquet):
    path = write_parquet({
        "observation.robot_yaw": [-np.pi, -np.pi, -np.pi],
        "observation.robot_orientation": ["[0, 0, 0, 1]"] * 3,
    })
    np.testing.assert_allclose(load_official_episode(path).yaw_rad, -np.pi)


def test_rejects_metadata_for_different_episode(write_parquet, tmp_path):
    path = write_parquet()
    metadata = tmp_path / "episodes.jsonl"
    metadata.write_text('{"episode_index": 9, "finish_status": "success"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one row"):
        load_official_episode(path, metadata_path=metadata)
