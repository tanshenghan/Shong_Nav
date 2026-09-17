"""RxR source coverage, split/language identity and unchanged LeRobot GT schema."""

import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from data_pipeline.build_r2r_train import expand_instructions, plan_routes
from data_pipeline.collect_r2r import configure_navigation_mesh, simulator_config, write_json
from data_pipeline.export_lerobot import export_dataset, read_source
from data_pipeline.tests.test_full_train import inputs
from data_pipeline.tests.test_lerobot import synthetic_source
from data_pipeline.vlnce_source import load_source, rxr_replay_actions, source_paths


def test_rxr_navigation_preserves_camera_and_rebuilds_only_in_memory(tmp_path):
    args = SimpleNamespace(config=Path(__file__).resolve().parents[2] / "habitat_server/configs/vlnce_r2r.yaml",
                           seed=1, gpu_id=0, width=470, height=280, dataset="r2r")
    r2r = simulator_config(args)
    args.dataset = "rxr"
    rxr = simulator_config(args)
    assert r2r["agents"]["main_agent"]["sim_sensors"] == rxr["agents"]["main_agent"]["sim_sensors"]
    assert r2r["turn_angle"] == rxr["turn_angle"] == 15
    assert r2r["habitat_sim_v0"]["allow_sliding"] is True
    assert rxr["habitat_sim_v0"]["allow_sliding"] is False
    assert "navmesh_policy" not in r2r
    sim, settings = Mock(), Mock()
    hsim = SimpleNamespace(NavMeshSettings=Mock(return_value=settings))
    scene = tmp_path / "scene.glb"
    navmesh = scene.with_suffix(".navmesh")
    navmesh.write_bytes(b"original mesh")
    configure_navigation_mesh(sim, hsim, r2r, scene)
    sim.pathfinder.load_nav_mesh.assert_called_once_with(str(navmesh))
    sim.recompute_navmesh.assert_not_called()
    sim.reset_mock()
    configure_navigation_mesh(sim, hsim, rxr, scene)
    settings.set_defaults.assert_called_once()
    assert settings.agent_height == .88 and settings.agent_radius == .18
    sim.recompute_navmesh.assert_called_once_with(sim.pathfinder, settings)
    sim.pathfinder.load_nav_mesh.assert_not_called()
    assert navmesh.read_bytes() == b"original mesh"
    sim.recompute_navmesh.return_value = False
    with pytest.raises(RuntimeError, match="recomputation failed"):
        configure_navigation_mesh(sim, hsim, rxr, scene)


def source_files(root, role="guide", split="train"):
    episodes, gt = inputs()
    for episode, language, text in zip(episodes, ["en-US", "hi-IN", "te-IN", "en-IN"],
                                       ["Stop here.", "यहाँ रुकें।", "ఇక్కడ ఆగండి.", "Stop here."], strict=True):
        episode["info"] = {"role": role}
        episode["instruction"].update(language=language, instruction_id=episode["episode_id"],
                                      instruction_text=text, annotator_id="annotator")
        if role == "follower":
            episode.pop("trajectory_id")
            episode["info"]["demonstration_id"] = episode["episode_id"]
    folder = root / split
    folder.mkdir(parents=True)
    for name, value in [(f"{split}_{role}.json.gz", {"episodes": episodes}), (f"{split}_{role}_gt.json.gz", gt)]:
        with gzip.open(folder / name, "wt", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
    return episodes, gt


def test_rxr_all_languages_keep_duplicate_text_and_original_identity(tmp_path):
    original, _ = source_files(tmp_path)
    episodes, gt, selection = load_source(tmp_path, "rxr")
    assert selection["selected_episode_count"] == 4
    assert set(selection["language_counts"]) == {"en-US", "en-IN", "hi-IN", "te-IN"}
    assert [e["instruction"]["instruction_text"] for e in episodes] == [e["instruction"]["instruction_text"] for e in original]
    plan = plan_routes(episodes, gt)
    assert plan["physical_routes"] == 1 and plan["source_episodes"] == 4
    planned = plan["shards"][0]["routes"][0]["episodes"]
    assert [e["source_metadata"]["language"] for e in planned] == ["en-US", "hi-IN", "te-IN", "en-IN"]


def test_rxr_language_filter_is_exact_and_retains_source_counts(tmp_path):
    source_files(tmp_path)
    episodes, gt, selection = load_source(tmp_path, "rxr", languages=["en-IN", "en-US"])
    assert [str(e["episode_id"]) for e in episodes] == ["0", "3"]
    assert set(gt) == {"0", "3"}
    assert selection["source_episode_count"] == 4 and selection["selected_episode_count"] == 2
    with pytest.raises(ValueError, match="cannot be combined"):
        load_source(tmp_path, "rxr", languages=["all", "en-US"])


def test_rxr_turns_preserve_source_heading_at_lightnav_sampling_rate(tmp_path):
    original = [2, 1, 3, 3, 0]
    assert rxr_replay_actions(original) == [2, 2, 1, 3, 3, 3, 3, 0]
    assert original == [2, 1, 3, 3, 0]
    for bad in ([], [1], [0, 1, 0], [4, 0]):
        with pytest.raises(ValueError, match="STOP"):
            rxr_replay_actions(bad)
    _, gt = source_files(tmp_path)
    gt["0"]["actions"] = original
    with gzip.open(tmp_path / "train/train_guide_gt.json.gz", "wt") as stream:
        json.dump(gt, stream)
    episodes, adapted_gt, selection = load_source(tmp_path, "rxr")
    assert adapted_gt["0"]["locations"] == gt["0"]["locations"]
    assert episodes[0]["source_metadata"]["source_action_count"] == 5
    assert episodes[0]["source_metadata"]["replay_frame_count"] == 8
    assert selection["replay_frames"] == selection["source_frames"] + 3
    assert plan_routes(episodes, adapted_gt)["frames"] == selection["replay_frames"]


def test_follower_and_validation_metadata_stay_separate(tmp_path):
    source_files(tmp_path, role="follower", split="val_seen")
    episodes, _, metadata = load_source(tmp_path, "rxr", split="val_seen", role="follower")
    assert metadata["split"] == "val_seen" and metadata["role"] == "follower"
    assert episodes[0]["source_metadata"]["demonstration_id"] == "0"
    with pytest.raises(FileNotFoundError):
        load_source(tmp_path, "rxr", split="val_seen", role="guide")
    with pytest.raises(ValueError, match="test_challenge"):
        source_paths(tmp_path, "rxr", "test_challenge")


@pytest.mark.parametrize("problem", ["missing_gt", "duplicate_id", "unsafe_id", "wrong_role"])
def test_bad_sources_fail_before_rendering(tmp_path, problem):
    episodes, gt = source_files(tmp_path)
    if problem == "missing_gt":
        del gt["0"]
    elif problem == "duplicate_id":
        episodes.append(copy.deepcopy(episodes[0]))
    elif problem == "unsafe_id":
        episodes[0]["episode_id"] = "../outside"
    else:
        episodes[0]["info"]["role"] = "follower"
    for filename, data in [("train_guide.json.gz", {"episodes": episodes}), ("train_guide_gt.json.gz", gt)]:
        with gzip.open(tmp_path / "train" / filename, "wt") as stream:
            json.dump(data, stream)
    with pytest.raises(ValueError):
        load_source(tmp_path, "rxr")


def test_r2r_source_read_still_accepts_original_schema(tmp_path):
    episodes, gt = inputs()
    folder = tmp_path / "train"
    folder.mkdir()
    for filename, data in [("train.json.gz", {"episodes": episodes}), ("train_gt.json.gz", gt)]:
        with gzip.open(folder / filename, "wt") as stream:
            json.dump(data, stream)
    loaded, loaded_gt, meta = load_source(tmp_path)
    assert loaded == episodes and loaded_gt == gt and meta["dataset"] == "r2r"


def test_instruction_clones_keep_their_own_language_and_annotation_id(tmp_path):
    raw = synthetic_source(tmp_path)
    folder = raw / "episodes/b_moving"
    meta = json.loads((folder / "episode.json").read_text())
    meta["source_metadata"] = {"language": "en-US", "instruction_id": "original"}
    write_json(folder / "episode.json", meta)
    route = dict(representative_id="b_moving", frames=12, physical_sha256="same",
        episodes=[dict(episode_id="b_moving"), dict(episode_id="c_hindi", instruction={"instruction_text": "यहाँ रुकें।"},
            source_metadata={"language": "hi-IN", "instruction_id": "hindi"})])
    expand_instructions(raw, dict(routes=[route], episodes=3))
    clone = read_source(raw / "episodes/c_hindi")
    assert clone["metadata"]["source_metadata"] == {"language": "hi-IN", "instruction_id": "hindi"}
    assert clone["metadata"]["instruction"] == "यहाँ रुकें।"
    assert clone["arrays"]["fm_clean_se2_diff"].tobytes() == read_source(folder)["arrays"]["fm_clean_se2_diff"].tobytes()


def test_official_export_preserves_rxr_metadata_without_changing_features(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    from data_pipeline.export_lerobot import build_features
    from data_pipeline.validate_lerobot import validate_lerobot

    raw = synthetic_source(tmp_path)
    before = build_features(read_source(raw / "episodes/b_moving"))
    for path in sorted((raw / "episodes").glob("*/episode.json")):
        meta = json.loads(path.read_text())
        meta.update(instruction="ఇక్కడ ఆగండి.", source_metadata=dict(dataset="rxr", split="train", role="guide", language="te-IN"))
        write_json(path, meta)
    exported = export_dataset(raw, tmp_path / "lerobot", repo_id="beta-nav/rxr-test")
    assert exported["features"] == before
    assert all(e["source_metadata"]["language"] == "te-IN" for e in exported["episodes"])
    assert validate_lerobot(tmp_path / "lerobot", raw)["frames"] == 13


def test_rxr_merge_validates_language_identity_and_rejects_other_splits(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    from data_pipeline.merge_lerobot import merge_datasets, verify_coverage

    raw = synthetic_source(tmp_path)
    manifest = json.loads((raw / "manifest.json").read_text())
    manifest["source_selection"] = dict(dataset="rxr", split="train", role="guide", languages=["all"])
    write_json(raw / "manifest.json", manifest)
    routes = []
    for episode_id, frames, language in [("a_stationary", 1, "hi-IN"), ("b_moving", 12, "te-IN")]:
        path = raw / "episodes" / episode_id / "episode.json"
        meta = json.loads(path.read_text())
        source_meta = dict(dataset="rxr", split="train", role="guide", language=language)
        meta["source_metadata"] = source_meta
        write_json(path, meta)
        routes.append(dict(frames=frames, scene_id="synthetic", episodes=[dict(episode_id=episode_id,
            instruction=dict(instruction_text=meta["instruction"]), source_metadata=source_meta)]))
    shards = [tmp_path / "shard0", tmp_path / "shard1"]
    for shard, episode_id in zip(shards, ["a_stationary", "b_moving"], strict=True):
        export_dataset(raw, shard, episode_ids=[episode_id])
    output = tmp_path / "merged"
    merge_datasets(shards, output, "beta-nav/rxr-test", tmp_path / "journal")
    plan = dict(source_episodes=2, frames=13, shards=[dict(routes=routes)])
    assert verify_coverage(output, plan)["all_source_metadata_exact"]
    broken = copy.deepcopy(plan)
    broken["shards"][0]["routes"][1]["episodes"][0]["source_metadata"]["language"] = "en-US"
    with pytest.raises(ValueError, match="language"):
        verify_coverage(output, broken)
    metadata_path = shards[1] / "meta/beta_nav.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_selection"]["split"] = "val_seen"
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="source splits"):
        merge_datasets(shards, tmp_path / "bad_merge", "beta-nav/rxr-test", tmp_path / "bad_journal")
