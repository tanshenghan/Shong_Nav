"""Full-split instruction coverage, safe route reuse and real LeRobot shard merging."""

import copy
import json

import numpy as np
import pytest

from data_pipeline.R2R.build_r2r_train import completed_shards, expand_instructions, plan_routes
from data_pipeline.R2R.collect_r2r import write_json
from data_pipeline.R2R.export_lerobot import export_dataset, read_source
from data_pipeline.R2R.tests.test_lerobot import synthetic_source


@pytest.mark.parametrize("direction", [-1, 1])
def test_fm_composition_accepts_equivalent_headings_across_pi(direction):
    from data_pipeline.R2R.geometry import future_targets
    from lightnav.traj_vocab import compose_to_abs, wrap_to_pi

    yaw = direction * np.arange(36) * np.deg2rad(15.)
    poses = np.repeat(np.eye(4)[None], len(yaw), axis=0)
    for t, angle in enumerate(yaw):
        c, s = np.cos(angle), np.sin(angle)
        poses[t, :3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
    result = future_targets(poses, 30)
    composed = compose_to_abs(result["se2_diff"])
    np.testing.assert_allclose(wrap_to_pi(composed[..., 2] - result["waypoints"][..., 2]), 0., atol=2e-5)
    assert np.max(np.abs(composed[..., 2] - result["waypoints"][..., 2])) > 6.
    assert result["valid"][0].all() and not result["valid"][-1].any()


def inputs():
    episodes = [dict(episode_id=str(i), trajectory_id=7, scene_id="scene", start_position=[0., 0., 0.],
        start_rotation=[0., 0., 0., 1.], goals=[dict(position=[0., 0., -1.])],
        instruction=dict(instruction_text=f"Instruction {i}")) for i in range(4)]
    gt = {str(i): dict(actions=[1, 0], locations=[[0., 0., 0.], [0., 0., -1.]]) for i in range(4)}
    return episodes, gt


def test_full_plan_keeps_every_instruction_and_reuses_identical_routes():
    episodes, gt = inputs()
    plan = plan_routes(episodes, gt, 1)
    assert (plan["source_episodes"], plan["physical_routes"], plan["frames"], plan["rendered_frames"]) == (4, 1, 8, 2)
    assert plan == plan_routes(list(reversed(episodes)), gt, 1)
    assert len(plan["shards"][0]["routes"][0]["episodes"]) == 4


def test_same_route_id_with_different_physics_is_rendered_separately():
    episodes, gt = inputs()
    gt["3"] = dict(actions=[2, 1, 0], locations=[[0., 0., 0.], [0., 0., -1.]])
    plan = plan_routes(episodes, gt, 1)
    assert plan["physical_routes"] == 2 and plan["frames"] == 9 and len(plan["shards"]) == 2
    with pytest.raises(ValueError, match="duplicate"):
        plan_routes(episodes + [episodes[0]], gt)
    del gt["3"]
    with pytest.raises(KeyError):
        plan_routes(episodes, gt)


def test_resume_rejects_a_shard_from_a_different_plan(tmp_path):
    marker = tmp_path / "shards/00000/meta/build_shard.json"
    marker.parent.mkdir(parents=True)
    write_json(marker, dict(complete=True, plan_sha256="old"))
    with pytest.raises(ValueError, match="different generation plan"):
        completed_shards(tmp_path, dict(plan_sha256="new", shards=[dict(id=0)]))


def test_resume_accepts_only_exact_empty_depth_rejection_removal(tmp_path, monkeypatch):
    import data_pipeline.R2R.build_r2r_train as builder
    from data_pipeline.R2R.collect_r2r import sha256

    repo, work = tmp_path / "repo", tmp_path / "work"
    collector = repo / "data_pipeline/R2R/collect_r2r.py"
    frozen = work / "shards/00000/meta/provenance/collect_r2r.py"
    collector.parent.mkdir(parents=True)
    frozen.parent.mkdir(parents=True)
    old_text = "def capture():\n" + builder.EMPTY_DEPTH_GUARD + "    return depth_valid\n"
    frozen.write_text(old_text)
    collector.write_text(old_text.replace(builder.EMPTY_DEPTH_GUARD, ""))
    monkeypatch.setattr(builder, "REPO", repo)
    original = dict(plan_sha256="old", frames=10, definition_sha256={str(collector): sha256(frozen)})
    current = dict(plan_sha256="new", frames=10, definition_sha256={str(collector): sha256(collector)})
    assert builder.compatible_resume_plan(original, current, work) == original
    assert frozen.read_text() == old_text
    assert (work / "definition_updates/empty_depth.json").is_file()
    with pytest.raises(ValueError, match="resume plan differs"):
        builder.compatible_resume_plan(original, dict(current, frames=9), work)
    collector.write_text(collector.read_text().replace("return depth_valid", "return True"))
    current["definition_sha256"][str(collector)] = sha256(collector)
    with pytest.raises(ValueError, match="only the exact"):
        builder.compatible_resume_plan(original, current, work)


def relocation_plans(tmp_path, monkeypatch, *, empty_depth_guard=False):
    import data_pipeline.R2R.build_r2r_train as builder
    from data_pipeline.R2R.collect_r2r import sha256

    repo, work = tmp_path / "repo", tmp_path / "work"
    old_root, new_root = repo / "data_pipeline", repo / "data_pipeline/R2R"
    frozen = work / "shards/00000/meta/provenance/collect_r2r.py"
    frozen.parent.mkdir(parents=True)
    (new_root / "configs").mkdir(parents=True)
    original_text = ('REPO = Path(__file__).resolve().parents[1]\n'
                     'from data_pipeline.geometry import future_targets\n'
                     'CONFIG = REPO / "data_pipeline/configs/pointing.json"\n'
                     'def capture():\n')
    frozen.write_text(original_text + (builder.EMPTY_DEPTH_GUARD if empty_depth_guard else "") + '    return depth_valid\n')
    collector = new_root / "collect_r2r.py"
    collector.write_text('REPO = Path(__file__).resolve().parents[2]\n'
                         'from data_pipeline.R2R.geometry import future_targets\n'
                         'CONFIG = REPO / "data_pipeline/R2R/configs/pointing.json"\n'
                         'def capture():\n    return depth_valid\n')
    config = new_root / "configs/pointing.json"
    config.write_text('{"opos_visibility_window": "last_half"}')
    geometry = new_root / "geometry.py"
    geometry.write_text('def future_targets():\n    return []\n')
    original = dict(plan_sha256="old", frames=10, pointing_config=str(old_root / "configs/pointing.json"),
        source_sha256={"source": "unchanged"}, rvq_sha256={"rvq": "unchanged"},
        definition_sha256={str(old_root / "collect_r2r.py"): sha256(frozen),
                           str(old_root / "configs/pointing.json"): sha256(config),
                           str(old_root / "geometry.py"): sha256(geometry)})
    current = dict(original, plan_sha256="new", pointing_config=str(config),
        definition_sha256={str(p): sha256(p) for p in (collector, config, geometry)})
    write_json(work / "plan.json", original)
    monkeypatch.setattr(builder, "REPO", repo)
    return builder, original, current, work, collector, frozen


@pytest.mark.parametrize("empty_depth_guard", [False, True])
def test_resume_relocation_preserves_plan_and_provenance(tmp_path, monkeypatch, empty_depth_guard):
    builder, original, current, work, collector, frozen = relocation_plans(
        tmp_path, monkeypatch, empty_depth_guard=empty_depth_guard)
    plan_bytes, source_bytes = (work / "plan.json").read_bytes(), frozen.read_bytes()
    unchanged = copy.deepcopy(original)
    assert builder.compatible_resume_plan(original, current, work) is original
    assert original == unchanged and (work / "plan.json").read_bytes() == plan_bytes
    assert frozen.read_bytes() == source_bytes
    audit_path = work / "definition_updates/r2r_directory_relocation.json"
    audit = json.loads(audit_path.read_text())
    assert audit["current_definition_sha256"] == current["definition_sha256"]
    assert audit["empty_depth_guard_removed"] is empty_depth_guard
    assert (audit_path.parent / "r2r_directory_relocation_collect_r2r.py").read_bytes() == collector.read_bytes()
    audit_bytes = audit_path.read_bytes()
    assert builder.compatible_resume_plan(original, current, work) is original
    assert audit_path.read_bytes() == audit_bytes


@pytest.mark.parametrize("change", ["collector", "geometry", "config", "source", "rvq", "unknown_path"])
def test_resume_relocation_rejects_changes_beyond_directory_move(tmp_path, monkeypatch, change):
    builder, original, current, work, collector, frozen = relocation_plans(tmp_path, monkeypatch)
    if change in {"collector", "geometry", "config"}:
        path = {"collector": collector, "geometry": collector.parent / "geometry.py",
                "config": collector.parent / "configs/pointing.json"}[change]
        path.write_text(path.read_text().replace("depth_valid", "True") if change == "collector" else "changed definition\n")
        current["definition_sha256"][str(path)] = builder.sha256(path)
    elif change == "unknown_path":
        current["definition_sha256"][str(collector.parent / "unrecognized.py")] = "unexpected"
    else:
        current[f"{change}_sha256"] = {change: "changed"}
    with pytest.raises(ValueError):
        builder.compatible_resume_plan(original, current, work)
    assert not (work / "definition_updates").exists()
    assert json.loads((work / "plan.json").read_text()) == original
    assert builder.sha256(frozen) == original["definition_sha256"][str(builder.REPO / "data_pipeline/collect_r2r.py")]


@pytest.mark.parametrize("historical_default", [False, True])
def test_worker_resolves_only_moved_default_config_without_changing_plan(tmp_path, monkeypatch, historical_default):
    pytest.importorskip("pyarrow")
    import data_pipeline.R2R.build_r2r_train as builder

    repo = tmp_path / "repo"
    monkeypatch.setattr(builder, "REPO", repo)
    config = str(repo / "data_pipeline/configs/pointing.json") if historical_default else str(tmp_path / "custom.json")
    expected = str(repo / "data_pipeline/R2R/configs/pointing.json") if historical_default else config
    plan = dict(shards=[dict(routes=[dict(representative_id="test")])], data_root="source", scenes_dir="scenes",
                rvq_bundle="rvq", pointing_config=config, split="train")
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    plan_bytes = plan_path.read_bytes()

    def inspect_capture(command, **kwargs):
        assert command[1] == str(repo / "data_pipeline/R2R/run.sh")
        assert command[command.index("--pointing-config") + 1] == expected
        raise RuntimeError("capture intercepted")

    monkeypatch.setattr(builder.subprocess, "run", inspect_capture)
    with pytest.raises(RuntimeError, match="capture intercepted"):
        builder.worker(plan_path, 0, 0)
    assert plan_path.read_bytes() == plan_bytes


def test_failed_batch_is_not_published_as_a_complete_dataset(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import data_pipeline.R2R.build_r2r_train as builder

    output = tmp_path / "dataset"
    plan = dict(plan_sha256="test", source_episodes=3, physical_routes=1, scenes=1,
                frames=30, rendered_frames=10, shards=[dict(id=0)])
    monkeypatch.setattr(builder, "create_plan", lambda args: plan)

    def fail(*args):
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(builder, "run_shard", fail)
    args = SimpleNamespace(output_dir=output, dry_run=False, workers=1, gpus=[1], retries=0, max_new_shards=None)
    with pytest.raises(RuntimeError, match="1 shards failed"):
        builder.run(args)
    status = json.loads((tmp_path / "dataset.work/status.json").read_text())
    assert status["state"] == "failed" and status["completed_shards"] == 0
    assert not output.exists()


def test_empty_depth_frame_survives_official_export_with_unknown_apos(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    from PIL import Image

    from data_pipeline.R2R.collect_r2r import export_targets
    from data_pipeline.R2R.pointing import PointingConfig
    from data_pipeline.R2R.validate_lerobot import validate_lerobot
    from lightnav.traj_vocab import load_rvq_bundle

    source = synthetic_source(tmp_path)
    folder = source / "episodes/b_moving"
    metadata = json.loads((folder / "episode.json").read_text())
    frames = [json.loads(s) for s in (folder / "frames.jsonl").read_text().splitlines()]
    Image.fromarray(np.zeros((16, 20, 3), np.uint8)).save(folder / "0.png")
    np.savez(folder / "0.npz", depth_m=np.zeros((16, 20), np.float32), valid=np.zeros((16, 20), bool))
    metadata.update(export_targets(folder, frames, load_rvq_bundle(tmp_path / "bundle", 10), True,
                                   metadata["camera"], PointingConfig()))
    write_json(folder / "episode.json", metadata)
    episode = read_source(folder)
    assert episode["rows"][0]["apos"]["state"] == "unknown"
    assert episode["arrays"]["apos_token_id"][0] == 0 and not episode["arrays"]["apos_valid_mask"][0]
    output = tmp_path / "export"
    export_dataset(source, output)
    assert validate_lerobot(output)["frames"] == 13
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["observation.depth_m"]["min"] == [9.]


@pytest.mark.parametrize("depth_value", [0., 10.])
def test_capture_preserves_finite_empty_depth_with_false_mask(tmp_path, monkeypatch, depth_value):
    from types import SimpleNamespace

    import data_pipeline.R2R.collect_r2r as collector

    sensor = np.eye(4)
    sensor[1, 3] = .88
    state = SimpleNamespace(matrix=np.eye(4), sensor_states={
        "rgb": SimpleNamespace(matrix=sensor), "depth": SimpleNamespace(matrix=sensor)})
    sim = SimpleNamespace(get_agent_state=lambda: state, previous_step_collided=False)
    monkeypatch.setattr(collector, "pose_matrix", lambda state: state.matrix.copy())
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    depth = np.full((16, 20), depth_value, np.float32)
    camera = dict(height=16, width=20, min_depth_m=0., max_depth_m=10., position_body_native=[0., .88, 0.])
    frame = collector.capture(sim, dict(rgb=np.zeros((16, 20, 3), np.uint8), depth=depth), tmp_path, 0, camera)
    assert frame["valid_depth_fraction"] == 0.
    with np.load(tmp_path / frame["depth"]) as saved:
        np.testing.assert_array_equal(saved["depth_m"], depth)
        assert not saved["valid"].any()


@pytest.mark.parametrize("depth_value", [float("nan"), -1.])
def test_capture_still_rejects_nonfinite_or_negative_depth(tmp_path, depth_value):
    from data_pipeline.R2R.collect_r2r import capture

    observation = dict(rgb=np.zeros((16, 20, 3), np.uint8), depth=np.full((16, 20), depth_value, np.float32))
    with pytest.raises(ValueError, match="invalid depth values"):
        capture(None, observation, tmp_path, 0, dict(height=16, width=20))


def test_merge_repairs_interrupted_provenance_copy(tmp_path):
    from data_pipeline.R2R.merge_lerobot import copy_provenance

    root, destination = tmp_path / "source", tmp_path / "assembling/provenance"
    original = root / "meta/provenance/episodes/000000/episode.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"complete":true}\n')
    target = destination / "episodes/000000/episode.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"compl')
    assert copy_provenance((root, destination)) == 1
    assert target.read_bytes() == original.read_bytes()
    assert copy_provenance((root, destination)) == 1


def test_instruction_views_share_only_physical_data(tmp_path):
    source = synthetic_source(tmp_path)
    original = read_source(source / "episodes/b_moving")
    route = dict(representative_id="b_moving", frames=12, physical_sha256="same", episodes=[
        dict(episode_id="b_moving", instruction=dict(instruction_text="Original")),
        dict(episode_id="c_instruction", instruction=dict(instruction_text="Different instruction"))])
    expand_instructions(source, dict(routes=[route], episodes=3))
    cloned = read_source(source / "episodes/c_instruction")
    assert cloned["metadata"]["instruction"] == "Different instruction"
    assert cloned["metadata"]["episode_id"] == "c_instruction"
    assert cloned["source"] == original["source"]
    assert cloned["arrays"]["fm_waypoints_ego"].tobytes() == original["arrays"]["fm_waypoints_ego"].tobytes()


@pytest.mark.parametrize("shared_instruction", [False, True])
def test_real_merge_remaps_tasks_and_indices_preserves_masks_and_resumes(tmp_path, monkeypatch, shared_instruction):
    pytest.importorskip("lerobot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    from data_pipeline.R2R.merge_lerobot import merge_datasets, official_load_smoke, verify_coverage
    from data_pipeline.R2R.validate_lerobot import validate_lerobot

    source = synthetic_source(tmp_path)
    if shared_instruction:
        for path in (source / "episodes").glob("*/episode.json"):
            metadata = json.loads(path.read_text())
            metadata["instruction"] = "Same text on distinct source episodes"
            write_json(path, metadata)
    roots = [tmp_path / "stationary", tmp_path / "moving"]
    export_dataset(source, roots[0], episode_ids=["a_stationary"])
    export_dataset(source, roots[1], episode_ids=["b_moving"])
    output = tmp_path / "merged"
    report = merge_datasets(roots, output, "beta-nav/merged", tmp_path / "journal", workers=2)
    assert report["episodes"] == 2 and report["frames"] == 13
    assert validate_lerobot(output)["all_30_source_arrays_exact"]
    assert official_load_smoke(output)["merged_dataloader_batch"]
    routes = [dict(frames=n, scene_id="synthetic", episodes=[
        dict(episode_id=ep, instruction=dict(instruction_text=
            "Same text on distinct source episodes" if shared_instruction else "Stop in " + ep))])
        for ep, n in (("a_stationary", 1), ("b_moving", 12))]
    plan = dict(frames=13, source_episodes=2, shards=[dict(routes=routes)])
    assert verify_coverage(output, plan)["all_instructions_exact"]
    resumed = merge_datasets(roots, output, "beta-nav/merged", tmp_path / "journal", workers=1)
    assert resumed["file_sha256"] == report["file_sha256"]
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["episode_index"]["max"] == [1.]
    assert stats["index"]["max"] == [12.]
    assert stats["task_index"]["max"] == ([0.] if shared_instruction else [1.])
    assert report["tasks"] == (1 if shared_instruction else 2)
    broken = copy.deepcopy(plan)
    broken["shards"][0]["routes"][0]["episodes"][0]["instruction"]["instruction_text"] = "Wrong"
    with pytest.raises(ValueError, match="instruction"):
        verify_coverage(output, broken)
