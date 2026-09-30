"""CPU tests for the original-LightNav per-step token census wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lightnav.habitat.runner import HabitatEvalConfig

_SPEC = importlib.util.spec_from_file_location(
    "eval_r2r_tokens", Path(__file__).resolve().parents[1] / "scripts" / "eval_r2r_tokens.py"
)
assert _SPEC is not None and _SPEC.loader is not None
collector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collector)

MOVE = {"action": "velocity_control",
        "action_args": {"linear_velocity": 1.0, "angular_velocity": 0.0}}
STOP = {"action": "velocity_control",
        "action_args": {"linear_velocity": -1.0, "angular_velocity": 0.0}}
ENV_INFO = {
    "scene_id": "sceneA", "episode_id": "17", "habitat_time_step": 0.1,
    "lin_vel_range": [0.0, 2.5], "ang_vel_range": [-180.0, 180.0],
}


class FakeEnv:
    def __init__(self, terminate_at=3, fail_at=None):
        self.terminate_at = terminate_at
        self.fail_at = fail_at
        self.closed = False
        self.step_count = 0
        self.reset_count = 0

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        return {"rgb": np.zeros((2, 2, 3)), "instruction": "Go forward"}, dict(ENV_INFO)

    def step(self, action):
        self.step_count += 1
        if self.step_count == self.fail_at:
            raise RuntimeError("simulator failed")
        done = self.step_count == self.terminate_at or action == STOP
        return {"rgb": np.zeros((2, 2, 3))}, 0.0, done, False, {
            "distance_to_goal": np.float32(1.0), "success": done, "spl": 0.5, "ndtw": 0.5,
        }

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, invalid=False):
        self.invalid = invalid
        self.acts = 0
        self.rvq = SimpleNamespace(is_stop=lambda codes: False)

    def reset(self, obs):
        pass

    def act(self, obs, info):
        self.acts += 1
        return STOP if self.invalid else MOVE

    def stop_action(self):
        return STOP

    def get_info(self):
        if self.invalid:
            return {"raw_text": "malformed <act_l0_12><act_l1_999>"}
        return {
            "raw_text": "<act_l0_12><act_l1_34><act_l2_56>",
            "rvq_codes": [12, 34, 56], "cluster_id": 12,
            "predicted_traj": np.array([[0.25, 0.0, 0.0]]), "action_waypoint_index": 0,
        }


def run(tmp_path, *, env=None, policy=None, **overrides):
    env = env or FakeEnv()
    policy = policy or FakePolicy()
    cfg = HabitatEvalConfig(
        model_path="fake-checkpoint", output_dir=str(tmp_path / "results"),
        **{"max_steps": 10, **overrides},
    )
    path = tmp_path / "steps.jsonl"
    results = collector.run_with_token_log(
        cfg, path,
        env_factory=lambda cfg: env,
        policy_factory=lambda *args: policy,
        engine_factory=lambda cfg: (None, None),
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return results, rows, env, policy


def test_full_split_step_coverage_and_identity(tmp_path):
    results, rows, env, policy = run(tmp_path)
    assert len(results) == 1  # The next reset repeats the same scene + episode.
    assert len(rows) == sum(result["steps"] for result in results) == policy.acts == 3
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert all(row["scene_id"] == "sceneA" and row["habitat_episode_id"] == "17" for row in rows)
    assert all(row["rvq_codes"] == [12, 34, 56] for row in rows)
    assert all(row["inference_performed"] and not row["decode_failed"] for row in rows)
    assert rows[-1]["terminated"] and not rows[0]["terminated"]
    assert env.closed and env.reset_count == 2


def test_budget_stop_has_no_stale_model_tokens(tmp_path):
    results, rows, _, policy = run(tmp_path, max_steps=2)
    assert len(rows) == results[0]["steps"] == 2
    assert results[0]["forced_stop"] and policy.acts == 1
    forced = rows[-1]
    assert forced["forced_stop"] and forced["stop"]
    assert not forced["inference_performed"] and not forced["decode_failed"]
    assert forced["raw_text"] == "" and forced["rvq_codes"] is None
    assert forced["cluster_id"] is None and forced["action_waypoint_index"] is None


def test_decode_failure_preserves_raw_emitted_tokens(tmp_path):
    _, rows, _, policy = run(tmp_path, policy=FakePolicy(invalid=True))
    assert len(rows) == policy.acts == 1
    assert rows[0]["decode_failed"] and rows[0]["inference_performed"] and rows[0]["stop"]
    assert rows[0]["raw_text"] == "malformed <act_l0_12><act_l1_999>"
    assert rows[0]["rvq_codes"] is None


def test_failed_env_step_preserves_prior_steps_and_propagates(tmp_path):
    env = FakeEnv(fail_at=2)
    with pytest.raises(RuntimeError, match="simulator failed"):
        run(tmp_path, env=env)
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["step"] == 0
    assert env.closed


@pytest.mark.parametrize("existing", ["steps.jsonl", "results/results.jsonl"])
def test_refuses_existing_data_before_loading_engine(tmp_path, existing):
    path = tmp_path / existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("preserve me\n")
    cfg = HabitatEvalConfig(model_path="unused", output_dir=str(tmp_path / "results"))
    with pytest.raises(FileExistsError):
        collector.run_with_token_log(
            cfg, tmp_path / "steps.jsonl",
            engine_factory=lambda cfg: pytest.fail("must reject before model load"),
        )
    assert path.read_text() == "preserve me\n"


def test_duplicate_episode_journal_rejected(tmp_path):
    journal = collector.StepTokenJournal(tmp_path / "steps.jsonl")
    try:
        journal.reset(ENV_INFO)
        journal.capture({}, STOP, forced_stop=True, latency_ms=0.0)
        journal.commit(STOP, True, False, {})
        journal.reset(ENV_INFO)
        journal.capture({}, STOP, forced_stop=True, latency_ms=0.0)
        with pytest.raises(RuntimeError, match="Duplicate token episode"):
            journal.commit(STOP, True, False, {})
    finally:
        journal.close()


@pytest.mark.parametrize("overrides", [
    {"action_output": "fm"}, {"beta_checkpoint": "beta.pt"}, {"oracle_stop_distance": 3.0},
])
def test_rejects_nonstandard_policy(tmp_path, overrides):
    cfg = HabitatEvalConfig(model_path="unused", output_dir=str(tmp_path), **overrides)
    with pytest.raises(ValueError):
        collector.run_with_token_log(cfg, tmp_path / "steps.jsonl")
    assert not (tmp_path / "steps.jsonl").exists()
