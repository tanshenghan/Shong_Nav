#!/usr/bin/env python3
"""Run original LightNav Habitat evaluation with a durable, image-free token journal.

All normal ``lightnav-eval-habitat`` arguments are supported, plus required
``--token-log PATH``. Each JSONL row is one successfully completed environment
step; ``step`` is zero-based and ``inference_performed`` excludes forced STOP.
``raw_text`` is the original decoded model output, including malformed output.
It is not re-tokenized or represented as exact generation token IDs.

Example::

    python scripts/eval_r2r_tokens.py --model_path checkpoints/LightNav-0 \
        --server tcp://127.0.0.1:5555 --episodes -1 --output_dir outputs/token_eval \
        --token-log outputs/token_eval/steps.jsonl

Use a fresh output directory for each shard/run. Existing journals are never
appended to, and nonempty results.jsonl is rejected before loading the model.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _stop_command(action: Any, info: dict[str, Any]) -> bool:
    """Interpret the normalized command using the server's physical speed ranges."""
    args = action.get("action_args", {}) if isinstance(action, dict) else {}
    if "linear_velocity" not in args or "angular_velocity" not in args:
        return False
    speeds = []
    for name, key in (("linear_velocity", "lin_vel_range"),
                      ("angular_velocity", "ang_vel_range")):
        low, high = info[key]
        raw = min(1.0, max(-1.0, float(args[name])))
        speeds.append(float(low) + (raw + 1.0) * 0.5 * (float(high) - float(low)))
    return (
        abs(speeds[0]) < float(info.get("min_abs_lin_speed", 0.01))
        and abs(speeds[1]) < float(info.get("min_abs_ang_speed", 0.5))
    )


class StepTokenJournal:
    """Shared state between policy and environment wrappers; no model dependencies."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8")
        self.scene_id = ""
        self.habitat_episode_id = ""
        self.step = 0
        self.env_info: dict[str, Any] = {}
        self.pending: dict[str, Any] | None = None
        self._started_episodes: set[tuple[str, str]] = set()

    def reset(self, info: dict[str, Any]) -> None:
        self.sync()
        if "scene_id" not in info or "episode_id" not in info:
            raise ValueError("Token logging requires reset scene_id and episode_id")
        self.scene_id = str(info["scene_id"])
        self.habitat_episode_id = str(info["episode_id"])
        self.step = 0
        self.env_info = dict(info)
        self.pending = None

    def capture(self, info: dict[str, Any], action: Any, *, forced_stop: bool,
                latency_ms: float, explicit_stop: bool = False) -> None:
        if self.pending is not None:
            raise RuntimeError("Unconsumed token capture: an environment step is missing")
        inference_performed = not forced_stop
        self.pending = {
            "raw_text": str(info.get("raw_text", "") or "") if inference_performed else "",
            "rvq_codes": info.get("rvq_codes") if inference_performed else None,
            "cluster_id": info.get("cluster_id") if inference_performed else None,
            "action_waypoint_index": info.get("action_waypoint_index") if inference_performed else None,
            "forced_stop": forced_stop,
            "inference_performed": inference_performed,
            "stop": _stop_command(action, self.env_info),
            "explicit_stop": bool(explicit_stop) if inference_performed else False,
            "decode_failed": inference_performed and info.get("predicted_traj") is None,
            "latency_ms": float(latency_ms),
            "action": _json_safe(action),
        }

    def commit(self, action: Any, terminated: bool, truncated: bool,
               info: dict[str, Any]) -> None:
        if self.pending is None:
            raise RuntimeError("Environment step has no policy token capture")
        key = (self.scene_id, self.habitat_episode_id)
        if self.step == 0:
            if key in self._started_episodes:
                raise RuntimeError(f"Duplicate token episode: {key!r}")
            self._started_episodes.add(key)
        if self.pending["action"] != _json_safe(action):
            raise RuntimeError("Executed action does not match token capture")
        row = {
            "schema_version": 1,
            "scene_id": self.scene_id,
            "habitat_episode_id": self.habitat_episode_id,
            "step": self.step,
            **self.pending,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "distance_to_goal": info.get("distance_to_goal"),
            "termination_reason": info.get("termination_reason", ""),
        }
        self._file.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
        self._file.flush()
        self.pending = None
        self.step += 1
        if terminated or truncated:
            self.sync()

    def sync(self) -> None:
        if not self._file.closed:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        try:
            self.sync()
        finally:
            self._file.close()


class TokenLoggingPolicy:
    def __init__(self, policy: Any, journal: StepTokenJournal) -> None:
        self._policy = policy
        self._journal = journal

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def act(self, obs: Any, info: Any = None) -> Any:
        start = time.monotonic()
        action = self._policy.act(obs, info)
        policy_info = self._policy.get_info()
        codes = policy_info.get("rvq_codes")
        rvq = getattr(self._policy, "rvq", None)
        explicit_stop = codes is not None and rvq is not None and rvq.is_stop(codes)
        self._journal.capture(
            policy_info, action, forced_stop=False,
            latency_ms=(time.monotonic() - start) * 1000.0, explicit_stop=explicit_stop,
        )
        return action

    def stop_action(self) -> Any:
        start = time.monotonic()
        action = self._policy.stop_action()
        # The underlying policy retains its previous prediction on stop_action().
        # Never copy that stale prediction into a step where inference did not run.
        self._journal.capture(
            {}, action, forced_stop=True, latency_ms=(time.monotonic() - start) * 1000.0,
        )
        return action


class TokenLoggingEnv:
    def __init__(self, env: Any, journal: StepTokenJournal) -> None:
        self._env = env
        self._journal = journal

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self) -> Any:
        obs, info = self._env.reset()
        self._journal.reset(info)
        return obs, info

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        _obs, _reward, terminated, truncated, info = result
        self._journal.commit(action, terminated, truncated, info)
        return result


def run_with_token_log(cfg: Any, token_log: str | Path, *, env_factory: Any = None,
                       policy_factory: Any = None, engine_factory: Any = None) -> list[dict]:
    """Inject logging around the existing evaluator without altering policy behavior."""
    if cfg.action_output != "token" or cfg.beta_checkpoint:
        raise ValueError("Token census requires original LightNav; Beta/FM evaluation is rejected")
    if cfg.oracle_stop_distance is not None:
        raise ValueError("Token census requires standard evaluation without oracle STOP")
    results_path = Path(cfg.output_dir or "output/habitat_eval") / "results.jsonl"
    if results_path.exists() and results_path.stat().st_size:
        raise FileExistsError(f"Refusing to append to nonempty evaluation results: {results_path}")
    if Path(token_log).resolve() == results_path.resolve():
        raise ValueError("Token log and results.jsonl must be different files")

    from lightnav.habitat.runner import (
        _default_env_factory,
        build_velocity_policy,
        run_habitat_eval,
    )

    make_env = env_factory or _default_env_factory
    make_policy = policy_factory or build_velocity_policy
    journal = StepTokenJournal(token_log)
    try:
        return run_habitat_eval(
            cfg,
            env_factory=lambda conf: TokenLoggingEnv(make_env(conf), journal),
            policy_factory=lambda conf, engine, bundle, info: TokenLoggingPolicy(
                make_policy(conf, engine, bundle, info), journal,
            ),
            engine_factory=engine_factory,
        )
    finally:
        journal.close()


def main(argv: list[str] | None = None) -> int:
    from lightnav.cli.eval_habitat import build_parser, config_from_args

    parser = build_parser()
    parser.prog = "eval_r2r_tokens.py"
    parser.description = __doc__.split("\n\n", 1)[0]
    parser.add_argument(
        "--token-log", required=True,
        help="New JSONL file for every environment step, including raw model text; no images.",
    )
    args = parser.parse_args(argv)
    run_with_token_log(config_from_args(args), args.token_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
