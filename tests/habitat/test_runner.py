"""run_habitat_eval: the episode loop over a fake env / policy, and action-decoder resolution.

No GPU, no simulator: the engine, env and policy factories are injected, so what is
exercised is the loop (episode dedupe / cap / language filter), the per-episode result
records, ``results.jsonl`` and the two ``summary.json`` schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lightnav.habitat.policy import TrajVocabVLNCEPolicy, extract_instruction
from lightnav.habitat.runner import (
    HabitatEvalConfig,
    _safe_scalar,
    build_velocity_policy,
    resolve_action_decoder,
    run_habitat_eval,
)

# (scene_id, episode_id, language, instruction). Note the repeated episode_id "1" in a
# second scene: HM3D episode ids are scene-local, so dedupe must key on the pair.
EPISODES = [
    ("sceneA", "1", "en-US", "go to the sofa"),
    ("sceneA", "2", "hi-IN", "turn around twice"),
    ("sceneB", "1", "en-IN", "leave the room"),
]

VELOCITY_KEYS = {
    "habitat_time_step": 1.0,
    "lin_vel_range": [0.0, 0.25],
    "ang_vel_range": [-30.0, 30.0],
}


class FakeEnv:
    """Cycles through EPISODES forever (like a Habitat env iterator wrapping around)."""

    def __init__(
        self,
        *,
        steps_to_terminate: int = 3,
        object_category: str | None = None,
        with_oracle: bool = True,
        with_language: bool = True,
    ) -> None:
        self.steps_to_terminate = steps_to_terminate
        self.object_category = object_category
        self.with_oracle = with_oracle
        self.with_language = with_language
        self.i = 0
        self.reset_calls = 0
        self.actions: list = []
        self.closed = False
        self._current = EPISODES[0]
        self._step = 0

    def _base_info(self) -> dict:
        scene, ep, lang, _ = self._current
        info = {"episode_id": ep, "scene_id": scene, "steps": self._step}
        if self.with_language:
            info["language"] = lang
        if self.object_category:
            info["object_category"] = self.object_category
        return info

    def reset(self):
        self._current = EPISODES[self.i % len(EPISODES)]
        self.i += 1
        self.reset_calls += 1
        self._step = 0
        obs = {"rgb": np.zeros((4, 4, 3), np.uint8), "instruction": {"text": self._current[3]}}
        info = self._base_info()
        info["distance_to_goal"] = 5.0  # reset-time distance: must NOT count toward min_distance
        info.update(VELOCITY_KEYS)
        return obs, info

    def step(self, action):
        self.actions.append(action)
        self._step += 1
        terminated = self._step >= self.steps_to_terminate
        scene = self._current[0]
        info = self._base_info()
        info.update(
            {
                "distance_to_goal": np.float32(5.0 - self._step),
                "success": 1.0 if scene == "sceneA" else 0.0,
                "spl": 0.5,
                "ndtw": 0.7,
                "soft_spl": np.float32(0.4),
            }
        )
        if self.with_oracle:
            info["oracle_success"] = 1.0
        if terminated:
            info["termination_reason"] = "agent_stop"
            info["termination_details"] = {"stop_step": self._step}
        obs = {"rgb": np.zeros((4, 4, 3), np.uint8)}
        return obs, 0.0, terminated, False, info

    def close(self) -> None:
        self.closed = True


class FakePolicy:
    def __init__(self) -> None:
        self.resets: list[str] = []
        self.acts = 0

    def reset(self, obs) -> None:
        self.resets.append(extract_instruction(obs))

    def act(self, obs, info) -> dict:
        self.acts += 1
        return {
            "action": "velocity_control",
            "action_args": {"linear_velocity": 0.1, "angular_velocity": 0.0},
        }

    def get_info(self) -> dict:
        return {"raw_text": "<traj_1>", "cluster_id": 1, "action_waypoint_index": 0}


class Harness:
    """Bundles the injected factories and records how they were called."""

    def __init__(self, env: FakeEnv) -> None:
        self.env = env
        self.policy = FakePolicy()
        self.policy_factory_calls: list = []
        self.engine = SimpleNamespace(max_new_tokens=8)
        self.bundle = SimpleNamespace(num_history_frames=4, action_method="flat", tokenizer=None)

    def engine_factory(self, cfg):
        return self.engine, self.bundle

    def env_factory(self, cfg):
        return self.env

    def policy_factory(self, cfg, engine, bundle, info):
        self.policy_factory_calls.append((engine, bundle, dict(info)))
        return self.policy

    def run(self, cfg: HabitatEvalConfig):
        return run_habitat_eval(
            cfg,
            env_factory=self.env_factory,
            policy_factory=self.policy_factory,
            engine_factory=self.engine_factory,
        )


def _cfg(tmp_path: Path, **overrides) -> HabitatEvalConfig:
    fields = dict(model_path="/path/to/model", output_dir=str(tmp_path / "out"), max_steps=50)
    fields.update(overrides)
    return HabitatEvalConfig(**fields)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- episode loop ----------------------------------------------------------------------


def test_full_split_runs_until_the_first_repeated_scene_episode_key(tmp_path):
    h = Harness(FakeEnv())
    results = h.run(_cfg(tmp_path, episodes=-1))

    assert [r["episode_id"] for r in results] == ["episode_000", "episode_001", "episode_002"]
    assert [(r["scene_id"], r["habitat_episode_id"]) for r in results] == [
        ("sceneA", "1"),
        ("sceneA", "2"),
        ("sceneB", "1"),
    ]
    # Three episodes + the fourth reset that reveals the wrap (no step is taken for it).
    assert h.env.reset_calls == 4
    assert h.policy.acts == 9
    assert h.env.closed is True
    # The policy is built lazily, once, from the first reset's info.
    assert len(h.policy_factory_calls) == 1
    engine, bundle, info = h.policy_factory_calls[0]
    assert engine is h.engine and bundle is h.bundle
    assert info["habitat_time_step"] == 1.0
    assert info["lin_vel_range"] == [0.0, 0.25] and info["ang_vel_range"] == [-30.0, 30.0]
    assert h.policy.resets == ["go to the sofa", "turn around twice", "leave the room"]
    # Every action sent to the env is the policy's velocity dict.
    assert all(a["action"] == "velocity_control" for a in h.env.actions)


def test_episode_result_record_fields(tmp_path):
    h = Harness(FakeEnv())
    results = h.run(_cfg(tmp_path))
    r = results[0]

    assert set(r) == {
        "episode_id",
        "habitat_episode_id",
        "raw_episode_id",
        "scene_id",
        "rollout_idx",
        "success",
        "oracle_success",
        "spl",
        "ndtw",
        "soft_spl",
        "object_category",
        "steps",
        "final_distance",
        "min_distance",
        "instruction",
        "termination_reason",
        "forced_stop",
        "oracle_stop",
        "stop_source",
        "policy_event_counts",
        "termination_details",
    }
    assert r["rollout_idx"] == 0
    assert r["success"] == 1.0
    assert r["oracle_success"] is True  # from the env's oracle_success metric
    assert r["spl"] == 0.5 and r["ndtw"] == 0.7
    assert r["soft_spl"] == pytest.approx(0.4)
    assert r["object_category"] == ""
    assert r["steps"] == 3
    assert float(r["final_distance"]) == pytest.approx(2.0)
    assert r["min_distance"] == pytest.approx(2.0)  # min over steps; reset-time 5.0 excluded
    assert r["instruction"] == "go to the sofa"
    assert r["termination_reason"] == "agent_stop"
    assert r["stop_source"] == "environment_stop"
    assert r["oracle_stop"] is False and r["policy_event_counts"] == {}
    assert r["termination_details"] == {"stop_step": 3}
    assert results[2]["success"] == 0.0  # sceneB fails


def test_results_jsonl_is_appended_per_episode_and_json_safe(tmp_path):
    cfg = _cfg(tmp_path)
    h = Harness(FakeEnv())
    results = h.run(cfg)

    rows = _read_jsonl(Path(cfg.output_dir) / "results.jsonl")
    assert len(rows) == 3
    assert [row["episode_id"] for row in rows] == [r["episode_id"] for r in results]
    assert isinstance(rows[0]["final_distance"], float)  # np.float32 -> plain float
    assert isinstance(rows[0]["soft_spl"], float)
    assert rows[0]["termination_details"] == {"stop_step": 3}

    # A second run appends rather than truncates.
    Harness(FakeEnv()).run(_cfg(tmp_path, episodes=1))
    assert len(_read_jsonl(Path(cfg.output_dir) / "results.jsonl")) == 4


def test_episodes_cap_stops_without_an_extra_reset(tmp_path):
    h = Harness(FakeEnv())
    results = h.run(_cfg(tmp_path, episodes=2))

    assert [r["episode_id"] for r in results] == ["episode_000", "episode_001"]
    assert h.env.reset_calls == 2
    assert h.policy.acts == 6


def test_language_filter_skips_without_stepping_and_does_not_count(tmp_path):
    h = Harness(FakeEnv())
    results = h.run(_cfg(tmp_path, episodes=-1, languages=["en-US", "en-IN"]))

    assert [(r["scene_id"], r["habitat_episode_id"]) for r in results] == [
        ("sceneA", "1"),
        ("sceneB", "1"),
    ]
    assert [r["episode_id"] for r in results] == ["episode_000", "episode_001"]  # numbered after filter
    assert h.policy.resets == ["go to the sofa", "leave the room"]  # hi-IN episode never reset
    assert h.policy.acts == 6
    assert h.env.reset_calls == 4  # the filtered episode still consumed a reset; then the wrap


def test_language_filter_with_a_cap_counts_only_admitted_episodes(tmp_path):
    h = Harness(FakeEnv())
    results = h.run(_cfg(tmp_path, episodes=2, languages=["en-US", "en-IN"]))
    assert [(r["scene_id"], r["habitat_episode_id"]) for r in results] == [
        ("sceneA", "1"),
        ("sceneB", "1"),
    ]
    assert h.env.reset_calls == 3


def test_no_language_filter_is_applied_when_languages_is_unset(tmp_path):
    h = Harness(FakeEnv(with_language=False))
    results = h.run(_cfg(tmp_path))
    assert len(results) == 3


def test_episode_manifest_runs_only_exact_scene_episode_pairs(tmp_path):
    manifest = tmp_path / "episodes.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes": [
                    {"scene_id": "sceneA", "habitat_episode_id": "2"},
                    {"scene_id": "sceneB", "habitat_episode_id": "1"},
                ]
            }
        )
    )
    h = Harness(FakeEnv())

    results = h.run(_cfg(tmp_path, episode_manifest=str(manifest)))

    assert [(r["scene_id"], r["habitat_episode_id"]) for r in results] == [
        ("sceneA", "2"),
        ("sceneB", "1"),
    ]
    assert h.policy.resets == ["turn around twice", "leave the room"]
    assert h.env.reset_calls == 4


def test_max_steps_truncation_leaves_success_false_and_reason_unknown(tmp_path):
    h = Harness(FakeEnv(steps_to_terminate=100))
    results = h.run(_cfg(tmp_path, episodes=1, max_steps=4))
    r = results[0]

    assert r["steps"] == 4
    assert r["success"] is False  # the env never terminated
    assert r["termination_reason"] == "unknown"
    assert r["termination_details"] == {}
    assert r["min_distance"] == pytest.approx(1.0)
    assert h.policy.acts == 4


def test_oracle_success_falls_back_to_the_distance_threshold(tmp_path):
    vlnce = Harness(FakeEnv(with_oracle=False)).run(_cfg(tmp_path, episodes=1))
    assert vlnce[0]["oracle_success"] is True  # min_distance 2.0 < 3.0 m (VLN-CE)

    objnav = Harness(FakeEnv(with_oracle=False, object_category="chair")).run(
        _cfg(tmp_path / "o", episodes=1)
    )
    assert objnav[0]["oracle_success"] is False  # 2.0 >= 0.1 m (ObjectNav)


# -- summaries ---------------------------------------------------------------------------


def test_vlnce_summary_json_schema(tmp_path):
    cfg = _cfg(tmp_path, backend="hf")
    Harness(FakeEnv()).run(cfg)

    summary = json.loads((Path(cfg.output_dir) / "summary.json").read_text())
    assert summary["num_episodes"] == 3
    assert summary["metrics"] == {
        "SR_success_rate_pct": pytest.approx(66.67, abs=0.01),
        "OS_oracle_success_pct": 100.0,
        "SPL_pct": 50.0,
        "NDTW_pct": 70.0,
        "NE_navigation_error_m": 2.0,
    }
    assert summary["table_format"] == "66.7 / 100.0 / 50.0 / 70.0 / 2.00"
    assert summary["avg_steps"] == 3.0
    assert isinstance(summary["total_time_sec"], float)
    assert summary["model"] == "/path/to/model" and summary["backend"] == "hf"
    assert "per_category_sr" not in summary and "pass_at_n" not in summary
    ep = summary["episodes"][0]
    assert set(ep) == {
        "episode_id",
        "habitat_episode_id",
        "scene_id",
        "rollout_idx",
        "success",
        "oracle_success",
        "spl",
        "ndtw",
        "steps",
        "final_distance",
        "min_distance",
        "instruction",
        "stop_source",
        "oracle_stop",
        "policy_event_counts",
    }
    assert ep["success"] == 1.0 and ep["oracle_success"] == 1.0
    assert ep["instruction"] == "go to the sofa"
    assert len(summary["episodes"]) == 3


def test_objectnav_summary_json_schema(tmp_path):
    cfg = _cfg(tmp_path)
    Harness(FakeEnv(object_category="chair")).run(cfg)

    summary = json.loads((Path(cfg.output_dir) / "summary.json").read_text())
    assert summary["num_episodes"] == 3
    assert summary["metrics"] == {
        "SR_success_rate_pct": pytest.approx(66.67, abs=0.01),
        "SPL_pct": 50.0,
        "SoftSPL_pct": pytest.approx(40.0, abs=0.01),
        "NE_navigation_error_m": 2.0,
    }
    assert summary["table_format"] == "66.7 / 50.0 / 40.0 / 2.00"
    assert summary["per_category_sr"] == {"chair": pytest.approx(66.67, abs=0.01)}
    assert summary["model"] == "/path/to/model"
    ep = summary["episodes"][0]
    assert set(ep) == {
        "episode_id",
        "habitat_episode_id",
        "scene_id",
        "rollout_idx",
        "success",
        "spl",
        "soft_spl",
        "object_category",
        "steps",
        "final_distance",
        "min_distance",
    }
    assert ep["object_category"] == "chair"


def test_safe_scalar_accepts_scalars_sequences_and_arrays():
    assert _safe_scalar(None, default=7.0) == 7.0
    assert _safe_scalar([]) == 0.0
    assert _safe_scalar([2.5, 9.0]) == 2.5
    assert _safe_scalar(np.array([1.5])) == 1.5
    assert _safe_scalar(np.float32(0.25)) == pytest.approx(0.25)
    assert _safe_scalar("nope", default=-1.0) == -1.0


# -- policy construction from the env's velocity settings ------------------------------------


def _centroids(K: int = 4, H: int = 10) -> np.ndarray:
    c = np.zeros((K, H, 3), dtype=np.float32)
    c[1, :, 0] = 0.25
    return c


def _write_vocab(d: Path, K: int = 4, H: int = 10) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"centroids_whole_chunk_K{K}_h{H}.npy"
    np.save(path, _centroids(K, H))
    return path


def _stub_engine(action_method: str = "flat"):
    bundle = SimpleNamespace(num_history_frames=6, action_method=action_method, tokenizer=None)
    engine = SimpleNamespace(
        bundle=bundle, max_new_tokens=8, reset_episode_state=lambda: None
    )
    return engine, bundle


def test_build_velocity_policy_requires_the_velocity_keys(tmp_path):
    engine, bundle = _stub_engine()
    cfg = _cfg(tmp_path, traj_vocab_path=str(_write_vocab(tmp_path / "vocab")), K=4, horizon=10)
    with pytest.raises(KeyError, match="habitat_time_step"):
        build_velocity_policy(cfg, engine, bundle, {"lin_vel_range": [0, 1]})


def test_build_velocity_policy_uses_env_ranges_and_bundle_history(tmp_path):
    engine, bundle = _stub_engine()
    cfg = _cfg(tmp_path, traj_vocab_path=str(_write_vocab(tmp_path / "vocab")), K=4, horizon=10)
    info = {"habitat_time_step": 0.1, "lin_vel_range": [0, 2.5], "ang_vel_range": [-300, 300]}

    policy = build_velocity_policy(cfg, engine, bundle, info)

    assert isinstance(policy, TrajVocabVLNCEPolicy)
    assert policy.dt == 0.1
    assert policy.lin_vel_range == (0.0, 2.5) and policy.ang_vel_range == (-300.0, 300.0)
    assert policy.num_history_frames == 6  # from the bundle
    assert policy.K == 4 and policy.rvq is None
    assert engine.max_new_tokens == 8  # no tokenizer -> budget untouched

    cfg.num_history_frames = 3
    assert build_velocity_policy(cfg, engine, bundle, info).num_history_frames == 3


# -- action decoder resolution ------------------------------------------------------------------


def _model_dir(tmp_path: Path, tasks: dict | None = None) -> Path:
    model = tmp_path / "exp" / "checkpoints" / "global_step_5000" / "hf_ckpt"
    model.mkdir(parents=True)
    if tasks is not None:
        (tmp_path / "exp" / "eval_config.json").write_text(
            json.dumps({"version": 1, "common": {}, "tasks": tasks})
        )
    return model


def test_explicit_traj_vocab_path_wins_and_takes_K_H_from_the_snapshot(tmp_path):
    model = _model_dir(
        tmp_path, {"vlnce": {"traj_vocab_K": 4, "predict_horizon": 10, "num_history_frames": 8}}
    )
    vocab_dir = _write_vocab(tmp_path / "vocab").parent
    cfg = _cfg(tmp_path, model_path=str(model), traj_vocab_path=str(vocab_dir))

    centroids, rvq = resolve_action_decoder(cfg, SimpleNamespace(action_method="flat"))

    assert rvq is None
    assert centroids.shape == (4, 10, 3)


def test_explicit_traj_vocab_npy_with_cli_K_H(tmp_path):
    npy = _write_vocab(tmp_path / "vocab", K=8, H=5)
    cfg = _cfg(tmp_path, traj_vocab_path=str(npy), K=8, horizon=5)
    centroids, rvq = resolve_action_decoder(cfg, SimpleNamespace(action_method="flat"))
    assert centroids.shape == (8, 5, 3) and rvq is None


def test_explicit_rvq_bundle_reads_the_horizon_from_the_manifest(tmp_path, rvq_bundle_writer):
    bundle_dir = tmp_path / "bundle"
    rvq_bundle_writer(bundle_dir, horizon=10)
    cfg = _cfg(tmp_path, action_tokenizer_bundle=str(bundle_dir))

    centroids, rvq = resolve_action_decoder(cfg, SimpleNamespace(action_method="rvq"))

    assert centroids is None
    assert rvq.horizon == 10 and rvq.levels == [4, 8, 8]


def test_explicit_rvq_bundle_horizon_mismatch_is_loud(tmp_path, rvq_bundle_writer):
    bundle_dir = tmp_path / "bundle"
    rvq_bundle_writer(bundle_dir, horizon=10)
    cfg = _cfg(tmp_path, action_tokenizer_bundle=str(bundle_dir), horizon=5)
    with pytest.raises(RuntimeError, match="horizon"):
        resolve_action_decoder(cfg, SimpleNamespace(action_method="rvq"))


def test_both_explicit_decoders_are_rejected(tmp_path):
    cfg = _cfg(tmp_path, traj_vocab_path="/path/to/vocab", action_tokenizer_bundle="/path/to/b")
    with pytest.raises(ValueError, match="only one"):
        resolve_action_decoder(cfg, SimpleNamespace(action_method="flat"))


def test_snapshot_rvq_bundle_is_used_when_present(tmp_path, rvq_bundle_writer):
    bundle_dir = tmp_path / "bundle"
    rvq_bundle_writer(bundle_dir, horizon=10)
    model = _model_dir(
        tmp_path,
        {
            "vlnce": {
                "predict_horizon": 10,
                "action_tokenizer": {"method": "rvq", "bundle_path": str(bundle_dir)},
            }
        },
    )
    centroids, rvq = resolve_action_decoder(
        _cfg(tmp_path, model_path=str(model)), SimpleNamespace(action_method="rvq")
    )
    assert centroids is None and rvq.levels == [4, 8, 8]


def test_snapshot_flat_vocab_is_used_when_the_centroids_file_exists(tmp_path):
    vocab_dir = _write_vocab(tmp_path / "vocab").parent
    model = _model_dir(
        tmp_path,
        {"vlnce": {"traj_vocab_path": str(vocab_dir), "traj_vocab_K": 4, "predict_horizon": 10}},
    )
    centroids, rvq = resolve_action_decoder(
        _cfg(tmp_path, model_path=str(model)), SimpleNamespace(action_method="flat")
    )
    assert rvq is None and centroids.shape == (4, 10, 3)


def test_stale_snapshot_paths_fall_through_to_sibling_dirs(tmp_path, rvq_bundle_writer):
    # The snapshot names paths that no longer exist (training-time locations).
    model = _model_dir(
        tmp_path,
        {
            "vlnce": {
                "traj_vocab_path": "/path/to/missing/vocab",
                "traj_vocab_K": 4,
                "predict_horizon": 10,
                "action_tokenizer": {"method": "rvq", "bundle_path": "/path/to/missing/bundle"},
            }
        },
    )
    rvq_bundle_writer(model / "action_tokenizer", horizon=10)
    centroids, rvq = resolve_action_decoder(
        _cfg(tmp_path, model_path=str(model)), SimpleNamespace(action_method="rvq")
    )
    assert centroids is None and rvq.horizon == 10

    # Flat sibling: <model>/traj_vocab/centroids_whole_chunk_K{K}_h{H}.npy with the snapshot K/H.
    model2 = _model_dir(
        tmp_path / "second",
        {"vlnce": {"traj_vocab_path": "/path/to/missing", "traj_vocab_K": 4, "predict_horizon": 10}},
    )
    _write_vocab(model2 / "traj_vocab", K=4, H=10)
    centroids, rvq = resolve_action_decoder(
        _cfg(tmp_path, model_path=str(model2)), SimpleNamespace(action_method="flat")
    )
    assert rvq is None and centroids.shape == (4, 10, 3)


def test_sibling_dir_uses_the_default_K_H_without_a_snapshot(tmp_path):
    model = _model_dir(tmp_path)
    _write_vocab(model / "traj_vocab", K=256, H=10)
    centroids, _ = resolve_action_decoder(
        _cfg(tmp_path, model_path=str(model)), SimpleNamespace(action_method="flat")
    )
    assert centroids.shape == (256, 10, 3)


def test_unresolvable_decoder_is_a_clear_error(tmp_path):
    model = _model_dir(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        resolve_action_decoder(
            _cfg(tmp_path, model_path=str(model)), SimpleNamespace(action_method="rvq")
        )
    message = str(exc.value)
    assert "--action_tokenizer_bundle" in message
    assert "action_tokenizer/manifest.json" in message
    # an RVQ checkpoint never falls back to flat centroids, so none are listed
    assert "centroids_whole_chunk" not in message


# -- visualisation: --save_video / --record_dir ----------------------------------------------


class FrameEnv(FakeEnv):
    """FakeEnv with camera-sized frames so the overlay has something to draw on."""

    SIZE = (36, 64)

    def _rgb(self) -> np.ndarray:
        rgb = np.zeros((*self.SIZE, 3), np.uint8)
        rgb[..., 0] = 40 + 50 * self._step
        rgb[..., 2] = 120
        return rgb

    def reset(self):
        obs, info = super().reset()
        obs["rgb"] = self._rgb()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs["rgb"] = self._rgb()
        return obs, reward, terminated, truncated, info


class TrajPolicy(FakePolicy):
    """Exposes a decoded chunk (and a pointing token) like the real policy does."""

    def get_info(self) -> dict:
        info = super().get_info()
        info["raw_text"] = "<apos_650><opos_114>"
        info["predicted_traj"] = np.tile(np.array([[0.25, 0.0, 0.05]], np.float32), (10, 1))
        return info


def _video_frames(path: Path) -> list[tuple[int, ...]]:
    import imageio.v2 as imageio

    with imageio.get_reader(str(path)) as reader:
        return [frame.shape for frame in reader]


def test_save_video_writes_one_mp4_per_episode_with_a_terminal_frame(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    cfg = _cfg(tmp_path, episodes=2, save_video=True, video_fps=5)
    h = Harness(FrameEnv(steps_to_terminate=3))
    h.policy = TrajPolicy()

    results = h.run(cfg)

    videos = Path(cfg.output_dir) / "videos"
    assert sorted(p.name for p in videos.iterdir()) == ["episode_000.mp4", "episode_001.mp4"]
    assert [r["video"] for r in results] == ["videos/episode_000.mp4", "videos/episode_001.mp4"]
    assert [r["steps"] for r in results] == [3, 3]
    for r in results:
        shapes = _video_frames(Path(cfg.output_dir) / r["video"])
        assert len(shapes) == 3 + 1  # one frame per policy step + the terminal observation
        assert shapes[0] == (36, 64, 3)
    assert not list(videos.glob("*.partial*"))
    # results.jsonl carries the same relative path.
    rows = _read_jsonl(Path(cfg.output_dir) / "results.jsonl")
    assert [row["video"] for row in rows] == ["videos/episode_000.mp4", "videos/episode_001.mp4"]


def test_save_video_handles_a_policy_without_a_decoded_chunk(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    cfg = _cfg(tmp_path, episodes=1, save_video=True)
    h = Harness(FrameEnv(steps_to_terminate=2))  # FakePolicy: no predicted_traj

    (result,) = h.run(cfg)

    assert result["video"] == "videos/episode_000.mp4"
    assert len(_video_frames(Path(cfg.output_dir) / result["video"])) == 3


def test_save_video_pads_odd_frames_for_the_encoder(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")

    class OddEnv(FrameEnv):
        SIZE = (35, 63)

    cfg = _cfg(tmp_path, episodes=1, save_video=True)
    (result,) = Harness(OddEnv(steps_to_terminate=2)).run(cfg)
    assert _video_frames(Path(cfg.output_dir) / result["video"])[0] == (36, 64, 3)


def test_save_video_false_writes_no_video_and_no_video_key(tmp_path):
    cfg = _cfg(tmp_path, episodes=1)
    assert cfg.save_video is False and cfg.record_dir == ""
    assert (cfg.video_fps, cfg.hfov_deg, cfg.cam_height, cfg.waypoint_dt_s) == (10, 120.0, 0.88, 0.1)

    (result,) = Harness(FrameEnv()).run(cfg)

    assert "video" not in result
    assert not (Path(cfg.output_dir) / "videos").exists()
    assert "video" not in _read_jsonl(Path(cfg.output_dir) / "results.jsonl")[0]


def test_save_video_fails_fast_when_the_video_extra_is_missing(tmp_path, monkeypatch):
    from lightnav.habitat import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_VIDEO_MODULES", ("cv2", "lightnav_no_such_video_module"))
    h = Harness(FrameEnv())

    with pytest.raises(ImportError, match=r"lightnav\[video\]") as exc:
        h.run(_cfg(tmp_path, episodes=1, save_video=True))

    assert "lightnav_no_such_video_module" in str(exc.value)
    assert h.env.reset_calls == 0  # before the engine and the env were built
    assert h.policy_factory_calls == []


def test_record_dir_writes_episode_dirs_that_lightnav_render_finds(tmp_path):
    from lightnav.viz.render_episode import find_episode_dirs, load_manifest, load_records

    record_root = tmp_path / "rec"
    cfg = _cfg(tmp_path, episodes=2, record_dir=str(record_root), hfov_deg=90.0, cam_height=0.5)
    h = Harness(FrameEnv(steps_to_terminate=3))
    h.policy = TrajPolicy()

    results = h.run(cfg)

    assert all("video" not in r for r in results)  # recording alone renders nothing
    episodes = find_episode_dirs([record_root])
    assert [(e.parent.name, e.name) for e in episodes] == [
        ("eval", "episode_000"),
        ("eval", "episode_001"),
    ]
    assert not list(record_root.rglob("actions.jsonl"))
    assert not list(record_root.rglob("*.tmp"))

    ep = episodes[0]
    records = load_records(ep)
    assert [r["step"] for r in records] == [0, 1, 2]
    assert sorted(p.name for p in ep.glob("*.jpg")) == [
        "image_000000.jpg", "image_000001.jpg", "image_000002.jpg",
    ]
    r0 = records[0]
    assert r0["instruction"] == "go to the sofa"
    assert r0["raw_text"] == "<apos_650><opos_114>"
    assert r0["waypoints"] == [[0.25, 0.0, pytest.approx(0.05)]] * 10
    assert r0["stop"] is False and r0["visible"] is None
    assert r0["pointing"]["frame_size"] == [64, 36] and r0["pointing"]["apos_state"] == "point"
    assert r0["frame_size"] == [64, 36]
    assert r0["latency_ms"] >= 0.0
    assert (r0["episode_id"], r0["habitat_episode_id"], r0["scene_id"]) == ("episode_000", "1", "sceneA")

    manifest = load_manifest(ep)
    assert manifest["task"] == "habitat" and manifest["model_path"] == "/path/to/model"
    assert manifest["video_timeline"] == "per_step" and manifest["video_fps"] == 10
    assert manifest["overlay_hfov_deg"] == 90.0 and manifest["overlay_cam_height"] == 0.5
    assert manifest["instruction"] == "go to the sofa" and manifest["frame_size"] == [64, 36]

    second = load_records(episodes[1])
    assert [r["episode_id"] for r in second] == ["episode_001"] * 3
    assert second[0]["instruction"] == "turn around twice"


def test_record_dir_and_save_video_together(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    from lightnav.viz.render_episode import find_episode_dirs

    cfg = _cfg(tmp_path, episodes=1, save_video=True, record_dir=str(tmp_path / "rec"))
    (result,) = Harness(FrameEnv(steps_to_terminate=2)).run(cfg)

    assert result["video"] == "videos/episode_000.mp4"
    assert len(find_episode_dirs([tmp_path / "rec"])) == 1


def test_a_video_that_fails_mid_episode_is_discarded_without_a_partial_file(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    from lightnav.habitat import runner as runner_mod

    real_write = runner_mod._EpisodeVideo.write
    calls = {"n": 0}

    def flaky_write(self, frame):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("encoder died")
        real_write(self, frame)

    monkeypatch.setattr(runner_mod._EpisodeVideo, "write", flaky_write)
    cfg = _cfg(tmp_path, episodes=2, save_video=True)
    h = Harness(FrameEnv(steps_to_terminate=3))

    results = h.run(cfg)

    videos = Path(cfg.output_dir) / "videos"
    # Episode 0 lost its encoder on the second frame: nothing renamed into place, no leftovers.
    assert "video" not in results[0]
    assert not (videos / "episode_000.mp4").exists()
    assert not list(videos.glob("*.partial*"))
    # The evaluation itself went on unharmed, and the next episode's video is complete.
    assert [r["steps"] for r in results] == [3, 3]
    assert results[1]["video"] == "videos/episode_001.mp4"
    assert len(_video_frames(videos / "episode_001.mp4")) == 4
    assert calls["n"] == 2 + 4  # 2 attempts on episode 0 (then dropped), 4 frames on episode 1


# -- forced STOP at the step budget (simulation only) --------------------------------------


class StoppablePolicy(FakePolicy):
    STOP = {"action": "velocity_control", "action_args": {"linear_velocity": -1.0, "angular_velocity": 0.0}}

    def stop_action(self) -> dict:
        return dict(self.STOP)


def test_oracle_stop_uses_privileged_distance_and_records_its_source(tmp_path):
    env = FakeEnv(steps_to_terminate=4)
    policy = StoppablePolicy()
    cfg = _cfg(tmp_path, episodes=1, max_steps=10, oracle_stop_distance=2.5)

    (result,) = run_habitat_eval(
        cfg,
        env_factory=lambda c: env,
        policy_factory=lambda c, e, b, i: policy,
        engine_factory=lambda c: (
            object(),
            SimpleNamespace(action_method="flat", num_history_frames=4),
        ),
    )

    assert policy.acts == 3
    assert env.actions[-1] == StoppablePolicy.STOP
    assert result["oracle_stop"] is True
    assert result["stop_source"] == "oracle_stop"
    assert result["policy_event_counts"] == {"oracle_stop": 1}


def _run_budget(tmp_path, policy, **cfg_overrides):
    env = FakeEnv(steps_to_terminate=10_000)  # the policy never stops on its own
    cfg = _cfg(tmp_path, episodes=1, max_steps=3, **cfg_overrides)
    results = run_habitat_eval(
        cfg,
        env_factory=lambda c: env,
        policy_factory=lambda c, e, b, i: policy,
        engine_factory=lambda c: (object(), SimpleNamespace(action_method="flat", num_history_frames=4)),
    )
    return env, results


def test_last_budgeted_action_is_an_explicit_stop(tmp_path):
    policy = StoppablePolicy()
    env, results = _run_budget(tmp_path, policy)
    assert len(env.actions) == 3
    assert env.actions[-1] == StoppablePolicy.STOP
    assert env.actions[0]["action_args"]["linear_velocity"] == 0.1  # model actions before it
    assert policy.acts == 2  # the model is not queried for the forced step
    assert results[0]["forced_stop"] is True and results[0]["steps"] == 3


def test_force_stop_can_be_disabled(tmp_path):
    policy = StoppablePolicy()
    env, results = _run_budget(tmp_path, policy, force_stop_at_max_steps=False)
    assert len(env.actions) == 3 and env.actions[-1] != StoppablePolicy.STOP
    assert policy.acts == 3 and results[0]["forced_stop"] is False


def test_force_stop_needs_a_policy_that_can_stop(tmp_path):
    policy = FakePolicy()  # no stop_action(): nothing is forced
    env, results = _run_budget(tmp_path, policy)
    assert policy.acts == 3 and results[0]["forced_stop"] is False


def test_force_stop_does_not_fire_when_the_policy_stops_earlier(tmp_path):
    env = FakeEnv(steps_to_terminate=2)
    policy = StoppablePolicy()
    results = run_habitat_eval(
        _cfg(tmp_path, episodes=1, max_steps=3),
        env_factory=lambda c: env,
        policy_factory=lambda c, e, b, i: policy,
        engine_factory=lambda c: (object(), SimpleNamespace(action_method="flat", num_history_frames=4)),
    )
    assert len(env.actions) == 2 and results[0]["forced_stop"] is False
