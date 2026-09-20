from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lightnav.habitat.beta_policy import BetaFMRuntime, BetaFMVLNCEPolicy, _prompt_only_inputs
from lightnav.inference.beta_model import resolve_beta_checkpoint


def _checkpoint(path: Path, step: int, fmt: str) -> None:
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"weights")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "format": fmt,
                "step": step,
                "beta_config": {"action_scale": [1.0, 1.0, 1.0]},
            }
        )
    )
    (path / "COMPLETE").write_text(f"{fmt}\n")


def test_resolve_beta_checkpoint_uses_newest_committed_pointer(tmp_path):
    full = tmp_path / "checkpoints" / "step-00004000"
    model_only = tmp_path / "model_checkpoints" / "step-00006000"
    _checkpoint(full, 4000, "betanav_checkpoint_v2")
    _checkpoint(model_only, 6000, "betanav_model_v1")
    (tmp_path / "latest.json").write_text(json.dumps({"checkpoint": full.name, "step": 4000}))
    (tmp_path / "latest_model.json").write_text(
        json.dumps({"checkpoint": model_only.name, "step": 6000})
    )

    path, metadata = resolve_beta_checkpoint(tmp_path)

    assert path == model_only.resolve()
    assert metadata["step"] == 6000


def test_prompt_inputs_trim_placeholder_answer_without_batching_pixels():
    inputs = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.ones(4, dtype=torch.bool),
        "labels": torch.tensor([-100, -100, 3, 4]),
        "pixel_values_videos": torch.randn(5, 8),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
        "ignored_processor_field": torch.tensor(1),
    }

    result = _prompt_only_inputs(inputs)

    assert result["input_ids"].tolist() == [[1, 2]]
    assert result["attention_mask"].shape == (1, 2)
    assert result["pixel_values_videos"].shape == (5, 8)
    assert "labels" not in result and "ignored_processor_field" not in result


class _FakeBetaModel:
    def __init__(self):
        self.expert = SimpleNamespace(step_embedding=torch.zeros(1))
        self.prefix = None

    def predict(self, prefix, **kwargs):
        self.prefix = prefix
        return SimpleNamespace(
            texts=["<act_l0_1><act_l1_2><act_l2_3>"],
            stop=torch.tensor([False]),
            rvq_waypoints=torch.zeros(1, 10, 3),
            se2_deltas=torch.tensor([[[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
            waypoints=torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]]),
        )


class _Processor:
    def process_sample(self, sample, **kwargs):
        return {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.ones(3, dtype=torch.bool),
            "labels": torch.full((3,), -100),
        }


def test_beta_policy_drives_from_fm_waypoint():
    model = _FakeBetaModel()
    bundle = SimpleNamespace(
        tokenizer=object(),
        data_processor=_Processor(),
        slowfast_tiers=None,
        pool_enable=False,
        pool_spatial=1,
        pool_mode="avg",
        video_fps=4,
        action_method="rvq",
        video_size=(4, 4),
        processor=SimpleNamespace(video_processor=None),
    )
    runtime = BetaFMRuntime(model, bundle, Path("/checkpoint"), 6000)
    rvq = SimpleNamespace(horizon=10)
    policy = BetaFMVLNCEPolicy(
        runtime,
        dt=1.0,
        lin_vel_range=(0.0, 0.25),
        ang_vel_range=(-30.0, 30.0),
        num_history_frames=4,
        rvq_bundle=rvq,
    )
    policy.reset({"instruction": {"text": "go"}})
    policy.agent.observe = lambda _rgb: None
    policy.agent._get_video_tensor = lambda: torch.zeros(1, 3, 4, 4)

    action = policy.act(
        {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)},
        {"scene_id": "scene", "episode_id": "episode"},
    )

    assert action["action"] == "velocity_control"
    assert np.isclose(action["action_args"]["linear_velocity"], -0.2)
    assert policy.get_info()["action_source"] == "fm"
    np.testing.assert_allclose(policy.get_info()["predicted_traj"][:, 0], [0.1, 0.2])
    assert model.prefix["input_ids"].shape == (1, 3)


def test_explicit_gate_suppresses_an_implicit_fm_zero_command():
    class Model(_FakeBetaModel):
        def predict(self, prefix, **kwargs):
            result = super().predict(prefix, **kwargs)
            result.explicit_stop = torch.tensor([False])
            result.decode_failed = torch.tensor([False])
            result.waypoints = torch.tensor(
                [[[-0.1, 0.0, 0.0], [0.0, 0.0, 0.1]]], dtype=torch.float32
            )
            return result

    model = Model()
    bundle = SimpleNamespace(
        tokenizer=object(),
        data_processor=_Processor(),
        slowfast_tiers=None,
        pool_enable=False,
        pool_spatial=1,
        pool_mode="avg",
        video_fps=4,
        action_method="rvq",
        video_size=(4, 4),
        processor=SimpleNamespace(video_processor=None),
    )
    runtime = BetaFMRuntime(model, bundle, Path("/checkpoint"), 6000, stop_mode="explicit_gate")
    policy = BetaFMVLNCEPolicy(
        runtime,
        dt=1.0,
        lin_vel_range=(0.0, 0.25),
        ang_vel_range=(-30.0, 30.0),
        num_history_frames=4,
        rvq_bundle=SimpleNamespace(horizon=10),
    )
    policy.reset({"instruction": {"text": "go"}})
    policy.agent.observe = lambda _rgb: None
    policy.agent._get_video_tensor = lambda: torch.zeros(1, 3, 4, 4)

    action = policy.act(
        {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)},
        {"scene_id": "scene", "episode_id": "episode"},
    )

    assert action["action_args"]["angular_velocity"] != 0.0
    assert policy.get_info()["stop"] is False
    assert policy.get_info()["stop_suppressed"] is True
    assert policy.get_info()["stop_source"] == "fm_zero_suppressed"
    assert policy.get_info()["action_waypoint_index"] == 1


def test_token_factory_loads_beta_checkpoint_instead_of_base_engine(monkeypatch):
    from lightnav.habitat.runner import HabitatEvalConfig, _default_engine_factory

    loaded_model = object()
    loaded_bundle = object()
    calls = []

    def load(config, checkpoint, **kwargs):
        calls.append((checkpoint, kwargs["task_type"]))
        return loaded_model, loaded_bundle, Path("/trained/step-29589"), {"step": 29589}

    def forbid_base(*args, **kwargs):
        raise AssertionError("must load trained BetaNav weights")

    monkeypatch.setattr("lightnav.inference.beta_model.load_beta_checkpoint", load)
    monkeypatch.setattr("lightnav.inference.build_engine", forbid_base)
    runtime, bundle = _default_engine_factory(HabitatEvalConfig(
        model_path="/base", backend="hf", action_output="token", beta_checkpoint="/trained"
    ))
    assert calls == [("/trained", "vlnce")]
    assert runtime.beta_model is loaded_model
    assert bundle is loaded_bundle
    assert runtime.checkpoint_step == 29589
    assert runtime.action_output == "token"


def test_fm_only_factory_and_policy_never_decode_tokens(monkeypatch):
    from lightnav.habitat.runner import HabitatEvalConfig, build_velocity_policy

    class Model(_FakeBetaModel):
        config = SimpleNamespace(training_mode="fm_only")

        def predict(self, *args, **kwargs):
            raise AssertionError("FM-only must never generate/decode tokens")

        def encode_context(self, inputs):
            assert inputs["input_ids"].shape == (1, 3)
            return "context"

        def sample_actions(self, context, **kwargs):
            assert context == "context"
            assert kwargs["generator"] is not None
            return torch.zeros(1, 30, 3), torch.zeros(1, 30, 3)

    def forbid_decoder(*args, **kwargs):
        raise AssertionError("FM-only must not load an RVQ decoder")

    monkeypatch.setattr("lightnav.habitat.runner.resolve_action_decoder", forbid_decoder)
    monkeypatch.setattr("lightnav.inference.samples.build_continuous_traj_sample", lambda *a: {})
    bundle = SimpleNamespace(
        data_processor=_Processor(), num_history_frames=4, slowfast_tiers=None,
        video_size=(4, 4), processor=SimpleNamespace(video_processor=None),
    )
    runtime = BetaFMRuntime(Model(), bundle, Path("/checkpoint"), 3699)
    cfg = HabitatEvalConfig(model_path="/base", action_output="fm", beta_checkpoint="/checkpoint")
    info = dict(habitat_time_step=0.1, lin_vel_range=(0, 2.5), ang_vel_range=(-300, 300),
                scene_id="scene", episode_id="episode")
    policy = build_velocity_policy(cfg, runtime, bundle, info)
    policy.reset({"instruction": "go"})
    policy.agent.observe = lambda rgb: None
    policy.agent._get_video_tensor = lambda: torch.zeros(1, 3, 4, 4)
    action = policy.act({"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}, info)
    assert not policy._command_would_stop(action)
    assert policy._command_would_stop(policy.stop_action())
    assert policy.get_info()["action_source"] == "fm_only"
    assert policy.get_info()["stop"] is False
    assert policy.get_info()["stop_suppressed"] is True
    assert "rvq_codes" not in policy.get_info()
    assert "raw_text" not in policy.get_info()


def test_frozen_rvq_stop_matches_baseline_and_uses_rvq_prompt(monkeypatch):
    from contextlib import nullcontext

    from lightnav.habitat.beta_policy import BetaFMOnlyVLNCEPolicy

    prompts = []
    monkeypatch.setattr("lightnav.habitat.beta_policy.build_vln_traj_sample",
                        lambda *args: prompts.append("rvq") or {})
    policy = object.__new__(BetaFMOnlyVLNCEPolicy)
    policy.runtime = SimpleNamespace(max_new_tokens=16)
    text = ["<act_l0_0><act_l1_0><act_l2_0>"]
    policy.bundle = SimpleNamespace(
        data_processor=_Processor(), tokenizer=SimpleNamespace(
            pad_token_id=0, eos_token_id=None, decode=lambda *a, **kw: text[0],
        ),
    )
    def generate(**kwargs):
        assert kwargs["do_sample"] is False
        return torch.tensor([[1, 2, 3, 4]])

    policy.model = SimpleNamespace(_hf_inputs=lambda inputs: nullcontext(inputs),
                                   vlm=SimpleNamespace(generate=generate))
    decoded = [np.zeros((10, 3), dtype=np.float32)]
    policy.agent = SimpleNamespace(
        _get_video_tensor=lambda: torch.zeros(1, 3, 4, 4), instruction="go",
        _history_frame_ids=[0], decode_waypoints=lambda text: (decoded[0], ""),
        rvq=SimpleNamespace(is_stop=lambda codes: codes[0] == 0),
    )
    policy._clear_last()
    assert policy._rvq_requests_stop()
    assert policy._last_stop_source == "explicit_token_stop"
    text[0] = "<act_l0_1><act_l1_0><act_l2_0>"
    policy._clear_last()
    assert policy._rvq_requests_stop()
    assert policy._last_stop_source == "near_zero_rvq_stop"
    decoded[0][:, 0] = 0.25
    policy._clear_last()
    assert not policy._rvq_requests_stop()
    text[0] = "invalid"
    policy._clear_last()
    assert policy._rvq_requests_stop()
    assert policy._decode_failed and policy._last_stop_source == "decode_failure"
    assert prompts == ["rvq"] * 4
