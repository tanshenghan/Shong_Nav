"""Finding the action decoder a checkpoint ships or references, without CLI flags."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lightnav.eval_config import find_eval_config, resolve_asset_path
from lightnav.serving.ws_server import resolve_decoder_args
from lightnav.tracking import build_tracking_agent, resolve_action_decoder_from_config


def _eval_config(model_dir: Path, tasks: dict) -> None:
    (model_dir / "eval_config.json").write_text(
        json.dumps({"version": 1, "common": {"video_size": [224, 320]}, "tasks": tasks})
    )


def _centroids(vocab_dir: Path, K: int = 8, H: int = 10) -> Path:
    vocab_dir.mkdir(parents=True, exist_ok=True)
    path = vocab_dir / f"centroids_whole_chunk_K{K}_h{H}.npy"
    arr = np.zeros((K, H, 3), dtype=np.float32)
    arr[1:, :, 0] = 0.1
    np.save(path, arr)
    return path


@pytest.fixture
def shipped_rvq_checkpoint(tmp_path, rvq_bundle_writer):
    """A checkpoint directory that ships two RVQ bundles referenced by relative paths."""
    model = tmp_path / "hf_ckpt"
    model.mkdir()
    rvq_bundle_writer(model / "action_tokenizer" / "vlnce", horizon=10)
    rvq_bundle_writer(model / "action_tokenizer" / "trackvla", horizon=10)
    _eval_config(
        model,
        {
            "vlnce": {
                "predict_horizon": 10,
                "action_tokenizer": {"method": "rvq", "bundle_path": "action_tokenizer/vlnce"},
            },
            "trackvla": {
                "predict_horizon": 10,
                "action_tokenizer": {"method": "rvq", "bundle_path": "action_tokenizer/trackvla"},
            },
        },
    )
    return model


# ---- resolve_asset_path -----------------------------------------------------------------


def test_absolute_asset_paths_are_returned_unchanged(tmp_path):
    assert resolve_asset_path(str(tmp_path), "/abs/somewhere") == Path("/abs/somewhere")


def test_relative_asset_paths_resolve_against_the_eval_config_directory(tmp_path):
    root = tmp_path / "run"
    ckpt = root / "checkpoints" / "global_step_5" / "hf_ckpt"
    ckpt.mkdir(parents=True)
    _eval_config(root, {})
    (root / "shared_vocab").mkdir()
    assert find_eval_config(str(ckpt)) == root / "eval_config.json"
    assert resolve_asset_path(str(ckpt), "shared_vocab") == root / "shared_vocab"


def test_relative_asset_paths_fall_back_to_the_model_directory(tmp_path):
    ckpt = tmp_path / "hf_ckpt"
    (ckpt / "action_tokenizer").mkdir(parents=True)
    _eval_config(tmp_path, {})  # config one level up, asset next to the weights
    assert resolve_asset_path(str(ckpt), "action_tokenizer") == ckpt / "action_tokenizer"


# ---- resolve_action_decoder_from_config ---------------------------------------------------


def test_snapshot_prefers_the_requested_task(shipped_rvq_checkpoint):
    model = shipped_rvq_checkpoint
    nav = resolve_action_decoder_from_config(model, "vlnce")
    track = resolve_action_decoder_from_config(model, "trackvla")
    assert nav == {"method": "rvq", "bundle_path": model / "action_tokenizer" / "vlnce", "horizon": 10}
    assert track["bundle_path"] == model / "action_tokenizer" / "trackvla"


def test_snapshot_falls_back_to_another_task_when_the_requested_one_is_missing(
    shipped_rvq_checkpoint,
):
    model = shipped_rvq_checkpoint
    cfg = json.loads((model / "eval_config.json").read_text())
    del cfg["tasks"]["trackvla"]
    (model / "eval_config.json").write_text(json.dumps(cfg))
    assert resolve_action_decoder_from_config(model, "trackvla")["bundle_path"].name == "vlnce"


def test_flat_snapshot_with_relative_vocab_dir(tmp_path):
    model = tmp_path / "hf_ckpt"
    model.mkdir()
    _centroids(model / "traj_vocab", K=8, H=10)
    _eval_config(
        model,
        {"trackvla": {"predict_horizon": 10, "traj_vocab_path": "traj_vocab", "traj_vocab_K": 8}},
    )
    assert resolve_action_decoder_from_config(model, "trackvla") == {
        "method": "flat",
        "traj_vocab_path": model / "traj_vocab",
        "K": 8,
        "horizon": 10,
    }


def test_sibling_directories_without_eval_config(tmp_path, rvq_bundle_writer):
    model = tmp_path / "hf_ckpt"
    rvq_bundle_writer(model / "action_tokenizer", horizon=7)
    resolved = resolve_action_decoder_from_config(model, "vlnce")
    assert resolved["method"] == "rvq" and resolved["bundle_path"] == model / "action_tokenizer"
    assert resolved["horizon"] == 7  # from the manifest when the config has no horizon


def test_task_specific_sibling_wins_over_the_generic_one(tmp_path, rvq_bundle_writer):
    model = tmp_path / "hf_ckpt"
    rvq_bundle_writer(model / "action_tokenizer", horizon=10)
    rvq_bundle_writer(model / "action_tokenizer" / "trackvla", horizon=10)
    assert resolve_action_decoder_from_config(model, "trackvla")["bundle_path"].name == "trackvla"


def test_dangling_snapshot_path_is_skipped(tmp_path, rvq_bundle_writer):
    model = tmp_path / "hf_ckpt"
    model.mkdir()
    _eval_config(
        model,
        {"vlnce": {"predict_horizon": 10,
                   "action_tokenizer": {"method": "rvq", "bundle_path": "/nonexistent/bundle"}}},
    )
    assert resolve_action_decoder_from_config(model, "vlnce") is None
    rvq_bundle_writer(model / "action_tokenizer", horizon=10)
    assert resolve_action_decoder_from_config(model, "vlnce")["bundle_path"] == model / "action_tokenizer"


def test_nothing_to_resolve_returns_none(tmp_path):
    (tmp_path / "hf_ckpt").mkdir()
    assert resolve_action_decoder_from_config(tmp_path / "hf_ckpt", "trackvla") is None


# ---- server argparse post-processing ------------------------------------------------------


def _args(**kw):
    base = dict(model_path="", traj_vocab_path=None, action_tokenizer_bundle=None, K=256,
                horizon=10, task="tracking")
    base.update(kw)
    return Namespace(**base)


def test_server_fills_the_decoder_from_the_checkpoint(shipped_rvq_checkpoint):
    args = resolve_decoder_args(_args(model_path=str(shipped_rvq_checkpoint), task="vln"))
    assert args.action_tokenizer_bundle == str(shipped_rvq_checkpoint / "action_tokenizer" / "vlnce")
    assert args.traj_vocab_path is None and args.horizon == 10
    args = resolve_decoder_args(_args(model_path=str(shipped_rvq_checkpoint), task="tracking"))
    assert args.action_tokenizer_bundle.endswith("trackvla")


def test_server_keeps_explicit_flags(shipped_rvq_checkpoint):
    args = resolve_decoder_args(_args(model_path=str(shipped_rvq_checkpoint), traj_vocab_path="/v", K=5))
    assert args.traj_vocab_path == "/v" and args.action_tokenizer_bundle is None and args.K == 5


def test_server_rejects_both_flags_and_unresolvable_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="only one"):
        resolve_decoder_args(_args(traj_vocab_path="/v", action_tokenizer_bundle="/b"))
    (tmp_path / "hf_ckpt").mkdir()
    with pytest.raises(ValueError, match="no action decoder"):
        resolve_decoder_args(_args(model_path=str(tmp_path / "hf_ckpt")))


# ---- build_tracking_agent without decoder arguments ----------------------------------------


def test_build_tracking_agent_resolves_the_shipped_bundle(shipped_rvq_checkpoint, monkeypatch):
    fake_engine = SimpleNamespace(
        bundle=SimpleNamespace(slowfast_tiers=None, video_size=(224, 320), num_history_frames=8),
        new_vit_cache=lambda: None,
        reset_episode_state=lambda: None,
    )
    seen = {}

    def fake_build_engine(cfg, task_type, max_new_tokens=None):
        seen["task_type"], seen["cfg"] = task_type, cfg
        return fake_engine, fake_engine.bundle

    monkeypatch.setattr("lightnav.tracking.build_engine", fake_build_engine)
    agent = build_tracking_agent(str(shipped_rvq_checkpoint), backend="hf", task_key="vlnce")
    assert agent.rvq is not None and agent.H == 10 and agent.centroids is None
    # The engine is built from the SAME eval_config task entry the decoder came from.
    assert seen["task_type"] == "vlnce"
    assert seen["cfg"].max_new_tokens >= 1 + len(agent.rvq.levels) + 1

    with pytest.raises(ValueError, match="only one"):
        build_tracking_agent("/x", traj_vocab_path="/v", action_tokenizer_bundle="/b")


def test_engine_task_falls_back_to_the_only_task_the_checkpoint_has(
    tmp_path, rvq_bundle_writer, monkeypatch
):
    """A tracking-only checkpoint asked for `vlnce` must still be built from `trackvla`,
    not silently dropped onto INFERENCE_FALLBACK_DEFAULTS."""
    model = tmp_path / "hf_ckpt"
    model.mkdir()
    rvq_bundle_writer(model / "action_tokenizer", horizon=10)
    _eval_config(
        model,
        {
            "trackvla": {
                "predict_horizon": 10,
                "action_tokenizer": {"method": "rvq", "bundle_path": "action_tokenizer"},
            }
        },
    )
    fake_engine = SimpleNamespace(
        bundle=SimpleNamespace(slowfast_tiers=None, video_size=(224, 320), num_history_frames=8),
        new_vit_cache=lambda: None,
        reset_episode_state=lambda: None,
    )
    seen = {}

    def fake_build_engine(cfg, task_type, max_new_tokens=None):
        seen["task_type"] = task_type
        return fake_engine, fake_engine.bundle

    monkeypatch.setattr("lightnav.tracking.build_engine", fake_build_engine)
    agent = build_tracking_agent(str(model), backend="hf", task_key="vlnce")
    assert seen["task_type"] == "trackvla"
    # None (the default) takes the history window from the checkpoint, not a hardcoded 64.
    assert agent.num_history_frames == 8


# ---- eval_config guards ----------------------------------------------------------------


def test_native_resolution_checkpoints_are_refused_loudly(tmp_path):
    from lightnav.inference.model import _resolve_processing_params

    model = tmp_path / "hf_ckpt"
    model.mkdir()
    (model / "eval_config.json").write_text(
        json.dumps({"version": 1, "common": {"video_size": [256, 448], "native_resolution": True},
                    "tasks": {"trackvla": {"num_history_frames": 8}}})
    )
    with pytest.raises(NotImplementedError, match="native_resolution"):
        _resolve_processing_params(str(model), "trackvla")
    (model / "eval_config.json").write_text(
        json.dumps({"version": 1, "common": {"video_size": [256, 448], "native_resolution": False},
                    "tasks": {"trackvla": {"num_history_frames": 8}}})
    )
    assert _resolve_processing_params(str(model), "trackvla")["video_size"] == (256, 448)
