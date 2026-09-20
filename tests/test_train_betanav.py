"""Training infrastructure regression tests: masking, accumulation, resume and skipped branches."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("accelerate")

from test_beta_model import batch as beta_batch_fixture  # noqa: E402,F401
from test_beta_model import model as beta_model_fixture  # noqa: E402,F401

from lightnav.training.checkpoint import (  # noqa: E402
    TrainState,
    load_checkpoint,
    resolve_resume,
    save_checkpoint,
)
from lightnav.training.config import (  # noqa: E402
    TrainConfig,
    learning_rate_factor,
    load_config,
    resume_contracts_match,
)
from lightnav.training.runtime import (  # noqa: E402
    backward_sums,
    clip_and_step,
    make_optimizers,
    normalize_gradients,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def model(request):
    return request.getfixturevalue("beta_model_fixture")


@pytest.fixture
def batch(request):
    return request.getfixturevalue("beta_batch_fixture")


@pytest.fixture
def cfg(tmp_path):
    return TrainConfig(dataset_root="data", model_path="model", data_manifest="manifest.json",
                       output_dir=str(tmp_path / "run"), optimizer="adamw", max_steps=4, warmup_steps=1)


@pytest.fixture
def accel():
    return SimpleNamespace(num_processes=1, process_index=0, is_main_process=True, device=torch.device("cpu"),
                           unwrap_model=lambda m: m, backward=lambda loss: loss.backward())


@pytest.mark.parametrize("name,value", [("gradient_accumulation_steps", 0), ("mixed_precision", "fp16"),
                                      ("lr_vlm", float("nan")), ("warmup_steps", 30000),
                                      ("freeze_vision", "false"), ("expert_hidden_size", 511),
                                      ("flow_time_distribution", "unknown"), ("flow_time_alpha", 0.),
                                      ("flow_time_beta", -1.), ("flow_time_alpha", float("nan")),
                                      ("flow_time_beta", float("inf"))])
def test_invalid_training_options_rejected(cfg, name, value):
    setattr(cfg, name, value)
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("horizon", [0, 31, 10.5, True])
def test_invalid_training_horizon_rejected(cfg, horizon):
    with pytest.raises(ValueError, match="fm_horizon"):
        replace(cfg, fm_horizon=horizon).validate()


def test_legacy_horizon_resume_remains_30_and_rejects_10(cfg):
    old = {"config": {k: v for k, v in cfg.resume_contract().items() if k != "fm_horizon"}}
    assert resume_contracts_match(old, {"config": cfg.resume_contract()})
    assert not resume_contracts_match(old, {"config": replace(cfg, fm_horizon=10).resume_contract()})


def test_yaml_is_strict_and_lr_schedule_has_correct_endpoints(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("dataset_root: data\nmodel_path: model\ndata_manifest: manifest\noutput_dir: out\nlearning_ratte: 1e-4\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)
    assert learning_rate_factor(0, 10, 100, .1) == .1
    assert learning_rate_factor(9, 10, 100, .1) == 1.
    assert learning_rate_factor(10, 10, 100, .1) == 1.
    assert learning_rate_factor(99, 10, 100, .1) == .1


def test_fm_only_omits_vlm_adam_and_preserves_frozen_weights_on_resume(model, batch, cfg, accel):
    from lightnav.inference.beta_model import BetaModel

    cfg.training_mode = "fm_only"
    model = BetaModel(model.vlm, replace(model.config, training_mode="fm_only"))
    batch = replace(batch, model_inputs={k: v[:, :3] for k, v in batch.model_inputs.items()},
                    labels=torch.full_like(batch.labels[:, :3], -100))
    opts = make_optimizers(model, cfg, 1)
    assert set(opts) == {"fm"}
    before = copy.deepcopy(model.vlm.state_dict())
    _update(model, batch, cfg, accel, opts)
    saved = Path(save_checkpoint(accel, model, opts, TrainState("fm-only", step=1), cfg,
                                 {"config": cfg.resume_contract()}))
    assert not (saved / "optimizer-vlm.pt").exists()
    expected = copy.deepcopy(model.state_dict())
    _update(model, batch, cfg, accel, opts)
    load_checkpoint(saved, accel, model, opts, {"config": cfg.resume_contract()})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    for key, value in model.vlm.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.vlm.parameters())


def test_new_default_options_allow_legacy_ce_fm_resume_but_reject_mode_change(cfg):
    current = {"config": cfg.resume_contract()}
    legacy = {"config": {k: v for k, v in current["config"].items() if k not in {"training_mode", "num_epochs"}}}
    assert resume_contracts_match(legacy, current)
    cfg.training_mode = "fm_only"
    assert not resume_contracts_match(legacy, {"config": cfg.resume_contract()})


def test_legacy_uniform_time_resume_is_compatible_but_sampling_changes_are_rejected(cfg):
    fields = {"flow_time_distribution", "flow_time_alpha", "flow_time_beta"}
    old = {"config": {k: v for k, v in cfg.resume_contract().items() if k not in fields}}
    assert resume_contracts_match(old, {"config": cfg.resume_contract()})
    for field, value in [("flow_time_distribution", "beta"), ("flow_time_alpha", 2.), ("flow_time_beta", 2.)]:
        changed = replace(cfg, **{field: value})
        assert not resume_contracts_match(old, {"config": changed.resume_contract()})


def test_training_only_does_not_construct_validation_loader(cfg, accel, monkeypatch):
    from lightnav.training import train_betanav as runner

    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return [object()]

    monkeypatch.setattr(runner, "Train_Lerobot_DataLoader", fake_loader)
    cfg.eval_every = 0
    cfg.fm_horizon = 10
    train, val = runner.build_loaders(cfg, {"train_episodes": [1, 2]}, accel)
    assert len(train) == 1 and val is None
    assert len(calls) == 1 and calls[0]["episode_indices"] == [1, 2]
    assert calls[0]["fm_horizon"] == 10


def test_legacy_resume_allows_evaluation_changes_but_not_training_changes(model, cfg, accel):
    from lightnav.training.train_betanav import _initialize_run

    old_contract = {"config": {**cfg.resume_contract(), "eval_every": 500, "eval_max_batches": 64,
                               "eval_sample_batches": 4, "save_best": True}, "data_sha256": "frozen"}
    opts = make_optimizers(model, cfg, 1)
    path = Path(save_checkpoint(accel, model, opts, TrainState("legacy", step=1), cfg, old_contract))
    new_cfg = replace(cfg, eval_every=0, save_best=False)
    contract = {"config": new_cfg.resume_contract(), "data_sha256": "frozen"}
    before = (path / "metadata.json").read_bytes()
    assert _initialize_run(new_cfg, {}, contract, path) == "legacy"
    assert load_checkpoint(path, accel, model, opts, contract).step == 1
    assert (path / "metadata.json").read_bytes() == before
    output = Path(cfg.output_dir)
    assert json.loads((output / "active_config.json").read_text())["eval_every"] == 0
    assert len(list((output / "resumes").glob("*.json"))) == 1
    for field, value in [("lr_vlm", cfg.lr_vlm * 2), ("batch_size", cfg.batch_size + 1),
                         ("lambda_fm", cfg.lambda_fm + 1), ("max_steps", cfg.max_steps + 1)]:
        changed = {**contract, "config": {**contract["config"], field: value}}
        assert not resume_contracts_match(old_contract, changed)
        with pytest.raises(ValueError, match="contract differs"):
            _initialize_run(new_cfg, {}, changed, path)
        with pytest.raises(ValueError, match="contract differs"):
            load_checkpoint(path, accel, model, opts, changed)
    assert not resume_contracts_match(old_contract, {**contract, "data_sha256": "changed"})


def test_expansion_requires_the_full_union_and_unchanged_training_semantics(tmp_path, cfg):
    from lightnav.training.runtime import file_hash
    from lightnav.training.train_betanav import validate_training_data_expansion

    original = dict(split_by="scene", train_episodes=[0], val_episodes=[1], train_scenes=["a"], val_scenes=["b"],
                    train_frames=4, val_frames=5, dataset_root="data", normalization={"action_scale": [1, 2, 3]}, source_sha256={"info": "hash"})
    p = tmp_path / "original.json"
    p.write_text(json.dumps(original))
    old = {"config": {**cfg.resume_contract(), "data_manifest": str(p)}, "data_sha256": file_hash(p)}
    new = {"config": {**cfg.resume_contract(), "data_manifest": "all.json"}, "data_sha256": "all-hash"}
    expanded = {**original, "split_by": "all", "train_episodes": [0, 1], "val_episodes": [],
                "train_scenes": ["a", "b"], "val_scenes": [], "train_frames": 9, "val_frames": 0}
    validate_training_data_expansion(old, new, expanded)
    with pytest.raises(ValueError, match="optimizer or schedule"):
        validate_training_data_expansion(old, {**new, "config": {**new["config"], "lr_fm": .4}}, expanded)
    for changed in ({"train_episodes": [0]}, {"normalization": {"action_scale": [2, 3, 4]}},
                    {"source_sha256": {"info": "changed"}}, {"train_episodes": [0, 1, 1]}):
        with pytest.raises(ValueError, match="full union"):
            validate_training_data_expansion(old, new, {**expanded, **changed})


def test_max_steps_override_is_explicit_and_does_not_relax_other_resume_fields(model, cfg, accel):
    from lightnav.training.train_betanav import _initialize_run

    old_contract = {"config": cfg.resume_contract()}
    path = Path(save_checkpoint(accel, model, make_optimizers(model, cfg, 1), TrainState("schedule", step=1), cfg, old_contract))
    new_cfg = replace(cfg, max_steps=3)
    contract = {"config": new_cfg.resume_contract()}
    with pytest.raises(ValueError, match="contract differs"):
        _initialize_run(new_cfg, {}, contract, path)
    assert _initialize_run(new_cfg, {}, contract, path, override_max_steps=True) == "schedule"
    audit = json.loads(next((Path(cfg.output_dir) / "resumes").glob("*.json")).read_text())
    assert audit["max_steps_override"] and audit["previous_max_steps"] == 4
    assert not audit["data_cursor_reset"]
    changed = {"config": {**contract["config"], "batch_size": cfg.batch_size + 1}}
    with pytest.raises(ValueError, match="contract differs"):
        _initialize_run(new_cfg, {}, changed, path, override_max_steps=True)
    with pytest.raises(ValueError, match="must exceed"):
        _initialize_run(replace(cfg, max_steps=1), {}, contract, path, override_max_steps=True)


def test_per_rank_performance_logs_remain_scalars_for_trackers(tmp_path, cfg, accel):
    from lightnav.training.checkpoint import TrainState
    from lightnav.training.train_betanav import RunLogger

    cfg.output_dir, cfg.tracker = str(tmp_path), "tensorboard"
    captured = []
    accel.init_trackers = lambda *a, **k: None
    accel.log = lambda values, step: captured.append((values, step))
    logger = RunLogger(accel, cfg, TrainState(run_id="test"))
    logger.write({"update_seconds": 2., "update_seconds_by_rank": [1., 2.]}, step=1)
    assert captured == [({"update_seconds": 2., "update_seconds_by_rank/rank_0": 1.,
                         "update_seconds_by_rank/rank_1": 2.}, 1)]
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["update_seconds_by_rank"] == [1., 2.]


def test_accumulation_matches_global_masked_objective_with_empty_fm_tail(model, batch, accel):
    reference = copy.deepcopy(model)
    short = copy.deepcopy(batch)
    short.fm_valid_mask[:] = False  # last accumulation microbatch has NO expert graph
    short.labels[:, 4:] = -100
    noise = torch.randn_like(batch.fm_clean_se2_diff)
    tau = torch.tensor([.2, .7])
    outputs = [reference(b, noise=noise, flow_time=tau) for b in (batch, short)]
    nc, nf = sum(x.ce_token_count for x in outputs), sum(x.fm_step_count for x in outputs)
    expected = sum(x.ce_loss * x.ce_token_count / nc + .3 * x.fm_loss * x.fm_step_count / nf for x in outputs)
    expected.backward()
    for b in (batch, short):
        backward_sums(model(b, noise=noise, flow_time=tau), accel, .3)
    normalize_gradients(model.optimizer_parameters(), [nc, nf], 1)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)


def _update(model, batch, cfg, accel, optimizers):
    out = model(batch, lambda_fm=cfg.lambda_fm)
    backward_sums(out, accel, cfg.lambda_fm)
    counts = [out.ce_token_count, out.fm_step_count]
    normalize_gradients(model.optimizer_parameters(), counts, 1)
    clip_and_step(model.optimizer_parameters(), optimizers, counts, cfg)


def test_all_stop_does_not_decay_or_advance_existing_fm_optimizer(model, batch, cfg, accel):
    opts = make_optimizers(model, cfg, 1)
    _update(model, batch, cfg, accel, opts)
    before = copy.deepcopy(model.expert.state_dict())
    opt_before = copy.deepcopy(opts["fm"].state_dict())
    stop = replace(batch, fm_valid_mask=torch.zeros_like(batch.fm_valid_mask))
    _update(model, stop, cfg, accel, opts)
    for key, tensor in model.expert.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    for key, value in opts["fm"].state_dict()["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, opt_before["state"][key][name], rtol=0, atol=0)


def test_full_checkpoint_restores_parameters_adam_and_rng_exactly(model, batch, cfg, accel):
    opts = make_optimizers(model, cfg, 1)
    _update(model, batch, cfg, accel, opts)
    state = TrainState(run_id="test", step=1, batch_offset=2, samples_seen=4)
    contract = {"dataset": "test-data", "lr": cfg.lr_vlm}
    path = Path(save_checkpoint(accel, model, opts, state, cfg, contract))
    assert resolve_resume(Path(cfg.output_dir), "auto") == path
    expected_random = (random.random(), np.random.rand(), torch.rand(2))
    _update(model, batch, cfg, accel, opts)
    expected = copy.deepcopy(model.state_dict())
    restored = copy.deepcopy(model)
    new_opts = make_optimizers(restored, cfg, 1)
    loaded = load_checkpoint(path, accel, restored, new_opts, contract)
    assert loaded.batch_offset == 2 and loaded.samples_seen == 4
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(2), expected_random[2], atol=0, rtol=0)
    _update(restored, batch, cfg, accel, new_opts)
    for key, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="contract differs"):
        load_checkpoint(path, accel, restored, new_opts, {"different": True})


def test_checkpoint_commit_and_retention_preserve_best(model, cfg, accel):
    cfg.keep_last = 1
    opts = make_optimizers(model, cfg, 1)
    for step in (1, 2, 3):
        save_checkpoint(accel, model, opts, TrainState("retention", step=step, best_validation=1.),
                        cfg, {}, best=step == 1)
    root = Path(cfg.output_dir)
    assert {p.name for p in (root / "checkpoints").iterdir()} == {"step-00000001", "step-00000003"}
    assert json.loads((root / "best.json").read_text())["checkpoint"] == "step-00000001"
    (root / "checkpoints/step-00000003/COMPLETE").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        resolve_resume(root, "auto")
