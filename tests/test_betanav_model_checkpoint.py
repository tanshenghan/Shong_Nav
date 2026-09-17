"""Background model export must never serialize live weights or drop a permanent snapshot."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightnav.training.checkpoint import TrainState, resolve_resume
from lightnav.training.config import TrainConfig
from lightnav.training.model_checkpoint import AsyncModelCheckpointer


@pytest.fixture
def model():
    result = torch.nn.Linear(4, 3)
    result.weight_alias = result.weight
    result.config = TrainState("test-config")
    return result


@pytest.fixture
def cfg(tmp_path):
    return TrainConfig(dataset_root="data", model_path="model", data_manifest="manifest.json",
                       output_dir=str(tmp_path / "run"), checkpoint_mode="model_only", save_best=False)


@pytest.fixture
def accel():
    return SimpleNamespace(num_processes=1, process_index=0, is_main_process=True, device=torch.device("cpu"),
                           unwrap_model=lambda m: m)


def test_snapshot_overlaps_updates_and_is_immutable_and_permanent(tmp_path, model, cfg, accel, monkeypatch):
    from lightnav.training import model_checkpoint as module

    entered, release = threading.Event(), threading.Event()
    original = module._write_snapshot

    def slow_write(*args):
        assert all(t.device.type == "cpu" for t in args[2].values())
        entered.set()
        assert release.wait(10), "test failed to release disk writer"
        return original(*args)

    monkeypatch.setattr(module, "_write_snapshot", slow_write)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    writer = AsyncModelCheckpointer(accel, cfg, {})
    try:
        path = Path(writer.submit(model, TrainState("async-test", step=3000)))
        assert entered.wait(5)
        assert writer.poll() is None
        assert not path.exists()
        # Stand in for subsequent updates while the disk write is deliberately blocked.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(2.)
        release.set()
        writer.poll(wait=True)
        actual = torch.load(path / "model.pt", weights_only=True)
        for k, v in expected.items():
            torch.testing.assert_close(actual[k], v, atol=0, rtol=0)
        assert {p.name for p in path.iterdir()} == {"model.pt", "metadata.json", "timing.json", "COMPLETE"}
        assert json.loads((path / "metadata.json").read_text())["training_state_complete"] is False
        cfg.keep_last = 1
        writer.submit(model, TrainState("async-test", step=6000))
        writer.poll(wait=True)
        assert path.is_dir()  # permanent even when keep_last=1
        assert json.loads((Path(cfg.output_dir) / "latest_model.json").read_text())["step"] == 6000
        with pytest.raises(RuntimeError, match="FileExistsError"):
            writer.submit(model, TrainState("async-test", step=6000))
        with pytest.raises(ValueError, match="no Adam/RNG"):
            resolve_resume(Path(cfg.output_dir), "auto")
    finally:
        release.set()
        writer.close()


def test_background_failure_is_reported_without_committing(model, cfg, accel, monkeypatch):
    from lightnav.training import model_checkpoint as module

    def failed_write(*_args):
        raise OSError("test disk is full")

    monkeypatch.setattr(module, "_write_snapshot", failed_write)
    writer = AsyncModelCheckpointer(accel, cfg, {})
    try:
        writer.submit(model, TrainState("failure", step=3000))
        with pytest.raises(RuntimeError, match="test disk is full"):
            writer.poll(wait=True)
        root = Path(cfg.output_dir)
        assert not (root / "latest_model.json").exists()
        assert not list((root / "model_checkpoints").glob("step-*/COMPLETE"))
    finally:
        writer.close()
