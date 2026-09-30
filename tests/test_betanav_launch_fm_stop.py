"""Platform launcher preparation and dispatch, without weights or CUDA."""

import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from lightnav.training import launch_fm_stop as launch
from lightnav.training.config import TrainConfig


@pytest.fixture
def setup_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "REPOSITORY", tmp_path)
    config = TrainConfig(dataset_root=str(tmp_path / "dataset"), model_path=str(tmp_path / "Light"),
                         data_manifest=str(tmp_path / "stop.json"), output_dir=str(tmp_path / "output"),
                         training_mode="fm_stop", stop_head_type="last_prefix_mlp_v1", num_epochs=3,
                         eval_every=0, save_best=False)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)))
    manifest = {"split_by": "all", "train_episodes": [0], "val_episodes": [], "val_frames": 0}
    calls = {"make": [], "read": [], "exec": []}

    def make(*args, **kwargs):
        calls["make"].append((args, kwargs))
        return manifest

    def read(cfg):
        calls["read"].append(cfg)
        return json.loads(Path(cfg.data_manifest).read_text())

    monkeypatch.setattr(launch, "make_manifest", make)
    monkeypatch.setattr(launch, "read_data_manifest", read)
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)
    monkeypatch.setattr(launch.os, "execv", lambda *args: calls["exec"].append(args))
    monkeypatch.setattr(launch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launch.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(launch.torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(launch.torch.cuda, "is_bf16_supported", lambda: True)
    return config, config_path, manifest, calls


@pytest.mark.parametrize(("gpus", "accumulation"), [(4, 16), (8, 8)])
def test_prepare_only_audits_and_adapts_batch(setup_launch, monkeypatch, gpus, accumulation):
    config, path, _, calls = setup_launch
    monkeypatch.setattr(launch.torch.cuda, "is_available", lambda: pytest.fail("prepare touched CUDA"))
    monkeypatch.setattr(launch.torch.cuda, "device_count", lambda: pytest.fail("prepare touched CUDA"))
    result = launch.main(["--config", str(path), "--prepare-only", "--num-gpus", str(gpus)])
    assert result["initialization"] == "fresh"
    assert result["gradient_accumulation_steps"] == accumulation
    assert result["global_batch"] == 512
    assert not calls["exec"]
    assert len(calls["make"]) == len(calls["read"]) == 1
    assert calls["make"][0][1]["with_stop_labels"] is True
    assert calls["make"][0][1]["val_fraction"] == 0
    assert not Path(config.output_dir).exists()
    assert "--warm-start" not in result["command"] and "--resume" not in result["command"]
    effective = yaml.safe_load(Path(result["configuration"]).read_text())
    assert effective["gradient_accumulation_steps"] == accumulation
    assert yaml.safe_load(path.read_text())["gradient_accumulation_steps"] == 1


def test_prepare_requires_gpu_count(setup_launch):
    _, path, _, calls = setup_launch
    with pytest.raises(SystemExit):
        launch.main(["--config", str(path), "--prepare-only"])
    assert not calls["make"] and not calls["exec"]


@pytest.mark.parametrize("changes", [
    {"training_mode": "fm_only", "stop_head_type": "none"}, {"cpu": True}, {"tiny_model": True},
    {"num_epochs": 2}, {"eval_every": 100}, {"save_best": True},
])
def test_rejects_wrong_experiment(setup_launch, changes):
    _, path, _, calls = setup_launch
    data = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**data, **changes}))
    with pytest.raises(ValueError):
        launch.main(["--config", str(path), "--prepare-only", "--num-gpus", "4"])
    assert not calls["make"] and not calls["exec"]


@pytest.mark.parametrize("extra", [["--num-gpus", "0"], ["--num-gpus", "3"],
                                  ["--num-gpus", "4", "--global-batch", "0"],
                                  ["--num-gpus", "4", "--batch-size", "0"]])
def test_rejects_invalid_batch(setup_launch, extra):
    _, path, _, calls = setup_launch
    with pytest.raises(ValueError):
        launch.main(["--config", str(path), "--prepare-only", *extra])
    assert not calls["make"] and not calls["exec"]


@pytest.mark.parametrize("problem", ["unavailable", "few", "bf16"])
def test_cuda_preflight_precedes_data_preparation(setup_launch, monkeypatch, problem):
    _, path, _, calls = setup_launch
    if problem == "unavailable":
        monkeypatch.setattr(launch.torch.cuda, "is_available", lambda: False)
    elif problem == "few":
        monkeypatch.setattr(launch.torch.cuda, "device_count", lambda: 2)
    else:
        monkeypatch.setattr(launch.torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises((ValueError, RuntimeError)):
        launch.main(["--config", str(path), "--num-gpus", "4"])
    assert not calls["make"] and not calls["exec"]


def test_existing_manifest_is_verified_without_overwriting(setup_launch):
    config, path, manifest, calls = setup_launch
    original = json.dumps({**manifest, "existing": True}, indent=1)
    Path(config.data_manifest).write_text(original)
    launch.main(["--config", str(path), "--prepare-only", "--num-gpus", "8"])
    assert not calls["make"] and len(calls["read"]) == 1
    assert Path(config.data_manifest).read_text() == original


def test_manifest_publication_preserves_concurrent_writer(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    link = launch.os.link

    def race(source, destination):
        path.write_text('{"winner": true}\n')
        link(source, destination)

    monkeypatch.setattr(launch.os, "link", race)
    launch.publish_manifest(path, {"winner": False})
    assert json.loads(path.read_text()) == {"winner": True}
    assert list(tmp_path.iterdir()) == [path]


def test_launch_inherits_platform_mask_and_dispatches_foreground(setup_launch, monkeypatch):
    _, path, _, calls = setup_launch
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-platform-0,GPU-platform-1,GPU-platform-2,GPU-platform-3")
    monkeypatch.setattr(launch.torch.cuda, "device_count", lambda: 4)
    launch.main(["--config", str(path)])
    executable, command = calls["exec"][0]
    assert executable == sys.executable and command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=4" in command and "--standalone" in command
    effective = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert effective["gradient_accumulation_steps"] == 16
    assert launch.os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-platform-0,")


def test_fresh_nonempty_output_rejected_before_manifest(setup_launch):
    config, path, _, calls = setup_launch
    output = Path(config.output_dir)
    output.mkdir()
    (output / "run.json").write_text("{}")
    with pytest.raises(FileExistsError, match="fresh output_dir"):
        launch.main(["--config", str(path), "--num-gpus", "4"])
    assert not calls["make"] and not calls["exec"]


def test_explicit_resume_preserves_manifest_and_passes_contract_to_trainer(setup_launch):
    config, path, manifest, calls = setup_launch
    Path(config.output_dir).mkdir()
    (Path(config.output_dir) / "run.json").write_text("{}")
    Path(config.data_manifest).write_text(json.dumps(manifest))
    launch.main(["--config", str(path), "--resume"])
    assert not calls["make"]
    assert calls["exec"][0][1][-2:] == ["--resume", "auto"]


def test_resume_never_regenerates_missing_manifest(setup_launch):
    _, path, _, calls = setup_launch
    with pytest.raises(FileNotFoundError, match="original STOP manifest"):
        launch.main(["--config", str(path), "--resume", "--prepare-only", "--num-gpus", "8"])
    assert not calls["make"] and not calls["exec"]


def test_scene_split_is_rejected(setup_launch):
    config, path, manifest, calls = setup_launch
    Path(config.data_manifest).write_text(json.dumps({**manifest, "split_by": "scene", "val_episodes": [1]}))
    with pytest.raises(ValueError, match="all episodes"):
        launch.main(["--config", str(path), "--prepare-only", "--num-gpus", "8"])
    assert not calls["exec"]
