"""CPU-only preflight checks never launch a simulator or read a mesh body."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightnav.habitat.eval_preflight import (
    build_habitat_env,
    check_habitat_python,
    validate_resources,
)


@pytest.fixture
def resources(tmp_path):
    dataset = tmp_path / "dataset"
    scenes = tmp_path / "scenes"
    scene = scenes / "mp3d/house/house.glb"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"glTF fixture")
    gt = dataset / "val_unseen/val_unseen_gt.json.gz"
    gt.parent.mkdir(parents=True)
    gt.write_bytes(gzip.compress(json.dumps({"1": {"locations": [[0, 1, 2]]}}).encode()))
    episodes = [{"episode_id": 1, "scene_id": "data/scene_datasets/mp3d/house/house.glb"}]
    return dataset, scenes, episodes, gt, scene


def test_environment_reuses_local_gl_and_preserves_allocation(tmp_path):
    gl = tmp_path / ".venv-beta-demo/egl/usr/lib/x86_64-linux-gnu"
    gl.mkdir(parents=True)
    original = {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b", "PYTHONPATH": "/old",
                "LD_LIBRARY_PATH": "/libs", "LD_PRELOAD": "/custom/preload.so"}
    env = build_habitat_env(tmp_path, original)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert env["PYTHONPATH"] == f"{tmp_path}/habitat_server:/old"
    assert env["LD_LIBRARY_PATH"] == f"{gl}:/libs"
    assert env["LD_PRELOAD"] == original["LD_PRELOAD"]
    assert original["PYTHONPATH"] == "/old"
    assert build_habitat_env(tmp_path, env) == env


def test_environment_does_not_invent_missing_libraries(tmp_path):
    env = build_habitat_env(tmp_path, {})
    assert "LD_LIBRARY_PATH" not in env
    assert "LD_PRELOAD" not in env
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_import_check_is_bounded_in_separate_python(monkeypatch):
    def run(command, **kwargs):
        assert command[:2] == ["/env/python", "-c"]
        assert "import habitat, habitat_sim, lightnav_habitat.serve" in command[2]
        assert "Env(" not in command[2]
        assert kwargs["timeout"] == 90
        assert kwargs["env"] == {"CUDA_VISIBLE_DEVICES": "GPU-a"}
        return SimpleNamespace(returncode=0, stderr="", stdout=(
            'noise\nLIGHTNAV_HABITAT_IMPORTS={"imports_ok":true,"python_version":"3.9.23"}\n'
        ))
    monkeypatch.setattr(subprocess, "run", run)
    result = check_habitat_python("/env/python", {"CUDA_VISIBLE_DEVICES": "GPU-a"})
    assert result == {"python": "/env/python", "imports_ok": True, "python_version": "3.9.23"}


def test_import_failure_keeps_missing_library_diagnostic(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout="", stderr="ImportError: libOpenGL.so.0: cannot open shared object file",
    ))
    with pytest.raises(RuntimeError, match="libOpenGL.so.0: cannot open"):
        check_habitat_python("/env/python", {})


def test_import_timeout_is_clear(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", 90)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out after 90 seconds"):
        check_habitat_python("/env/python", {})


def test_resource_fingerprints_do_not_read_mesh_bodies(resources, monkeypatch):
    dataset, scenes, episodes, gt, scene = resources
    original = Path.open

    class UnreadMesh:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            raise AssertionError("Preflight must not read mesh bodies")

    monkeypatch.setattr(Path, "open", lambda path, *args, **kwargs: (
        UnreadMesh() if path == scene else original(path, *args, **kwargs)
    ))
    result = validate_resources(dataset, scenes, episodes)
    assert result["num_episodes"] == result["num_scenes"] == 1
    assert result["ndtw"]["sha256"] == hashlib.sha256(gt.read_bytes()).hexdigest()
    assert result["scenes"] == [{"path": str(scene), "size_bytes": scene.stat().st_size,
                                 "mtime_ns": scene.stat().st_mtime_ns}]


@pytest.mark.parametrize("locations", [None, [], [[1, 2]], [[1, 2, float("nan")]], [[True, 2, 3]], [["1", 2, 3]]])
def test_ndtw_requires_nonempty_finite_3d_paths(resources, locations):
    dataset, scenes, episodes, gt, _ = resources
    gt.write_bytes(gzip.compress(json.dumps({"1": {"locations": locations}}).encode()))
    with pytest.raises(ValueError, match="NDTW"):
        validate_resources(dataset, scenes, episodes)


def test_ndtw_requires_every_episode(resources):
    dataset, scenes, episodes, _, _ = resources
    episodes[0]["episode_id"] = 2
    with pytest.raises(ValueError, match="episode 2"):
        validate_resources(dataset, scenes, episodes)


@pytest.mark.parametrize("scene_id", ["../outside.glb", "data/scene_datasets/../outside.glb"])
def test_scene_parent_traversal_is_rejected(resources, scene_id):
    dataset, scenes, episodes, _, _ = resources
    episodes[0]["scene_id"] = scene_id
    with pytest.raises(ValueError, match="must not contain"):
        validate_resources(dataset, scenes, episodes)


def test_scene_symlink_escape_is_rejected(resources, tmp_path):
    dataset, scenes, episodes, _, scene = resources
    outside = tmp_path / "outside.glb"
    outside.write_bytes(b"fixture")
    scene.unlink()
    scene.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes scenes_dir"):
        validate_resources(dataset, scenes, episodes)


def test_permission_error_names_resource_without_fallback(resources, monkeypatch):
    dataset, scenes, episodes, gt, _ = resources
    original = Path.read_bytes

    def read_bytes(path):
        if path == gt:
            raise PermissionError("denied")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(PermissionError, match="job account needs read access") as failure:
        validate_resources(dataset, scenes, episodes)
    assert str(gt) in str(failure.value)


def test_scene_permission_error_is_explicit(resources, monkeypatch):
    dataset, scenes, episodes, _, scene = resources
    original = Path.open

    def open_file(path, *args, **kwargs):
        if path == scene:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(PermissionError, match="scene geometry"):
        validate_resources(dataset, scenes, episodes)


def test_duplicate_episode_is_rejected(resources):
    dataset, scenes, episodes, _, _ = resources
    with pytest.raises(ValueError, match="Duplicate"):
        validate_resources(dataset, scenes, episodes * 2)


@pytest.mark.parametrize("split", ["../train", "val/unseen", "..", "/val_unseen"])
def test_split_cannot_escape_root(resources, split):
    dataset, scenes, episodes, _, _ = resources
    with pytest.raises(ValueError, match="single directory"):
        validate_resources(dataset, scenes, episodes, split=split)
