"""CPU-only resource checks for queued R2R evaluation jobs.

This module deliberately does not import Torch or Habitat. Habitat imports run in
the independent Python 3.9 interpreter; checks never construct a simulator.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

_IMPORT_MARKER = "LIGHTNAV_HABITAT_IMPORTS="


def _prepend_path(value: str, existing: str) -> str:
    return os.pathsep.join(dict.fromkeys([value, *filter(None, existing.split(os.pathsep))]))


def build_habitat_env(
    repo: str | Path, base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Reuse local GL libraries and retain the caller's allocated GPU visibility.

    The MuJoCo demo already supplies these GLVND libraries in its local runtime.
    Missing libraries are diagnosed by ``check_habitat_python``; this function
    never downloads, installs, or rewrites an environment.
    """
    repo = Path(repo).resolve()
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONPATH"] = _prepend_path(str(repo / "habitat_server"), env.get("PYTHONPATH", ""))
    gl_dir = repo / ".venv-beta-demo/egl/usr/lib/x86_64-linux-gnu"
    if gl_dir.is_dir():
        env["LD_LIBRARY_PATH"] = _prepend_path(str(gl_dir), env.get("LD_LIBRARY_PATH", ""))
    vendor = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if vendor.is_file():
        env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HABITAT_SIM_LOG", "quiet")
    env.setdefault("MAGNUM_LOG", "quiet")
    return env


def check_habitat_python(python: str | Path, env: Mapping[str, str]) -> dict[str, Any]:
    """Import Habitat in its own interpreter, with a bounded startup timeout."""
    code = (
        "import sys, json; "
        "assert sys.version_info[:2] == (3, 9), 'Habitat requires its separate Python 3.9 environment'; "
        "import habitat, habitat_sim, lightnav_habitat.serve; "
        f"print({_IMPORT_MARKER!r} + json.dumps(dict("
        "python_version=sys.version.split()[0], habitat=habitat.__file__, "
        "habitat_sim=habitat_sim.__file__, imports_ok=True)))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code], env=dict(env), capture_output=True,
            text=True, timeout=90, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Habitat import check timed out after 90 seconds: {python}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot execute Habitat Python {python}: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"Habitat import check failed for {python} (exit {result.returncode}). "
            "Use the separate Python 3.9 Habitat environment. If libOpenGL.so.0 or "
            "libEGL.so.1 is missing, make existing GLVND libraries available through "
            f"LD_LIBRARY_PATH on the compute node.\n{detail}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_IMPORT_MARKER):
            try:
                metadata = json.loads(line[len(_IMPORT_MARKER):])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid Habitat import check response from {python}") from exc
            if not isinstance(metadata, dict) or metadata.get("imports_ok") is not True:
                raise RuntimeError(f"Invalid Habitat import check response from {python}")
            return {"python": str(python), **metadata}
    raise RuntimeError(f"Habitat import check returned no confirmation from {python}")


def _resource_error(kind: str, path: Path, exc: OSError) -> OSError:
    if isinstance(exc, PermissionError):
        return PermissionError(
            f"Cannot read {kind}: {path}. The job account needs read access to this "
            "resource and search access to its parent directories."
        )
    return OSError(f"Cannot read {kind}: {path}: {exc}")


def _scene_path(scene_id: str, scenes_dir: Path) -> Path:
    if not isinstance(scene_id, str) or not scene_id or "\x00" in scene_id:
        raise ValueError("Each R2R episode needs a nonempty scene_id path")
    source = scene_id.removeprefix("data/scene_datasets/")
    relative = Path(source)
    if ".." in relative.parts:
        raise ValueError(f"Scene path must not contain '..': {scene_id}")
    path = scenes_dir / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _resource_error("scene geometry", path, exc) from exc
    if not resolved.is_relative_to(scenes_dir):
        raise ValueError(f"Scene path escapes scenes_dir: {scene_id}")
    if resolved.suffix.lower() != ".glb":
        raise ValueError(f"R2R scene geometry must be a .glb file: {resolved}")
    return resolved


def validate_resources(
    dataset_root: str | Path,
    scenes_dir: str | Path,
    episodes: Sequence[Mapping[str, Any]],
    split: str = "val_unseen",
) -> dict[str, Any]:
    """Validate NDTW coverage and readable scene geometry without reading meshes.

    ``episodes`` is the original split's episode list, already loaded by the
    caller. GLB fingerprints record size and mtime, not a content digest. The
    smaller compressed NDTW file receives a SHA-256 content digest.
    """
    if not isinstance(split, str) or not split or Path(split).name != split or split in {".", ".."}:
        raise ValueError("split must be a single directory name")
    if not episodes:
        raise ValueError("Cannot validate an empty R2R episode list")
    dataset_root = Path(dataset_root).absolute()
    try:
        scenes_dir = Path(scenes_dir).resolve(strict=True)
        if not stat.S_ISDIR(scenes_dir.stat().st_mode):
            raise ValueError(f"scenes_dir is not a directory: {scenes_dir}")
    except OSError as exc:
        raise _resource_error("scene directory", Path(scenes_dir), exc) from exc
    gt_path = dataset_root / split / f"{split}_gt.json.gz"
    try:
        compressed = gt_path.read_bytes()
    except OSError as exc:
        raise _resource_error("NDTW ground truth", gt_path, exc) from exc
    try:
        gt = json.loads(gzip.decompress(compressed))
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid compressed NDTW JSON: {gt_path}: {exc}") from exc
    if not isinstance(gt, dict):
        raise ValueError(f"NDTW ground truth must map episode IDs to paths: {gt_path}")

    scenes: dict[Path, dict[str, Any]] = {}
    keys: set[tuple[str, str]] = set()
    checked_gt: set[str] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping) or episode.get("episode_id") is None:
            raise ValueError(f"R2R episode {index} needs an episode_id")
        episode_id = str(episode["episode_id"])
        if not episode_id:
            raise ValueError(f"R2R episode {index} has an empty episode_id")
        if episode_id not in checked_gt:
            entry = gt.get(episode_id)
            locations = entry.get("locations") if isinstance(entry, dict) else None
            if not isinstance(locations, list) or not locations:
                raise ValueError(f"NDTW path missing or empty for episode {episode_id}: {gt_path}")
            for location in locations:
                if not isinstance(location, list) or len(location) != 3 or any(
                    isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    for value in location
                ):
                    raise ValueError(f"NDTW locations must be finite 3D coordinates for episode {episode_id}")
            checked_gt.add(episode_id)
        scene = _scene_path(episode.get("scene_id"), scenes_dir)
        key = (scene.relative_to(scenes_dir).as_posix(), episode_id)
        if key in keys:
            raise ValueError(f"Duplicate R2R episode key: {key}")
        keys.add(key)
        if scene in scenes:
            continue
        try:
            metadata = scene.stat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
                raise ValueError(f"Scene geometry must be a nonempty regular file: {scene}")
            with scene.open("rb"):
                pass  # Verify read permission without reading or hashing the mesh body.
        except OSError as exc:
            raise _resource_error("scene geometry", scene, exc) from exc
        scenes[scene] = {
            "path": str(scene), "size_bytes": metadata.st_size, "mtime_ns": metadata.st_mtime_ns,
        }
    return {
        "dataset_root": str(dataset_root), "scenes_dir": str(scenes_dir), "split": split,
        "num_episodes": len(episodes), "num_scenes": len(scenes),
        "episode_keys_sha256": hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest(),
        "ndtw": {
            "path": str(gt_path), "sha256": hashlib.sha256(compressed).hexdigest(),
            "size_bytes": len(compressed), "num_entries": len(gt),
            "num_checked_entries": len(checked_gt),
        },
        "scenes": [scenes[path] for path in sorted(scenes)],
    }
