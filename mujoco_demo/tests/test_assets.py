import json
from pathlib import Path

from vln_mujoco.model import scene_path


def test_asset_manifest_matches_bundled_files() -> None:
    assets = scene_path().parents[2]
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scene_id"] == "procthor-val-2"
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["total_bytes"] == sum(item["bytes"] for item in manifest["files"])
    assert all((assets / Path(item["path"])).is_file() for item in manifest["files"])
