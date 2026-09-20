#!/usr/bin/env python3
"""Copy exactly one MolmoSpaces MJCF and the files it references into this repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_xml", type=Path, help="MolmoSpaces scene XML")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vln_mujoco" / "assets",
    )
    return parser.parse_args()


def revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> None:
    args = arguments()
    source = args.source_xml.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"scene XML does not exist: {source}")
    if source.parent.name != "procthor-10k-val" or source.name != "val_2_ceiling.xml":
        raise SystemExit("vln_mujoco is pinned to procthor-10k-val/val_2_ceiling.xml")
    assets_root = source.parents[2]
    project_root = assets_root.parent
    destination = args.destination.expanduser().resolve()

    sources = {source}
    for element in ET.parse(source).getroot().iter():
        filename = element.get("file")
        if not filename:
            continue
        path = (source.parent / filename).resolve()
        try:
            path.relative_to(assets_root)
        except ValueError as exc:
            raise SystemExit(f"asset escapes MolmoSpaces assets root: {path}") from exc
        if not path.is_file():
            raise SystemExit(f"referenced scene asset is missing: {path}")
        sources.add(path)

    stem = source.stem.removesuffix("_ceiling")
    for optional in (
        source.parent / f"{stem}_map.png",
        source.parent / f"{stem}_metadata.json",
        source.parent / f"{stem}.json",
        source.parent / "thumbnails" / f"{stem}_thumbnail.jpg",
    ):
        if optional.is_file():
            sources.add(optional.resolve())

    total_bytes = 0
    entries: list[dict[str, object]] = []
    for path in sorted(sources):
        relative = path.relative_to(assets_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        size = target.stat().st_size
        total_bytes += size
        entries.append({"path": relative.as_posix(), "bytes": size, "sha256": digest(target)})

    manifest = {
        "scene": "procthor-10k-val/val_2_ceiling.xml",
        "scene_id": "procthor-val-2",
        "upstream": "https://github.com/allenai/molmospaces",
        "upstream_revision": revision(project_root),
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "creator": "Allen Institute for AI (Ai2)",
        "attribution": "Scene and models by the Allen Institute for AI (Ai2), licensed under CC BY 4.0.",
        "modifications": "Only files referenced by val_2 were copied; the ceiling MJCF is used and runtime freezes environment joints.",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Vendored {len(entries)} files ({total_bytes / 1024 / 1024:.1f} MiB)")
    print(manifest_path)


if __name__ == "__main__":
    main()
