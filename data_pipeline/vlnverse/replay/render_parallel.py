"""Render disjoint pose ranges in separate Blender CPU processes, then verify and merge."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def frame_ranges(count, workers):
    if not isinstance(count, int) or not isinstance(workers, int) or min(count, workers) < 1:
        raise ValueError("count and workers must be positive integers")
    edges = np.linspace(0, count, min(count, workers) + 1, dtype=int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-usd", type=Path, required=True)
    parser.add_argument("--poses-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12, help="CPU threads per Blender process")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args(argv)
    with np.load(args.poses_npz, allow_pickle=False) as values:
        count = len(values["yaw_rad"])
    ranges = frame_ranges(count, args.workers)
    if min(args.threads, args.samples) < 1:
        raise ValueError("threads and samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shards = args.output_dir / "shards"
    shards.mkdir()

    def render(item):
        index, (start, stop) = item
        directory = shards / str(index)
        command = [sys.executable, "-m", "data_pipeline.vlnverse.replay.blender_render",
                   "--scene-usd", str(args.scene_usd), "--poses-npz", str(args.poses_npz),
                   "--output-dir", str(directory), "--frame-start", str(start), "--frame-stop", str(stop),
                   "--threads", str(args.threads), "--samples", str(args.samples)]
        with (shards / f"{index}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        return json.loads((directory / "render_manifest.json").read_text())

    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        manifests = list(executor.map(render, enumerate(ranges)))
    combined = dict(manifests[0])
    frames = []
    for index, ((start, stop), manifest) in enumerate(zip(ranges, manifests, strict=True)):
        for key in ("source_sha256", "scene_usd", "camera", "renderer", "samples", "input_frames"):
            if manifest[key] != combined[key]:
                raise ValueError(f"render shard mismatch: {key}")
        checks = manifest["frames"]
        if ([row["index"] for row in checks] != list(range(start, stop))
                or manifest["rendered_frames"] != stop - start):
            raise ValueError("incomplete render shard")
        for frame in range(start, stop):
            shutil.move(str(shards / str(index) / f"{frame:06d}.png"), args.output_dir / f"{frame:06d}.png")
        frames.extend(checks)
    combined.update(rendered_frames=count, frame_range=[0, count], frames=frames,
                    parallel_workers=len(ranges), total_cpu_threads=len(ranges) * args.threads,
                    shard_manifests=[str((shards / str(i) / "render_manifest.json").resolve())
                                     for i in range(len(ranges))],
                    elapsed_s=max(m["elapsed_s"] for m in manifests),
                    elapsed_semantics="maximum per-worker render time; process startup excluded")
    (args.output_dir / "render_manifest.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps({"frames": count, "workers": len(ranges), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
