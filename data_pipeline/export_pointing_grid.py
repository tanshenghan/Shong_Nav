#!/usr/bin/env python3
"""Add explicit image pixels and 48x27 grid cells to an existing pointing dataset.

Writes a NEW overlay, preserving old geometry, RVQ/FM, API results and provenance.
CPU-only; does not call a model or re-render Habitat observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data_pipeline.annotate_r2r import provenance_code_paths  # noqa: E402
from data_pipeline.collect_r2r import sha256, write_json  # noqa: E402
from data_pipeline.pointing import (  # noqa: E402
    attach_pixel_grid,
    pixel_grid_arrays,
    pixel_grid_metadata,
)
from data_pipeline.validate_pointing import validate_pointing  # noqa: E402


def write_pixel_csv(path, rows):
    fields = ["frame_id"]
    for channel in ("apos", "opos"):
        fields.extend(channel + "_" + key for key in
                      ("state", "u", "v", "pixel_row", "pixel_col", "grid_row", "grid_col", "cell_id"))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {"frame_id": row["frame_id"]}
            for channel in ("apos", "opos"):
                p = row[channel]
                record[channel + "_state"] = p["state"]
                if p["grid"] is not None:
                    values = dict(u=p["pixel"][0], v=p["pixel"][1], pixel_row=p["pixel_rc"][0],
                                  pixel_col=p["pixel_rc"][1], grid_row=p["grid"]["row"],
                                  grid_col=p["grid"]["col"], cell_id=p["grid"]["cell_id"])
                    record.update({channel + "_" + k: v for k, v in values.items()})
            writer.writerow(record)


def export_grid(dataset, output, episode_id=None):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("grid export requires a new output directory")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("format") != "beta_nav_pointing_v1":
        raise ValueError("input must be an existing pointing annotation overlay")
    validate_pointing(dataset)
    folders = sorted(p for p in (dataset / "episodes").iterdir()
                     if p.is_dir() and (episode_id is None or p.name == str(episode_id)))
    if not folders:
        raise ValueError("no matching episode")
    records = [json.loads((p / "episode.json").read_text()) for p in folders]
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    parent = provenance / "source_run"
    if (dataset / "provenance").is_dir():
        shutil.copytree(dataset / "provenance", parent)
    else:
        parent.mkdir()
    shutil.copyfile(dataset / "manifest.json", parent / "input_manifest.json")
    code_paths = provenance_code_paths(REPO / "data_pipeline")
    for path in code_paths:
        shutil.copyfile(path, provenance / path.name)
    # Preserve the exact shared model-grid encoder used by these labels, too.
    encoder = REPO / "src/lightnav/vln_utils.py"
    shutil.copyfile(encoder, provenance / "vln_utils.py")
    manifest.update(
        episodes=[p.name for p in folders],
        pointing_grid=pixel_grid_metadata(records[0]["camera"]["width"], records[0]["camera"]["height"]),
        derived_from_annotations=str(dataset),
        derived_manifest_sha256=sha256(dataset / "manifest.json"),
        transformation="add_pixel_and_grid_indices_without_changing_labels",
        api_calls_this_export=0, parent_provenance="provenance/source_run",
        code_sha256={p.name: sha256(p) for p in [*code_paths, encoder]},
    )
    write_json(output / "manifest.json", manifest)
    for folder, metadata in zip(folders, records):
        destination = output / "episodes" / folder.name
        shutil.copytree(folder, destination,
                        ignore=shutil.ignore_patterns("episode.json", "samples.jsonl", "targets.npz",
                                                      "pointing_pixels.csv"))
        rows = [json.loads(s) for s in (folder / "samples.jsonl").read_text().splitlines()]
        w, h = metadata["camera"]["width"], metadata["camera"]["height"]
        with np.load(folder / "targets.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        for channel in ("apos", "opos"):
            for row in rows:
                row[channel] = attach_pixel_grid(row[channel], w, h)
            arrays.update(pixel_grid_arrays([r[channel] for r in rows], channel))
        np.savez_compressed(destination / "targets.npz", **arrays)
        with (destination / "samples.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        write_pixel_csv(destination / "pointing_pixels.csv", rows)
        metadata.update(pointing_grid=pixel_grid_metadata(w, h),
                        source_annotation=str(folder),
                        source_annotation_hashes={name: sha256(folder / name) for name in
                                                  ("episode.json", "samples.jsonl", "targets.npz")})
        write_json(destination / "episode.json", metadata)
    validation = validate_pointing(output)
    summary = {**validation, "complete": True, "api_calls_this_export": 0,
               "opos_requested": manifest.get("opos_requested", False),
               "bound_objects": sum(r["object_bound"] for r in records),
               "derived_from_annotations": str(dataset)}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Existing pointing annotation dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW enriched dataset root")
    parser.add_argument("--episode-id", help="Only enrich this episode; default all input episodes")
    args = parser.parse_args()
    print(json.dumps(export_grid(args.dataset, args.output_dir, args.episode_id), indent=2))


if __name__ == "__main__":
    main()
