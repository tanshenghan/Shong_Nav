"""Bounded-memory LeRobot v3 merge: preserve every observation/GT and remap indices.

The official image-backed writer creates each input shard. Arrow merges its
Parquet row groups without decoding/re-encoding RGB or materializing depth as
Python lists. Standard metadata is made with the official LeRobot utilities.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from data_pipeline.collect_r2r import write_json
from data_pipeline.lerobot_stats import aggregate_masked_stats


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_digest(table):
    """Logical values, including list boundaries/nulls; independent of Arrow buffers."""
    import pyarrow as pa

    digest = hashlib.sha256()

    def update(array):
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()
        if isinstance(array, pa.ExtensionArray):
            array = array.storage
        digest.update(np.asarray([len(array)], dtype="<i8").tobytes())
        digest.update(np.packbits(array.is_valid().to_numpy(zero_copy_only=False)).tobytes())
        kind = array.type
        if pa.types.is_list(kind) or pa.types.is_large_list(kind):
            digest.update(np.diff(array.offsets.to_numpy()).astype("<i8").tobytes())
            update(array.flatten())
        elif pa.types.is_fixed_size_list(kind):
            digest.update(str(kind.list_size).encode())
            update(array.flatten())
        elif pa.types.is_struct(kind):
            for index in range(kind.num_fields):
                digest.update(kind[index].name.encode())
                update(array.field(index))
        elif pa.types.is_binary(kind) or pa.types.is_string(kind) or pa.types.is_large_binary(kind) or pa.types.is_large_string(kind):
            for value in array.to_pylist():
                data = b"" if value is None else value.encode() if isinstance(value, str) else value
                digest.update(len(data).to_bytes(8, "little"))
                digest.update(data)
        else:
            values = array.to_numpy(zero_copy_only=False).copy()
            if np.issubdtype(values.dtype, np.floating):
                values[np.isnan(values)] = np.nan
            digest.update(str(values.dtype).encode())
            digest.update(values.tobytes())

    for name in table.column_names:
        digest.update(name.encode())
        update(table[name])
    return digest.hexdigest()


def merge_file(job):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source, destination, marker = map(Path, (job["source"], job["destination"], job["marker"]))
    if marker.is_file():
        result = json.loads(marker.read_text())
        if result["job"] == job and destination.is_file() and file_digest(destination) == result["sha256"]:
            return result
        raise ValueError(f"completed merge file changed: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    original = pq.ParquetFile(source)
    expected = []
    with pq.ParquetWriter(temporary, original.schema_arrow, compression="snappy", use_dictionary=True) as writer:
        for index in range(original.num_row_groups):
            table = original.read_row_group(index, use_threads=False)
            for name, offset in (("index", job["frame_offset"]), ("episode_index", job["episode_offset"])):
                field = table.schema.field(name)
                table = table.set_column(table.schema.get_field_index(name), field, pc.add(table[name], offset))
            name = "task_index"
            values = pc.take(pa.array(job["task_mapping"], type=pa.int64()), table[name])
            table = table.set_column(table.schema.get_field_index(name), table.schema.field(name), values)
            expected.append(table_digest(table))
            writer.write_table(table)
    saved = pq.ParquetFile(temporary)
    if saved.num_row_groups != len(expected):
        raise ValueError("merge changed episode row groups")
    for index, digest in enumerate(expected):
        if table_digest(saved.read_row_group(index, use_threads=False)) != digest:
            raise ValueError(f"merge readback differs: {destination}, row group {index}")
    temporary.replace(destination)
    result = dict(job=job, complete=True, sha256=file_digest(destination), frames=saved.metadata.num_rows,
                  row_groups=saved.num_row_groups, all_columns_exact_after_index_remap=True)
    write_json(marker, result)
    return result


def copy_provenance(job):
    """Copy one immutable shard's small files; repair interrupted destination copies."""
    root, destination = map(Path, job)
    source = root / "meta/provenance"
    files = [(p, destination / p.relative_to(source)) for p in source.rglob("*") if p.is_file()]
    files.extend((root / "meta" / name, destination / name) for name in ("validation.json", "build_shard.json")
                 if (root / "meta" / name).is_file())
    for original, target in files:
        expected = file_digest(original)
        if target.is_file() and file_digest(target) == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".copying")
        shutil.copyfile(original, temporary)
        if file_digest(temporary) != expected:
            raise ValueError(f"provenance copy changed: {original}")
        temporary.replace(target)
    return len(files)


def merge_datasets(roots, output, repo_id, journal, workers=4):
    """Merge validated pipeline shards; restart reuses verified completed data files."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import write_info, write_stats, write_tasks

    roots, output, journal = list(map(Path, roots)), Path(output), Path(journal)
    journal.mkdir(parents=True, exist_ok=True)
    first = json.loads((roots[0] / "meta/beta_nav.json").read_text())
    signature = dict(roots=[str(p) for p in roots], repo_id=repo_id)
    state = journal / "merge.json"
    if state.is_file():
        if json.loads(state.read_text()) != signature:
            raise ValueError("merge inputs changed")
        dst = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
        dst.root = output
        dst.info = json.loads((output / "meta/info.json").read_text())
    else:
        dst = LeRobotDatasetMetadata.create(repo_id, fps=1, features=first["features"],
            root=output, robot_type="lightnav_habitat", use_videos=False)
        write_json(state, signature)
    tasks, records, episode_stats, jobs, provenance_jobs = {}, [], [], [], []
    episode_offset, frame_offset, file_offset = 0, 0, 0
    provenance = output / "meta/provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for shard_index, root in enumerate(roots):
        custom = json.loads((root / "meta/beta_nav.json").read_text())
        info = json.loads((root / "meta/info.json").read_text())
        if custom["features"] != first["features"] or custom["rvq_assets"] != first["rvq_assets"] or info["fps"] != 1:
            raise ValueError("shards have incompatible schema, calibration assets or time basis")
        if custom.get("source_selection") != first.get("source_selection"):
            raise ValueError("shards have different source splits, roles or language selections")
        source_tasks = pd.read_parquet(root / "meta/tasks.parquet").sort_values("task_index")
        mapping = []
        for task in source_tasks.index:
            if task not in tasks:
                tasks[task] = len(tasks)
            mapping.append(tasks[task])
        file_mapping = {}
        for path in sorted((root / "data").rglob("*.parquet")):
            chunk, index = file_offset // 1000, file_offset % 1000
            relative = Path(f"data/chunk-{chunk:03d}/file-{index:03d}.parquet")
            source_key = (int(path.parent.name.split("-")[-1]), int(path.stem.split("-")[-1]))
            file_mapping[source_key] = (chunk, index)
            jobs.append(dict(source=str(path), destination=str(output / relative),
                marker=str(journal / f"file-{file_offset:06d}.json"),
                frame_offset=frame_offset, episode_offset=episode_offset, task_mapping=mapping))
            file_offset += 1
        rows = []
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            rows.extend(pq.read_table(path).to_pylist())
        rows.sort(key=lambda row: row["episode_index"])
        if len(rows) != len(custom["episodes"]) or len(rows) != info["total_episodes"]:
            raise ValueError("episode metadata counts disagree")
        meta_chunk, meta_file = shard_index // 1000, shard_index % 1000
        for local, (row, record) in enumerate(zip(rows, custom["episodes"], strict=True)):
            if row["episode_index"] != local or record["episode_index"] != local or row["length"] != record["frames"]:
                raise ValueError("episode metadata order/length mismatch")
            row["episode_index"] += episode_offset
            row["dataset_from_index"] += frame_offset
            row["dataset_to_index"] += frame_offset
            chunk, index = file_mapping[(row["data/chunk_index"], row["data/file_index"])]
            row.update({"data/chunk_index": chunk, "data/file_index": index,
                        "meta/episodes/chunk_index": meta_chunk, "meta/episodes/file_index": meta_file})
            for key, offset in (("index", frame_offset), ("episode_index", episode_offset)):
                for statistic in ("min", "max", "mean"):
                    field = f"stats/{key}/{statistic}"
                    row[field] = (np.asarray(row[field]) + offset).tolist()
            if row["tasks"] != [record["instruction"]]:
                raise ValueError("instruction and task metadata disagree")
            for statistic in ("min", "max", "mean"):
                row[f"stats/task_index/{statistic}"] = [float(tasks[record["instruction"]])]
            stats = {}
            for name, values in row.items():
                if name.startswith("stats/"):
                    _, key, statistic = name.split("/", 2)
                    stats.setdefault(key, {})[statistic] = np.asarray(values)
            episode_stats.append(stats)
            records.append(dict(record, episode_index=episode_offset + local, source_shard=shard_index))
        meta_path = output / f"meta/episodes/chunk-{meta_chunk:03d}/file-{meta_file:03d}.parquet"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), meta_path, compression="snappy")
        dest = provenance / "shards" / f"{shard_index:05d}"
        provenance_jobs.append((root, dest))
        episode_offset += info["total_episodes"]
        frame_offset += info["total_frames"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        provenance_file_count = sum(pool.map(copy_provenance, provenance_jobs))
        files = list(pool.map(merge_file, jobs))
    if sum(f["frames"] for f in files) != frame_offset:
        raise ValueError("merged data frame coverage differs from metadata")
    dst.info.update(total_episodes=episode_offset, total_frames=frame_offset, total_tasks=len(tasks),
                    splits={"train": f"0:{episode_offset}"})
    write_info(dst.info, output)
    write_tasks(pd.DataFrame({"task_index": list(tasks.values())}, index=list(tasks)), output)
    write_stats(aggregate_masked_stats(episode_stats), output)
    shutil.copytree(roots[0] / "assets", output / "assets", dirs_exist_ok=True)
    metadata = dict(first, repo_id=repo_id, source_dataset="complete selected source; see provenance/build_plan.json",
                    episodes=records, source_shards=len(roots), source_manifest_sha256=None)
    write_json(output / "meta/beta_nav.json", metadata)
    # The first shard's frozen GT implementation is retained at the standard location too.
    for name in first["code_sha256"]:
        shutil.copyfile(roots[0] / "meta/provenance" / name, provenance / name)
    shutil.copyfile(roots[0] / "README.md", output / "README.md")
    return dict(frames=frame_offset, episodes=episode_offset, tasks=len(tasks), files=len(files),
                all_parquet_columns_readback_exact=True, provenance_files_verified=provenance_file_count, file_sha256={
                    str(Path(f["job"]["destination"]).relative_to(output)): f["sha256"] for f in files})


def verify_coverage(root, plan):
    """Check every instruction, source ID, frame range and task mapping after merge."""
    import pandas as pd
    import pyarrow.parquet as pq

    root = Path(root)
    custom = json.loads((root / "meta/beta_nav.json").read_text())
    expected = {str(e["episode_id"]): (r["frames"], e["instruction"]["instruction_text"], r["scene_id"])
                for s in plan["shards"] for r in s["routes"] for e in r["episodes"]}
    source_metadata = {str(e["episode_id"]): e.get("source_metadata")
                       for s in plan["shards"] for r in s["routes"] for e in r["episodes"]}
    if len(custom["episodes"]) != len(expected) or {e["episode_id"] for e in custom["episodes"]} != expected.keys():
        raise ValueError("missing or duplicate source episodes after merge")
    tasks = pd.read_parquet(root / "meta/tasks.parquet").sort_values("task_index").index.tolist()
    columns = ["index", "episode_index", "frame_index", "timestamp", "task_index", "source.episode_id", "source.scene_id"]
    offset, episode_index = 0, 0
    for path in sorted((root / "data").rglob("*.parquet")):
        file = pq.ParquetFile(path)
        for index in range(file.num_row_groups):
            table = file.read_row_group(index, columns=columns)
            n = table.num_rows
            record = custom["episodes"][episode_index]
            source_id = record["episode_id"]
            frames, instruction, scene = expected[source_id]
            if (record["frames"], record["instruction"], record["scene_id"]) != (frames, instruction, scene) or n != frames:
                raise ValueError("source instruction, scene or frame count changed")
            if record.get("source_metadata") != source_metadata[source_id]:
                raise ValueError("source language, split, role or annotation identity changed")
            for name, values in (("index", np.arange(offset, offset + n)), ("episode_index", np.full(n, episode_index)),
                                 ("frame_index", np.arange(n)), ("timestamp", np.arange(n, dtype=float))):
                np.testing.assert_array_equal(table[name].to_numpy(), values)
            if set(table["source.episode_id"].to_pylist()) != {source_id} or set(table["source.scene_id"].to_pylist()) != {scene}:
                raise ValueError("source identity changed in frame rows")
            if {tasks[t] for t in table["task_index"].to_pylist()} != {instruction}:
                raise ValueError("task index points to the wrong instruction")
            offset += n
            episode_index += 1
    if offset != plan["frames"] or episode_index != plan["source_episodes"]:
        raise ValueError("full-split frame/episode coverage mismatch")
    return dict(all_source_ids_exact=True, all_instructions_exact=True, all_source_metadata_exact=True,
                global_frame_indices_exact=True,
                episodes=episode_index, frames=offset)


def official_load_smoke(root):
    """Official loader with selected episodes: avoid caching a second entire dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    root = Path(root)
    custom = json.loads((root / "meta/beta_nav.json").read_text())
    count = len(custom["episodes"])
    indices = sorted({0, count // 2, count - 1})
    dataset = LeRobotDataset(custom["repo_id"], root=root, episodes=indices, video_backend="pyav")
    offset = 0
    for index in indices:
        record = custom["episodes"][index]
        for t in sorted({0, record["frames"] // 2, record["frames"] - 1}):
            row = dataset[offset + t]
            assert row["task"] == record["instruction"] and row["source.episode_id"] == record["episode_id"]
            assert row["frame_index"].item() == t and row["episode_index"].item() == index
            assert tuple(row["action.fm.waypoints_ego"].shape) == (30, 3)
        offset += record["frames"]
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert tuple(batch["action.fm.waypoints_ego"].shape) == (2, 30, 3)
    return dict(official_merged_loader_episode_indices=indices, merged_dataloader_batch=True)


def merge_train(work, plan, workers=4):
    work = Path(work)
    output = work / "assembled"
    roots = [work / "shards" / f"{s['id']:05d}" for s in plan["shards"]]
    report = merge_datasets(roots, output, plan["repo_id"], work / "merge", workers)
    report.update(verify_coverage(output, plan))
    report.update(official_load_smoke(output))
    report.update(complete=True, plan_sha256=plan["plan_sha256"], source_episodes=plan["source_episodes"],
        physical_routes=plan["physical_routes"], scenes=plan["scenes"],
        all_shards_independently_validated=True, shard_validation="all frames: official loader, RGB/depth/GT hashes and geometry",
        merged_validation="every Parquet value read back exactly after index remapping; complete source coverage; official loader sample")
    shutil.copyfile(work / "plan.json", output / "meta/provenance/build_plan.json")
    if (work / "definition_updates").is_dir():
        shutil.copytree(work / "definition_updates", output / "meta/provenance/definition_updates", dirs_exist_ok=True)
        report["compatible_definition_updates"] = [
            json.loads(p.read_text()) for p in sorted((work / "definition_updates").glob("*.json"))]
    for name in ("build_r2r_train.py", "merge_lerobot.py", "vlnce_source.py", "run_rxr.sh", "run_r2r_train.sh"):
        shutil.copyfile(Path(__file__).parent / name, output / "meta/provenance" / name)
    report["source_selection"] = plan.get("source_selection")
    write_json(output / "meta/full_train_validation.json", report)
    selection = plan.get("source_selection") or dict(dataset="r2r", split="train", role=None, languages=["all"])
    info = json.loads((output / "meta/info.json").read_text())
    nbytes = sum(p.stat().st_size for p in (output / "data").rglob("*.parquet"))
    write_json(output / "meta/summary.json", dict(complete=True, format="LeRobotDataset v3.0",
        dataset_root=plan["output_dir"], repo_id=plan["repo_id"], source_selection=selection,
        episodes=report["episodes"], frames=report["frames"], tasks=report["tasks"], files=report["files"],
        physical_routes=plan["physical_routes"], scenes=plan["scenes"], features=len(info["features"]),
        data_parquet_bytes=nbytes, data_parquet_gib=round(nbytes / 1024**3, 3),
        width=plan.get("width", 470), height=plan.get("height", 280), fps=1, time_basis="frame_index",
        physical_dt_s=None, fm_horizon=30, rvq_horizon=10, opos_visibility_window="last_half",
        opos_rule="t >= floor(N/2) AND actual projection/depth visibility", unknown_pointing_token_id=0,
        self_contained=True, plan_sha256=plan["plan_sha256"], validation="meta/full_train_validation.json"))
    with (output / "README.md").open("a") as stream:
        stream.write(f"\nComplete local {selection['dataset'].upper()} {selection['split']} "
                     f"(role={selection['role']}, languages={selection['languages']}): "
                     f"{plan['source_episodes']} instruction episodes, "
                     f"{plan['physical_routes']} physical routes, {plan['scenes']} scenes, {plan['frames']} frames. "
                     "All selected original instructions are retained. Source splits and roles are stored separately. "
                     "LeRobot's train index slice is a storage convention; the original source split is recorded "
                     "in meta/summary.json and per-episode source_metadata.\n")
    output.rename(Path(plan["output_dir"]))
    return report
