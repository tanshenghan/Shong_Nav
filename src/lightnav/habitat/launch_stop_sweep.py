"""Run eight paired STOP ablations on the official R2R val_seen development split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lightnav.habitat.launch_fm_stop_eval import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA,
    DEFAULT_SCENES,
    REPOSITORY,
    gpu_tokens,
    write_json,
)
from lightnav.habitat.stop_report import build_stop_report, check_coverage

THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
CONFIRMATIONS = (1, 2)
# These fields must agree with the preflight snapshot, not just with another group.
PREFLIGHT_FIELDS = (
    "checkpoint", "checkpoint_step", "checkpoint_metadata_sha256", "episode_file_sha256",
    "split", "resources", "development_audit",
)
SHARED_FIELDS = (
    "model_path", "dataset_root", "scenes_dir", "habitat_python", "habitat_config",
    "fm_seed", "fm_steps", "max_steps", "success_distance", "workers_per_gpu",
    "shards_per_worker", "cpu_threads", "num_shards", "episodes", "shards",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-path", default="/mnt/cfs/Models/Light")
    parser.add_argument("--dataset-root", default=DEFAULT_DATA)
    parser.add_argument("--scenes-dir", default=DEFAULT_SCENES)
    parser.add_argument("--habitat-python", default=str(REPOSITORY / ".venv-habitat/bin/python"))
    parser.add_argument("--habitat-config", "--config", default=str(REPOSITORY / "habitat_server/configs/vlnce_r2r.yaml"))
    parser.add_argument("--split", choices=("val_seen",), default="val_seen")
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--shards-per-worker", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=29600)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--load-parallelism", type=int, default=2,
                        help="global model-loading limit; each group loads one model at a time")
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--load-timeout", type=int, default=1200)
    parser.add_argument("--shard-timeout", type=int, default=21600)
    parser.add_argument("--shutdown-timeout", type=int, default=30)
    parser.add_argument("--fm-seed", "--seed", type=int, default=42)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true",
                      help="print the eight-group plan without accessing data, models or GPUs")
    mode.add_argument("--check-only", action="store_true",
                      help="validate shared data, model metadata and Habitat imports; do not launch GPUs")
    args = parser.parse_args(argv)
    for field in ("shards_per_worker", "cpu_threads", "load_parallelism", "ready_timeout",
                  "load_timeout", "shard_timeout", "shutdown_timeout", "fm_steps"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.fm_seed < 0 or not 1 <= args.max_steps <= 500:
        parser.error("fm-seed must be nonnegative and max-steps must be in [1,500]")
    if args.base_port < 1024 or args.base_port + 8 * args.workers_per_gpu > 65536:
        parser.error("all 48 worker ports must fit between 1024 and 65535")
    return args


def child_arguments(args, group):
    """Only threshold and confirmation count vary between the eight protocols."""
    fields = ("checkpoint", "model_path", "dataset_root", "scenes_dir", "habitat_python",
              "habitat_config", "split", "workers_per_gpu", "shards_per_worker", "cpu_threads",
              "ready_timeout", "load_timeout", "shard_timeout", "shutdown_timeout",
              "fm_seed", "fm_steps", "max_steps")
    command = []
    for field in fields:
        command.extend(("--" + field.replace("_", "-"), str(getattr(args, field))))
    command.extend(("--num-gpus", "1", "--load-parallelism", "1",
                    "--base-port", str(group["base_port"]),
                    "--stop-threshold", str(group["stop_threshold"]),
                    "--stop-confirm-steps", str(group["stop_confirm_steps"]),
                    "--output-dir", group["output_dir"]))
    return command


def make_plan(args, output, tokens=None):
    tokens = [f"allocated_gpu_{i}" for i in range(8)] if tokens is None else tokens
    if len(tokens) != 8 or len(set(tokens)) != 8:
        raise ValueError("the sweep requires exactly eight distinct allocated GPU tokens")
    groups = []
    for threshold in THRESHOLDS:
        for confirmation in CONFIRMATIONS:
            index = len(groups)
            name = f"threshold_{str(threshold).replace('.', 'p')}_confirm_{confirmation}"
            group = dict(name=name, gpu_token=tokens[index], stop_threshold=threshold,
                         stop_confirm_steps=confirmation,
                         base_port=args.base_port + index * args.workers_per_gpu,
                         output_dir=str(output / name))
            group["command"] = [sys.executable, "-u", "-m", "lightnav.habitat.launch_fm_stop_eval",
                                *child_arguments(args, group)]
            groups.append(group)
    return dict(schema_version="lightnav_stop_sweep_v1", output_dir=str(output), split=args.split,
                purpose="development_parameter_selection_not_final_test",
                checkpoint=args.checkpoint, fm_seed=args.fm_seed, fm_steps=args.fm_steps,
                max_steps=args.max_steps, workers_per_gpu=args.workers_per_gpu,
                total_workers=8 * args.workers_per_gpu, global_load_parallelism=args.load_parallelism,
                stop_confirmation_semantics="consecutive policy decisions; continue FM movement until confirmed",
                gpu_assignment_verified=tokens[0] != "allocated_gpu_0", groups=groups)


def start_group(group, output):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=group["gpu_token"], PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(REPOSITORY / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with (output / f"{group['name']}.log").open("x") as log:
        process = subprocess.Popen(group["command"], env=env, cwd=REPOSITORY,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return dict(group=group, process=process)


def group_loading(slot, worker_count):
    """A group retains one global load slot until every replica has booted."""
    if slot["process"].poll() is not None:
        return False
    files = list(Path(slot["group"]["output_dir"]).glob("workers/worker_*/status.json"))
    if len(files) < worker_count:
        return True
    return any(json.loads(path.read_text())["status"] == "starting" for path in files)


def _proc_identity(pid):
    """Linux process identity and ancestry for safe cleanup of our own sessions."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[3]), int(fields[19])  # ppid, session, starttime
    except (OSError, ValueError, IndexError):
        return None


def _owned_worker_groups(slots):
    owned = []
    for slot in slots:
        leader = slot["process"].pid
        for path in Path(slot["group"]["output_dir"]).glob("workers/worker_*/status.json"):
            pid = json.loads(path.read_text()).get("pid")
            identity = _proc_identity(pid) if isinstance(pid, int) else None
            if identity and identity[0] == leader and identity[1] == pid:
                owned.append((pid, identity[2]))
    return owned


def stop_groups(slots, timeout=30.):
    """Let child launchers clean up their workers; kill only verified owned groups."""
    worker_groups = _owned_worker_groups(slots)
    for slot in slots:
        if slot["process"].poll() is None:
            slot["process"].terminate()
    deadline = time.monotonic() + timeout
    for slot in slots:
        process = slot["process"]
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            # This launcher is our unreaped Popen child, so its PID cannot be reused.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    for pid, started in worker_groups:
        identity = _proc_identity(pid)
        if identity and identity[1:] == (pid, started):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_groups(plan, *, spawn=start_group, loading=group_loading, cleanup=stop_groups, interval=1.):
    output = Path(plan["output_dir"])
    pending, slots, completed = list(plan["groups"]), [], set()
    handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"STOP sweep interrupted by signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        while pending or len(completed) < len(slots):
            for slot in slots:
                code = slot["process"].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"{slot['group']['name']} exited {code}; see {output / (slot['group']['name'] + '.log')}")
                state = json.loads((Path(slot["group"]["output_dir"]) / "status.json").read_text())
                if state.get("status") != "completed":
                    raise RuntimeError(f"{slot['group']['name']} exited without completed status: {state}")
                completed.add(slot["group"]["name"])
            booting = sum(loading(slot, plan["workers_per_gpu"]) for slot in slots)
            while pending and booting < plan["global_load_parallelism"]:
                group = pending.pop(0)
                print(f"Starting {group['name']} on {group['gpu_token']}", flush=True)
                slots.append(spawn(group, output))
                booting += 1
            write_json(output / "progress.json", dict(completed_groups=sorted(completed),
                       launched_groups=len(slots), waiting_groups=len(pending), loading_groups=booting))
            if pending or len(completed) < len(slots):
                time.sleep(interval)
    finally:
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            cleanup(slots)
        finally:
            for signum, previous in handlers.items():
                signal.signal(signum, previous)


def _read_json(path):
    return json.loads(path.read_text())


def _metric_mean(records, field):
    values = [float(row[field]) for row in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"missing or non-finite {field} in sweep results")
    return statistics.fmean(values)


def validate_preflight(preflight):
    for field in PREFLIGHT_FIELDS:
        if field not in preflight:
            raise ValueError(f"single-group preflight did not record required field: {field}")
    audit = preflight["development_audit"]
    if (preflight["split"] != "val_seen" or not isinstance(audit, dict)
            or audit.get("verified") is not True or audit.get("split") != "val_seen"):
        raise ValueError("val_seen development independence must be verified before the sweep")
    count = preflight.get("num_episodes")
    if type(count) is not int or count < 1 or audit.get("num_development_episodes") != count:
        raise ValueError("development audit episode count differs from the evaluation dataset")


def summarize_sweep(plan, preflight):
    """Reject mixed checkpoints, data, protocols or incomplete episode coverage."""
    validate_preflight(preflight)
    rows, reference, reference_instructions = [], None, None
    for group in plan["groups"]:
        folder = Path(group["output_dir"])
        if _read_json(folder / "status.json").get("status") != "completed":
            raise ValueError(f"group is not completed: {group['name']}")
        job, summary = _read_json(folder / "job.json"), _read_json(folder / "summary.json")
        for field in PREFLIGHT_FIELDS:
            if field not in preflight or job.get(field) != preflight[field]:
                raise ValueError(f"{group['name']}: preflight mismatch for {field}")
        for field in ("stop_threshold", "stop_confirm_steps"):
            if job.get(field) != group[field] or summary.get(field) != group[field]:
                raise ValueError(f"{group['name']}: protocol mismatch for {field}")
        expected_summary = dict(beta_checkpoint=preflight["checkpoint"],
                                beta_checkpoint_step=preflight["checkpoint_step"],
                                training_mode="fm_stop", stop_protocol="learned",
                                fm_seed=plan["fm_seed"], fm_steps=plan["fm_steps"], max_steps=plan["max_steps"])
        for field, value in expected_summary.items():
            if summary.get(field) != value:
                raise ValueError(f"{group['name']}: summary mismatch for {field}")
        shared = {field: job[field] for field in SHARED_FIELDS}
        if reference is not None and shared != reference:
            raise ValueError(f"{group['name']}: shared protocol or episode manifest differs")
        reference = shared
        records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines() if line.strip()]
        coverage = check_coverage(job["episodes"], records)
        if (not coverage["complete"] or not _read_json(folder / "coverage.json").get("complete")
                or summary.get("num_episodes") != len(records) or len(records) != preflight["num_episodes"]):
            raise ValueError(f"{group['name']}: episode coverage is not complete")
        if any(record.get("oracle_stop") for record in records):
            raise ValueError("oracle STOP is prohibited in a development sweep")
        instructions = sorted((str(row["scene_id"]), str(row["habitat_episode_id"]), row["instruction"]) for row in records)
        if reference_instructions is not None and instructions != reference_instructions:
            raise ValueError(f"{group['name']}: episode instructions differ")
        reference_instructions = instructions
        diagnostics = build_stop_report(records, success_distance=job["success_distance"])
        learned = [row for row in records if row.get("stop_source") == "learned_head"]
        failed = [row for row in learned if not row["success"]]
        metrics = diagnostics["metrics"]
        rows.append(dict(group=group["name"], threshold=group["stop_threshold"],
                         confirm_steps=group["stop_confirm_steps"], episodes=len(records),
                         SR_pct=metrics["SR_success_rate_pct"], OS_pct=metrics["OS_oracle_success_pct"],
                         SPL_pct=metrics["SPL_pct"], nDTW_pct=100 * _metric_mean(records, "ndtw"),
                         NE_m=_metric_mean(records, "final_distance"), mean_steps=_metric_mean(records, "steps"),
                         OS_minus_SR_pp=metrics["OS_minus_SR_percentage_points"],
                         learned_stop_count=len(learned), learned_stop_failures=len(failed),
                         failed_stop_never_reached=sum(not row["oracle_success"] for row in failed),
                         failed_stop_reached_then_left=sum(bool(row["oracle_success"]) for row in failed),
                         max_steps_count=sum(row.get("stop_source") == "max_steps" for row in records),
                         fm_zero_stop_count=sum(row.get("stop_source") == "fm_zero_command" for row in records),
                         output_dir=str(folder)))
    best = max(rows, key=lambda row: (row["SR_pct"], row["SPL_pct"], -row["mean_steps"]))
    report = dict(schema_version="lightnav_stop_sweep_comparison_v1", split=plan["split"],
                  purpose=plan["purpose"], preflight=preflight, rows=rows,
                  development_selection=dict(group=best["group"], threshold=best["threshold"],
                      confirm_steps=best["confirm_steps"], rule="maximize SR, then SPL, then minimize mean steps",
                      caveat="Development selection only; evaluate the selected setting once on val_unseen."))
    output = Path(plan["output_dir"])
    write_json(output / "comparison.json", report)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# STOP development sweep", "", "Split: val_seen. These are development results, not final val_unseen results.", "",
             "| Threshold | Confirm | SR % | OS % | SPL % | nDTW % | NE m | Steps | Failed STOP | Never reached | Reached then left | Max steps |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['threshold']} | {row['confirm_steps']} | {row['SR_pct']:.2f} | {row['OS_pct']:.2f} | "
                     f"{row['SPL_pct']:.2f} | {row['nDTW_pct']:.2f} | {row['NE_m']:.3f} | {row['mean_steps']:.1f} | "
                     f"{row['learned_stop_failures']} | {row['failed_stop_never_reached']} | "
                     f"{row['failed_stop_reached_then_left']} | {row['max_steps_count']} |")
    lines.extend(("", "Failed STOP counts only failed learned-head terminations, not budget stops.", "",
                  f"Development selection: **{best['group']}** (SR, then SPL, then fewer steps).",
                  "Run the selected setting on val_unseen separately; the sweep does not do so automatically.", ""))
    (output / "comparison.md").write_text("\n".join(lines))
    return report


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or REPOSITORY / "outputs/betanav_stop_sweeps" / f"{stamp}_r2r_val_seen").resolve()
    plan = make_plan(args, output)
    if args.plan_only:
        if args.output_dir:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "sweep_plan.json", plan)
        print(json.dumps(plan, indent=2), flush=True)
        return plan

    # Shared preflight happens before any GPU is used or any group is launched.
    from lightnav.habitat.launch_fm_stop_eval import main as single_main

    preflight = single_main([*child_arguments(args, plan["groups"][0]), "--check-only"])
    args.checkpoint = preflight["checkpoint"]  # Freeze a resolved path; never follow latest mid-sweep.
    validate_preflight(preflight)
    if args.check_only:
        plan = make_plan(args, output)
        result = dict(plan=plan, preflight=preflight)
        if args.output_dir:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "preflight.json", preflight)
            write_json(output / "sweep_plan.json", plan)
        print(json.dumps(result, indent=2), flush=True)
        return result

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("STOP sweep requires eight allocated CUDA GPUs")
    tokens = gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count(), 8)
    plan = make_plan(args, output, tokens)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "sweep_plan.json", plan)
    write_json(output / "preflight.json", preflight)
    started = time.monotonic()
    write_json(output / "status.json", dict(status="running", pid=os.getpid(), groups=8))
    print(f"STOP sweep outputs: {output}", flush=True)
    try:
        run_groups(plan)
        report = summarize_sweep(plan, preflight)
        write_json(output / "status.json", dict(status="completed", groups=8,
                   elapsed_seconds=time.monotonic() - started, comparison=str(output / "comparison.json")))
    except BaseException as exc:
        write_json(output / "status.json", dict(status="failed", error=repr(exc),
                   elapsed_seconds=time.monotonic() - started))
        raise
    print(f"STOP sweep complete: {output / 'comparison.md'}", flush=True)
    return report


if __name__ == "__main__":
    main()
