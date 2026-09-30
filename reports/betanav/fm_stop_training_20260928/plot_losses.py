#!/usr/bin/env python3
"""Reproduce audited FM + STOP training curves from the original JSONL log.

Requires matplotlib and numpy; CPU-only, no model weights are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RUN = Path(
    "/mnt/cfs/Models/beta_nav/r2r_fm_stop_h30_prefix_kv_all36_beta1_1p5"
)
LOSS_KEYS = ("train_loss", "train_fm", "train_stop")


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def file_evidence(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mtime_utc": utc(path.stat().st_mtime),
    }


def window_summary(rows, start, end):
    selected = [row for row in rows if start <= row["step"] <= end]
    return {
        "inclusive_step_range": [start, end],
        "logged_observations": len(selected),
        "observed_steps": [row["step"] for row in selected],
        "arithmetic_mean_of_logged_update_losses": {
            key: float(np.mean([row[key] for row in selected])) for key in LOSS_KEYS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--window-steps", type=int, default=100)
    args = parser.parse_args()
    if args.window_steps <= 0:
        parser.error("--window-steps must be positive")
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    latest_path = run / "latest.json"
    latest = json.loads(latest_path.read_text())
    checkpoint = run / "checkpoints" / latest["checkpoint"]
    trainer_path = checkpoint / "trainer.json"
    metadata_path = checkpoint / "metadata.json"
    complete_path = checkpoint / "COMPLETE"
    trainer = json.loads(trainer_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    config = metadata["config"]
    assert complete_path.read_text().strip() == metadata["format"]
    assert metadata["training_state_complete"] is True
    assert latest["step"] == trainer["step"] == metadata["step"] == config["max_steps"]
    assert trainer["epoch"] == config["num_epochs"] and trainer["batch_offset"] == 0
    assert trainer["vlm_updates"] == 0
    assert trainer["fm_updates"] == trainer["stop_updates"] == trainer["step"]
    assert config["training_mode"] == "fm_stop"

    metrics_path = run / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    rows = [row for row in rows if all(key in row for key in LOSS_KEYS)]
    steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
    assert len(steps) > 0 and np.all(np.diff(steps) > 0)
    for row in rows:
        assert all(math.isfinite(row[key]) for key in LOSS_KEYS)
        expected_total = config["lambda_fm"] * row["train_fm"] + config["lambda_stop"] * row["train_stop"]
        assert math.isclose(row["train_loss"], expected_total, rel_tol=1e-7, abs_tol=1e-7)
    expected_log_steps = [1] + list(range(config["log_every"], trainer["step"] + 1, config["log_every"]))
    missing_log_steps = sorted(set(expected_log_steps) - set(steps.tolist()))

    first, last = rows[0], rows[-1]
    first_window = window_summary(rows, int(steps[0]), int(steps[0]) + args.window_steps - 1)
    last_window = window_summary(rows, int(steps[-1]) - args.window_steps + 1, int(steps[-1]))
    counts = metadata["contract"]["stop_supervision"]["counts"]["train"]
    unique_frames = sum(counts.values())
    global_batch = config["batch_size"] * config["gradient_accumulation_steps"] * metadata["world_size"]
    steps_per_epoch = math.ceil(unique_frames / global_batch)
    summary = {
        "run_dir": str(run),
        "completed": True,
        "completion_evidence": {
            "latest": latest,
            "trainer": trainer,
            "checkpoint_path": str(checkpoint),
            "training_state_complete": metadata["training_state_complete"],
            "complete_marker": complete_path.read_text().strip(),
            "planned_max_steps": config["max_steps"],
            "planned_num_epochs": config["num_epochs"],
            "checkpoint_marker_mtime_utc": utc(complete_path.stat().st_mtime),
        },
        "training": {
            "mode": config["training_mode"],
            "world_size": metadata["world_size"],
            "global_batch_size": global_batch,
            "dataset_unique_frames": unique_frames,
            "steps_per_epoch": steps_per_epoch,
            "samples_seen_including_ddp_padding": trainer["samples_seen"],
            "stop_positive_weight": metadata["contract"]["stop_supervision"]["positive_weight"],
            "lambda_fm": config["lambda_fm"],
            "lambda_stop": config["lambda_stop"],
        },
        "logging": {
            "num_records": len(rows),
            "log_every_steps": config["log_every"],
            "first_logged_step": first["step"],
            "last_logged_step": last["step"],
            "final_checkpoint_step": trainer["step"],
            "missing_expected_logged_steps": missing_log_steps,
            "unlogged_final_steps": list(range(last["step"] + 1, trainer["step"] + 1)),
            "first_logged_utc": utc(first["time_unix"]),
            "last_logged_utc": utc(last["time_unix"]),
            "elapsed_hours_between_first_and_last_log": (last["time_unix"] - first["time_unix"]) / 3600,
            "note": "Each loss record is one logged update, not an average of all intervening updates.",
        },
        "first_logged_losses": {key: first[key] for key in LOSS_KEYS},
        "last_logged_losses": {key: last[key] for key in LOSS_KEYS},
        "first_window": first_window,
        "last_window": last_window,
        "relative_reduction_between_window_means": {
            key: 1.0 - last_window["arithmetic_mean_of_logged_update_losses"][key]
            / first_window["arithmetic_mean_of_logged_update_losses"][key]
            for key in LOSS_KEYS
        },
        "smoothing": {
            "kind": "trailing arithmetic mean of observed logged update losses",
            "window_global_steps": args.window_steps,
            "definition": "At logged step s, average existing log records whose steps satisfy s-window < step <= s; no interpolation or missing-step loss imputation.",
        },
        "interpretation": "Training losses only. They do not measure held-out navigation success rate, oracle success, SPL, or STOP accuracy/calibration.",
        "source_files": [file_evidence(path) for path in (latest_path, trainer_path, metadata_path, complete_path, metrics_path)],
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    labels = ("Total training loss", "FM loss", "Weighted STOP BCE")
    colors = ("#276A99", "#1B855E", "#AB5437")
    for index, (axis, key, label, color) in enumerate(zip(axes, LOSS_KEYS, labels, colors)):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        smoothed = np.asarray([
            values[(steps > step - args.window_steps) & (steps <= step)].mean()
            for step in steps
        ])
        axis.plot(steps, values, color=color, alpha=0.23, linewidth=0.8, label="Observed logged update")
        axis.plot(steps, smoothed, color=color, linewidth=2.0, label=f"Trailing {args.window_steps}-step mean of log records")
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#DCE0E4", linewidth=0.6)
        for epoch in range(1, config["num_epochs"]):
            boundary = epoch * steps_per_epoch
            axis.axvline(boundary, color="#A3A8AE", linestyle=":", linewidth=1)
            if index == 0:
                axis.text(boundary + 20, 0.97, f"Epoch {epoch + 1}", transform=axis.get_xaxis_transform(), va="top", color="#62676D", fontsize=9)
        axis.axvspan(last["step"], trainer["step"], color="#777777", alpha=0.12)
        axis.axvline(trainer["step"], color="#62676D", linestyle="--", linewidth=0.9)
        if index == 0:
            axis.legend(loc="upper right", frameon=False, fontsize=9)
    axes[-1].set_xlabel("Global optimizer step")
    axes[-1].set_xlim(0, trainer["step"] + 35)
    fig.suptitle("R2R: frozen LightNav VLM + trained FM and StopHead", fontsize=15, y=0.98)
    fig.text(0.5, 0.94, "8 GPUs | 3 epochs | global batch 512 | completed checkpoint: step 3699", ha="center", color="#51565B")
    fig.text(0.10, 0.033, "370 observed records: step 1, then every 10 steps through 3690. Steps 3691–3699 have no loss record.", fontsize=9, color="#51565B")
    fig.text(0.10, 0.012, "Training loss only; navigation SR/OS and STOP accuracy require separate evaluation.", fontsize=9, color="#51565B")
    fig.subplots_adjust(left=0.10, right=0.975, top=0.90, bottom=0.105, hspace=0.20)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"loss_curves.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": True, "checkpoint": str(checkpoint), "output_dir": str(output), "last_window": last_window}, indent=2))


if __name__ == "__main__":
    main()
