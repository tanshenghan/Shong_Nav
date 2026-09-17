"""Render the completed BetaNav training history, retaining the final resume lineage."""

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/betanav-mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
SOURCE = Path("/mnt/cfs/Models/beta_nav/r2r_ce_fm_01_start/metrics.jsonl")
CHECKPOINT = SOURCE.parent / "checkpoints/step-00029589"
START, END = 2000, 29589
DATA_SWITCH = 3047
EPOCH_BOUNDARIES = (12911, 22775)
RESUMES = (18000, 26000)


def trailing_mean(steps, values, window, boundaries):
    result = np.empty(len(values))
    segment_start = 0
    for i, step in enumerate(steps):
        if i and any(steps[i - 1] <= boundary < step for boundary in boundaries):
            segment_start = i
        result[i] = np.mean(values[max(segment_start, i - window + 1):i + 1])
    return result


def main():
    raw = SOURCE.read_bytes()
    train_rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        if isinstance(row, dict) and "train_loss" in row and row["step"] <= END:
            train_rows.append({**row, "source_line": line_number})
    by_step = {row["step"]: row for row in train_rows}
    full = [by_step[step] for step in sorted(by_step)]
    assert all(b["time_unix"] > a["time_unix"] for a, b in zip(full, full[1:]))
    selected = [row for row in full if START <= row["step"] <= END]
    assert {row["step"] for row in selected} == set(range(START, END + 1, 10))
    assert all(math.isfinite(row[key]) for row in full for key in
               ("train_loss", "train_ce", "train_fm", "update_seconds"))
    assert all(abs(row["train_loss"] - row["train_ce"] - 2 * row["train_fm"]) < 1e-5
               for row in selected)
    final_state = json.loads((CHECKPOINT / "trainer.json").read_text())
    assert final_state["step"] == END and (CHECKPOINT / "COMPLETE").exists()

    all_steps = np.array([row["step"] for row in full])
    visible = (all_steps >= START) & (all_steps <= END)
    steps = all_steps[visible]
    curves = {}
    for key in ("train_loss", "train_ce", "train_fm", "update_seconds"):
        values = np.array([row[key] for row in full])
        is_time = key == "update_seconds"
        boundaries = (DATA_SWITCH, *RESUMES) if is_time else (DATA_SWITCH,)
        smoothed = trailing_mean(all_steps, values, 10 if is_time else 50, boundaries)
        curves[key] = (values[visible], smoothed[visible])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 18, "axes.labelsize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#303030", "axes.linewidth": 1.1,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5))
    fig.subplots_adjust(left=0.066, right=0.982, top=0.855, bottom=0.13, wspace=0.125, hspace=0.36)
    fig.suptitle("BetaNav | 8 GPUs, global batch 64 | completed 29,589 / 29,589",
                 y=0.97, fontsize=22, weight="bold")
    fig.text(0.5, 0.926, "Training history from step 2,000 to completion", ha="center", fontsize=18)

    panels = (
        ("train_loss", "Total training loss: CE + 2 × FM", "#2166b8", "Loss", (0.8, 2.82)),
        ("train_ce", "Token cross entropy", "#188570", "Loss", (-0.025, 1.30)),
        ("train_fm", "Flow matching loss", "#ce7b19", "Loss", (0.24, 1.15)),
        ("update_seconds", "Sampled update time", "#6d50ae", "Seconds", (2.2, 11.8)),
    )
    ticks = [2000, 6000, 10000, 14000, 18000, 22000, 26000, 29589]
    for ax, (key, title, color, ylabel, ylim) in zip(axes.flat, panels):
        raw_values, smooth_values = curves[key]
        # The full-data migration changes the training population. Do not join curves across it.
        boundaries = (DATA_SWITCH, *RESUMES) if key == "update_seconds" else (DATA_SWITCH,)
        groups = np.searchsorted(np.array(boundaries), steps, side="left")
        for group in np.unique(groups):
            mask = groups == group
            ax.plot(steps[mask], raw_values[mask], color=color, alpha=0.19, lw=0.65, zorder=2)
            ax.plot(steps[mask], smooth_values[mask], color=color, lw=2.7, zorder=3)
        ax.set(title=title, xlabel="Global step", ylabel=ylabel, xlim=(START, END), ylim=ylim)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, color="#e5e5e5", lw=0.9, alpha=0.65)
        ax.set_axisbelow(True)
        n = 10 if key == "update_seconds" else 50
        handles = [
            Line2D([], [], color=color, alpha=0.25, lw=0.8, label="Logged batch (every 10 steps)"),
            Line2D([], [], color=color, lw=2.7, label=f"{n}-point moving average"),
        ]
        ax.legend(handles=handles, frameon=False, loc="upper left" if key in
                  ("train_fm", "update_seconds") else "upper right", fontsize=11)

    axes[0, 0].axvline(DATA_SWITCH, color="#8b8b8b", ls="--", lw=1.35, zorder=1)
    axes[0, 0].annotate("Full-data switch\n3,047", xy=(DATA_SWITCH, 2.38), xytext=(5900, 2.65),
                        ha="left", fontsize=10, color="#555555",
                        arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.9})
    fm_ax = axes[1, 0]
    for boundary in EPOCH_BOUNDARIES:
        fm_ax.axvline(boundary, color="#999999", ls=":", lw=1.7, zorder=1)
        fm_ax.text(boundary + 180, 1.09, f"Epoch boundary\n{boundary:,}", fontsize=10,
                   color="#555555", va="top")
    time_ax = axes[1, 1]
    for boundary, label_y in ((18000, 6.0), (26000, 7.5)):
        time_ax.axvline(boundary, color="#999999", ls="--", lw=1.25, zorder=1)
        time_ax.text(boundary + 220, label_y, f"Resume\n{boundary:,}", fontsize=10,
                     color="#555555", va="top")
    fig.text(0.066, 0.061,
             "Loss smoothing: trailing 50 logged batches (~500 steps). Time smoothing: 10 batches (~100 steps). "
             "Earlier history is used at the left edge.", fontsize=10, color="#555555")
    fig.text(0.066, 0.036,
             "Final checkpoint: 29,589. Last logged loss: 29,580 (no extrapolation). "
             "Duplicate resume records removed; update times exclude restart downtime.",
             fontsize=10, color="#555555")
    stem = ROOT / "training_loss_002000_029589"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=200, facecolor="white")
    plt.close(fig)

    fields = ["step", "time_unix", "train_loss", "train_ce", "train_fm", "update_seconds",
              "loss_ma50", "ce_ma50", "fm_ma50", "update_seconds_ma10", "lr_vlm", "lr_fm",
              "epoch", "batch_offset", "samples", "samples_seen", "source_line"]
    with (ROOT / "metrics_002000_029580.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(selected):
            out = {key: row[key] for key in fields if key in row}
            for key, column in (("train_loss", "loss_ma50"), ("train_ce", "ce_ma50"),
                                ("train_fm", "fm_ma50"), ("update_seconds", "update_seconds_ma10")):
                out[column] = float(curves[key][1][i])
            writer.writerow(out)
    raw_in_range = sum(START <= row["step"] <= END for row in train_rows)
    summary = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "display_range": [START, END], "actual_logged_range": [int(steps[0]), int(steps[-1])],
        "unique_training_points": len(selected), "raw_records_in_range": raw_in_range,
        "superseded_duplicates_removed": raw_in_range - len(selected),
        "missing_expected_points": [], "selected_timestamps_strictly_increase": True,
        "deduplication": "Last recorded value per global step; monotonic timestamps verified. "
                         "Retains the final resumed branch at 18,000 and 26,000.",
        "smoothing": {"loss": "Trailing 50 points; reset at full-data switch 3047.",
                      "time": "Trailing 10 points; reset at 3047, 18000, 26000.",
                      "left_edge": "Retained history before 2000 supplies the full moving-average window."},
        "data_switch": DATA_SWITCH, "full_data_epoch_boundaries": EPOCH_BOUNDARIES,
        "final_checkpoint": str(CHECKPOINT), "final_state": final_state,
        "no_final_loss_extrapolation": True,
        "last_100_logged_points_mean": {key: float(np.mean(curves[key][0][-100:])) for key in curves},
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
