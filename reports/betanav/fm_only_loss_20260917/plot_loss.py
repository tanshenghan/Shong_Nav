"""Plot the complete frozen-VLM/FM-only run from its immutable training logs."""

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
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
SOURCE = Path("/mnt/cfs/Models/beta_nav/r2r_fm_only_frozen_lightnav_20260916/metrics.jsonl")
END = 3699
EPOCH_BOUNDARIES = (1233, 2466)


def trailing_mean(values, window):
    return np.array([np.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))])


def main():
    raw = SOURCE.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    rows = [row for row in rows if "train_loss" in row]
    steps = np.array([r["step"] for r in rows])
    assert steps.tolist() == [1, *range(10, END + 1, 10)]
    assert all(b["time_unix"] > a["time_unix"] for a, b in zip(rows, rows[1:]))
    assert all(math.isfinite(r[k]) for r in rows for k in
               ("train_loss", "train_fm", "train_ce", "update_seconds"))
    assert all(r["train_loss"] == 2 * r["train_fm"] and r["train_ce"] == 0
               and r["grad_norm_vlm"] == 0 and r["ce_tokens"] == 0 for r in rows)
    checkpoint = SOURCE.parent / "checkpoints/step-00003699"
    state = json.loads((checkpoint / "trainer.json").read_text())
    assert (checkpoint / "COMPLETE").exists()
    assert state["step"] == END and state["epoch"] == 3 and state["batch_offset"] == 0
    assert state["vlm_updates"] == 0
    values = {k: np.array([r[k] for r in rows]) for k in ("train_loss", "train_fm", "update_seconds")}
    smoothed = {k: trailing_mean(v, 20) for k, v in values.items() if k != "update_seconds"}
    # Startup includes worker initialization. Show it in a note, not in the steady-update average.
    time_smooth = trailing_mean(values["update_seconds"][1:], 10)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 17, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "axes.linewidth": 1,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5))
    fig.subplots_adjust(left=.066, right=.979, top=.85, bottom=.135, hspace=.37, wspace=.16)
    fig.suptitle("BetaNav | Frozen VLM + FM only | 8 GPUs | global batch 512",
                 fontsize=22, weight="bold", y=.971)
    fig.text(.5, .925, "Completed 3,699 updates / 3 full epochs  |  Original LightNav + fresh FM expert",
             ha="center", fontsize=16)

    panels = [
        (axes[0, 0], "train_loss", "Total training loss: 2 × FM", "#2166b8", 0, (0, 7.15)),
        (axes[0, 1], "train_fm", "Flow matching loss", "#ce7b19", 0, (0, 3.58)),
        (axes[1, 0], "train_fm", "FM loss detail: step 500 onward", "#ce7b19", 500, (.40, .71)),
    ]
    for ax, key, title, color, start, ylim in panels:
        mask = steps >= start
        ax.plot(steps[mask], values[key][mask], color=color, alpha=.29, lw=.9, label="Logged batch")
        ax.plot(steps[mask], smoothed[key][mask], color=color, lw=2.7, label="20-point moving average")
        ax.set(title=title, xlabel="Global step", ylabel="Loss", xlim=(start, END), ylim=ylim)
        ax.legend(frameon=False, loc="upper right", fontsize=10.5)

    axes[0, 0].annotate("First logged loss: 6.824", xy=(1, values["train_loss"][0]),
                        xytext=(290, 5.95), fontsize=11, color="#3e566f",
                        arrowprops={"arrowstyle": "-", "color": "#8897a6", "lw": .9})
    axes[0, 1].text(.98, .60, "Last 20 logged batches\nFM mean: 0.495", transform=axes[0, 1].transAxes,
                    ha="right", fontsize=12, color="#825015")
    for boundary, label in zip(EPOCH_BOUNDARIES, ("Epoch 2", "Epoch 3")):
        axes[1, 0].text(boundary + 30, .704, label, va="top", fontsize=10, color="#686868")

    time_ax = axes[1, 1]
    time_ax.plot(steps[1:], values["update_seconds"][1:], color="#7353b4", alpha=.29, lw=.85,
                 label="Logged update (step 10 onward)")
    time_ax.plot(steps[1:], time_smooth, color="#7353b4", lw=2.5, label="10-point moving average")
    time_ax.set(title="Sampled update time after startup", xlabel="Global step", ylabel="Seconds",
                xlim=(0, END), ylim=(6.35, 8.6))
    time_ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    time_ax.text(.02, .94, "Startup step 1: 35.82 s\n(includes loader initialization)",
                 transform=time_ax.transAxes, va="top", fontsize=10, color="#65557d")
    for ax in axes.flat:
        for boundary in EPOCH_BOUNDARIES:
            ax.axvline(boundary, color="#919191", ls=":", lw=1.25, zorder=1)
        ticks = [0, 500, 1000, 1500, 2000, 2500, 3000, END]
        if ax is axes[1, 0]:
            ticks = ticks[1:]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, color="#e4e6e9", alpha=.75, lw=.8)
        ax.set_axisbelow(True)

    fig.text(.066, .073,
             "Thin lines: logged batches (step 1, then every 10 steps). Thick loss lines: trailing 20 points (~200 steps).",
             fontsize=10.5, color="#535353")
    fig.text(.066, .047,
             "Dotted lines: epoch boundaries at 1,233 and 2,466. Training loss only; CE is zero and the complete VLM is frozen.",
             fontsize=10.5, color="#535353")
    fig.text(.066, .021,
             "All 370 expected log points are present, covering steps 1 through 3,690. Final training checkpoint: step 3,699.",
             fontsize=10.5, color="#535353")
    stem = ROOT / "training_loss_000001_003699"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=180, facecolor="white")
    plt.close(fig)

    fields = ["step", "time_unix", "train_loss", "train_fm", "train_ce", "loss_ma20", "fm_ma20",
              "update_seconds", "update_seconds_ma10_excluding_startup", "lr_fm", "samples_seen", "epoch"]
    with (ROOT / "metrics_000001_003690.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            result = {k: row[k] for k in fields if k in row}
            result.update(loss_ma20=float(smoothed["train_loss"][i]), fm_ma20=float(smoothed["train_fm"][i]),
                          update_seconds_ma10_excluding_startup="" if i == 0 else float(time_smooth[i - 1]))
            writer.writerow(result)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "training_mode": "fm_only", "world_size": 8, "global_batch": 512,
        "logged_points": len(rows), "logged_step_range": [int(steps[0]), int(steps[-1])],
        "no_missing_or_duplicate_points": True, "no_nonfinite_losses": True,
        "final_checkpoint": str(checkpoint), "final_state": state,
        "epoch_boundaries": EPOCH_BOUNDARIES,
        "smoothing": "Trailing 20 logged batches for loss; trailing 10 for time, excluding startup step 1.",
        "first_recorded_loss": {k: float(v[0]) for k, v in values.items()},
        "last_recorded_loss": {k: float(v[-1]) for k, v in values.items()},
        "last_20_logged_means": {k: float(np.mean(v[-20:])) for k, v in values.items()},
        "epoch_fm_means_of_logged_batches": {
            str(i + 1): float(np.mean(values["train_fm"][(steps > start) & (steps <= end)]))
            for i, (start, end) in enumerate(zip((0, *EPOCH_BOUNDARIES), (*EPOCH_BOUNDARIES, END)))
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
