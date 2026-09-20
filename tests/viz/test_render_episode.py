"""Offline rendering of recorded episodes: ``render_episode_dir`` (frame counts on both
timebases, skipped steps, overwrite / min_steps / height), episode discovery and record
loading, and the ``lightnav-render`` CLI. Needs the ``video`` extra (cv2 + imageio-ffmpeg)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from lightnav.cli import render as render_cli
from lightnav.viz import EpisodeRecorder, encode_jpeg_bytes
from lightnav.viz.render_episode import (
    find_episode_dirs,
    load_manifest,
    load_records,
    render_episode_dir,
)

pytest.importorskip("cv2")
pytest.importorskip("imageio_ffmpeg")

T0 = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
H, W = 36, 64
# Gaps before steps 0/1/2 -> at 10 fps realtime: 1 + 3 + 1 = 5 frames; per_step: 3.
DTS_MS = (0.0, 300.0, 100.0)


def _rgb(step: int) -> np.ndarray:
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 0] = 40 + 60 * step
    rgb[..., 2] = 120
    return rgb


def _record_episode(root: Path, *, timeline: str = "realtime", fps: int = 10,
                    n_steps: int = 3, label: str = "robot") -> Path:
    """Record ``n_steps`` steps with the DTS_MS gaps; return the episode directory."""
    rec = EpisodeRecorder(root, task="tracking", model_path="/path/to/model", hfov_deg=112.0,
                          cam_height=0.5, video_fps=fps, timeline=timeline, run_name="run_test")
    conn = rec.begin_connection(label)
    conn.begin_episode()
    t = T0
    for step in range(n_steps):
        t += timedelta(milliseconds=DTS_MS[step % len(DTS_MS)])
        conn.record_step(
            step=step,
            seq=step,
            image=encode_jpeg_bytes(_rgb(step)),
            instruction="follow the person",
            waypoints=np.column_stack([np.linspace(0.3, 2.0, 10), np.zeros(10), np.zeros(10)]),
            stop=step == n_steps - 1,
            visible=True,
            raw_text="<apos_650><opos_114>",
            latency_ms=8.0,
            pointing={"mode": "grid", "frame_size": [W, H], "apos_px": [32.0, 18.0],
                      "opos_px": [10.0, 30.0], "apos_state": "point", "opos_state": "point"},
            received_at=t,
        )
    rec.close()
    return rec.run_dir / label / "episode_000"


def _frames(path: Path) -> list[tuple[int, ...]]:
    import imageio.v2 as imageio

    with imageio.get_reader(str(path)) as reader:
        return [frame.shape for frame in reader]


def _leftovers(episode_dir: Path) -> list[str]:
    return sorted(p.name for p in episode_dir.iterdir() if ".partial" in p.name or p.suffix == ".tmp")


# -- render_episode_dir ----------------------------------------------------------------------


def test_realtime_timeline_repeats_frames_by_step_duration(tmp_path, capsys):
    ep = _record_episode(tmp_path)

    assert render_episode_dir(ep) is True

    out = ep / "traj_pointing.mp4"
    assert out.is_file()
    shapes = _frames(out)
    assert len(shapes) == 5
    assert shapes[0] == (H, W, 3)
    assert _leftovers(ep) == []
    summary = capsys.readouterr().out
    assert "steps=3/3" in summary and "frames=5" in summary and "timeline=realtime" in summary
    assert "fwd=auto" in summary and "hud(dt=0.1s)" in summary


def test_per_step_timeline_from_the_manifest_writes_one_frame_per_step(tmp_path):
    ep = _record_episode(tmp_path, timeline="per_step")
    assert json.loads((ep / "manifest.json").read_text())["video_timeline"] == "per_step"

    assert render_episode_dir(ep) is True
    assert len(_frames(ep / "traj_pointing.mp4")) == 3


def test_timeline_and_fps_overrides_beat_the_manifest(tmp_path, capsys):
    ep = _record_episode(tmp_path)

    assert render_episode_dir(ep, timeline="per_step", out_name="steps.mp4") is True
    assert len(_frames(ep / "steps.mp4")) == 3

    # 20 fps realtime: gaps 0/300/100 ms -> 1 + 6 + 2 frames.
    assert render_episode_dir(ep, fps=20, out_name="fast.mp4") is True
    assert len(_frames(ep / "fast.mp4")) == 9
    assert "fps=20" in capsys.readouterr().out


def test_missing_or_undecodable_image_skips_the_step_and_reports_it(tmp_path, capsys):
    ep = _record_episode(tmp_path)
    (ep / "image_000001.jpg").unlink()
    (ep / "image_000002.jpg").write_bytes(b"not a jpeg")

    assert render_episode_dir(ep) is True

    assert len(_frames(ep / "traj_pointing.mp4")) == 1  # only step 0 remains
    captured = capsys.readouterr()
    assert "steps=1/3" in captured.out
    assert "2 step(s) had no usable image, skipped: 1, 2" in captured.err
    assert "undecodable image_000002.jpg" in captured.err


def test_nothing_renderable_returns_false_and_leaves_no_file(tmp_path, capsys):
    ep = _record_episode(tmp_path)
    for img in ep.glob("*.jpg"):
        img.unlink()
    assert render_episode_dir(ep) is False
    assert not (ep / "traj_pointing.mp4").exists()
    assert _leftovers(ep) == []
    assert "nothing written" in capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "actions.json").write_text("[]")
    assert render_episode_dir(empty) is False


def test_existing_output_is_kept_unless_overwrite(tmp_path, capsys):
    ep = _record_episode(tmp_path)
    out = ep / "traj_pointing.mp4"
    out.write_bytes(b"stale")

    assert render_episode_dir(ep) is True
    assert out.read_bytes() == b"stale"
    assert "exists, skipped" in capsys.readouterr().out

    assert render_episode_dir(ep, overwrite=True) is True
    assert len(_frames(out)) == 5


def test_min_steps_skips_short_episodes_as_a_success(tmp_path, capsys):
    ep = _record_episode(tmp_path)
    assert render_episode_dir(ep, min_steps=4) is True
    assert not (ep / "traj_pointing.mp4").exists()
    assert "< min_steps 4" in capsys.readouterr().out
    assert render_episode_dir(ep, min_steps=3) is True
    assert (ep / "traj_pointing.mp4").exists()


def test_height_resamples_every_frame(tmp_path, capsys):
    ep = _record_episode(tmp_path, timeline="per_step")
    assert render_episode_dir(ep, height=72) is True
    shapes = _frames(ep / "traj_pointing.mp4")
    assert shapes == [(72, 128, 3)] * 3
    assert "72p" in capsys.readouterr().out


def test_odd_frame_sizes_are_padded_for_the_encoder(tmp_path):
    ep = _record_episode(tmp_path, timeline="per_step")
    odd = np.zeros((35, 63, 3), np.uint8)
    for step in range(3):
        (ep / f"image_{step:06d}.jpg").write_bytes(encode_jpeg_bytes(odd))
    assert render_episode_dir(ep) is True
    assert _frames(ep / "traj_pointing.mp4")[0] == (36, 64, 3)


def test_render_options_forward_offset_no_hud_no_pointing(tmp_path, capsys):
    ep = _record_episode(tmp_path, timeline="per_step")
    assert render_episode_dir(ep, forward_offset=0.0, hud=False, pointing=False,
                              dt_s=0.25, traj_width=0.4, out_name="bare.mp4") is True
    line = capsys.readouterr().out
    assert "fwd=0.00m" in line and "hud(" not in line
    assert len(_frames(ep / "bare.mp4")) == 3


def test_manifest_defaults_apply_when_the_manifest_is_missing(tmp_path, capsys):
    ep = _record_episode(tmp_path)
    (ep / "manifest.json").unlink()
    assert render_episode_dir(ep) is True
    # 10 fps / realtime / dt 0.1 fallbacks
    assert "fps=10" in capsys.readouterr().out
    assert len(_frames(ep / "traj_pointing.mp4")) == 5


def test_unfinished_episode_renders_from_the_jsonl(tmp_path):
    ep = _record_episode(tmp_path, timeline="per_step")
    records = json.loads((ep / "actions.json").read_text())
    (ep / "actions.json").unlink()
    (ep / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    assert render_episode_dir(ep) is True
    assert len(_frames(ep / "traj_pointing.mp4")) == 3


# -- discovery and loading -------------------------------------------------------------------


def test_find_episode_dirs_accepts_roots_and_episode_dirs_dedupes_and_sorts(tmp_path, capsys):
    ep_a = _record_episode(tmp_path / "a", label="zeta")
    ep_b = _record_episode(tmp_path / "a", label="alpha")
    unfinished = tmp_path / "b" / "run_x" / "conn001" / "episode_000"
    unfinished.mkdir(parents=True)
    (unfinished / "actions.jsonl").write_text("{}\n")
    (tmp_path / "noise").mkdir()

    found = find_episode_dirs([tmp_path / "a", ep_a, tmp_path / "b", tmp_path / "noise",
                               tmp_path / "missing"])

    assert found == sorted([ep_b, ep_a, unfinished])
    assert "not found" in capsys.readouterr().err
    assert find_episode_dirs([]) == []


def test_load_records_reads_arrays_and_jsonl_and_skips_junk(tmp_path):
    ep = tmp_path / "ep"
    ep.mkdir()
    assert load_records(ep) == []

    (ep / "actions.jsonl").write_text('{"step": 0}\nnot json\n\n[1, 2]\n{"step": 1}\n')
    assert load_records(ep) == [{"step": 0}, {"step": 1}]

    (ep / "actions.json").write_text('[{"step": 5}, "junk", null]')
    assert load_records(ep) == [{"step": 5}]  # actions.json wins over the jsonl

    (ep / "actions.json").write_text("{not json")
    assert load_records(ep) == []
    (ep / "actions.json").write_text('{"step": 5}')
    assert load_records(ep) == []


def test_load_manifest_falls_back_to_empty(tmp_path, capsys):
    ep = tmp_path / "ep"
    ep.mkdir()
    assert load_manifest(ep) == {}
    (ep / "manifest.json").write_text("{bad")
    assert load_manifest(ep) == {}
    assert "unreadable manifest" in capsys.readouterr().err
    (ep / "manifest.json").write_text("[1]")
    assert load_manifest(ep) == {}
    (ep / "manifest.json").write_text('{"video_fps": 7}')
    assert load_manifest(ep) == {"video_fps": 7}


# -- CLI -------------------------------------------------------------------------------------


def test_cli_renders_every_episode_under_a_tree(tmp_path, capsys):
    ep_a = _record_episode(tmp_path, label="a")
    ep_b = _record_episode(tmp_path, label="b", timeline="per_step")

    assert render_cli.main([str(tmp_path)]) == 0

    assert len(_frames(ep_a / "traj_pointing.mp4")) == 5
    assert len(_frames(ep_b / "traj_pointing.mp4")) == 3
    out = capsys.readouterr().out
    assert "2 episode(s)" in out and "2/2 episode(s) rendered" in out


def test_cli_returns_one_for_a_bad_path(tmp_path, capsys):
    assert render_cli.main([str(tmp_path / "does-not-exist")]) == 1
    assert "no episodes found" in capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    assert render_cli.main([str(empty)]) == 1


def test_cli_returns_one_when_an_episode_cannot_be_rendered(tmp_path, capsys):
    good = _record_episode(tmp_path / "good")
    bad = _record_episode(tmp_path / "bad")
    for img in bad.glob("*.jpg"):
        img.unlink()

    assert render_cli.main([str(good), str(bad)]) == 1
    assert (good / "traj_pointing.mp4").exists()
    assert not (bad / "traj_pointing.mp4").exists()
    assert "1/2 episode(s) rendered" in capsys.readouterr().out


def test_cli_height_and_out_name_control_the_output(tmp_path):
    ep = _record_episode(tmp_path, timeline="per_step")
    argv = [str(ep), "--height", "72", "--out-name", "tall.mp4", "--fps", "20",
            "--timeline", "realtime", "--forward-offset", "0", "--no-hud", "--no-pointing",
            "--dt", "0.2", "--traj-width", "0.3", "--min-steps", "1"]
    assert render_cli.main(argv) == 0
    shapes = _frames(ep / "tall.mp4")
    assert shapes == [(72, 128, 3)] * 9  # 20 fps realtime over 0/300/100 ms gaps


def test_cli_overwrite_flag(tmp_path, capsys):
    ep = _record_episode(tmp_path, timeline="per_step")
    (ep / "traj_pointing.mp4").write_bytes(b"stale")
    assert render_cli.main([str(ep)]) == 0
    assert (ep / "traj_pointing.mp4").read_bytes() == b"stale"
    assert render_cli.main([str(ep), "--overwrite"]) == 0
    assert len(_frames(ep / "traj_pointing.mp4")) == 3


def test_cli_rejects_bad_arguments():
    with pytest.raises(SystemExit) as exc:
        render_cli.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        render_cli.main(["x", "--forward-offset", "near"])
    with pytest.raises(SystemExit):
        render_cli.main(["x", "--timeline", "wallclock"])


def test_cli_parser_defaults():
    args = render_cli.build_parser().parse_args(["some/dir"])
    assert args.paths == [Path("some/dir")]
    assert args.out_name == "traj_pointing.mp4"
    assert args.fps is None and args.timeline is None and args.dt is None
    assert args.height == 0 and args.forward_offset == "auto" and args.min_steps == 0
    assert args.traj_width == 0.25
    assert not args.no_pointing and not args.no_hud and not args.overwrite
    assert render_cli.build_parser().parse_args(["d", "--forward-offset", "0.5"]).forward_offset == 0.5


def test_cli_reports_a_missing_video_dependency_as_exit_one(tmp_path, monkeypatch, capsys):
    import sys

    ep = _record_episode(tmp_path, timeline="per_step")
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert render_cli.main([str(ep)]) == 1
    assert "lightnav[video]" in capsys.readouterr().err
    assert not (ep / "traj_pointing.mp4").exists()
    assert _leftovers(ep) == []
