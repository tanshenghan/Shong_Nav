"""EpisodeRecorder / ConnectionRecorder: on-disk layout, manifest, per-step records,
the jsonl -> actions.json hand-over, step timing, and the promise that recording
failures never reach the caller. Needs numpy + Pillow only."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from lightnav.viz import ConnectionRecorder, EpisodeRecorder, decode_rgb_bytes, encode_jpeg_bytes
from lightnav.viz import recorder as recorder_mod
from lightnav.viz.render_episode import find_episode_dirs, load_records

T0 = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
CAM = dict(hfov_deg=112.0, cam_height=0.5)


def _rgb(h: int = 6, w: int = 8, value: int = 90) -> np.ndarray:
    rgb = np.full((h, w, 3), value, np.uint8)
    rgb[..., 1] = 30
    return rgb


def _recorder(root: Path, **overrides) -> EpisodeRecorder:
    fields = dict(task="tracking", model_path="/path/to/model", run_name="run_test", **CAM)
    fields.update(overrides)
    return EpisodeRecorder(root, **fields)


def _step_kwargs(step: int, *, dt_ms: float = 0.0, **overrides) -> dict:
    fields = dict(
        step=step,
        seq=step + 10,
        image=encode_jpeg_bytes(_rgb()),
        instruction="go to the sofa",
        waypoints=np.tile([[0.25, 0.0, 0.05]], (10, 1)).astype(np.float32),
        stop=False,
        visible=True,
        raw_text="<traj_7>",
        latency_ms=12.5,
        pointing={"mode": "grid", "frame_size": [8, 6], "apos_px": [4.0, 3.0],
                  "apos_state": "point", "opos_px": None, "opos_state": "none"},
        received_at=T0 + timedelta(milliseconds=dt_ms),
    )
    fields.update(overrides)
    return fields


RECORD_KEYS = {
    "step", "seq", "received_at", "step_dt_ms", "step_fps", "instruction", "waypoints",
    "stop", "visible", "raw_text", "latency_ms", "pointing", "frame_size",
}


# -- layout ----------------------------------------------------------------------------------


def test_run_dir_is_created_under_the_root_with_the_run_name(tmp_path):
    rec = _recorder(tmp_path / "rec")
    assert rec.run_dir == tmp_path / "rec" / "run_test"
    assert rec.run_dir.is_dir()

    auto = EpisodeRecorder(tmp_path / "auto", task="t", model_path="m", **CAM)
    assert auto.run_dir.parent == tmp_path / "auto"
    assert auto.run_dir.name.startswith("run_") and len(auto.run_dir.name) == len("run_YYYYmmdd_HHMMSS")


def test_recorder_rejects_an_unknown_timeline(tmp_path):
    with pytest.raises(ValueError, match="timeline"):
        _recorder(tmp_path, timeline="wallclock")


def test_episode_layout_manifest_images_and_records(tmp_path):
    rec = _recorder(tmp_path, video_fps=15, timeline="realtime", waypoint_dt_s=0.2,
                    forward_offset=None, extra={"note": "unit test"})
    conn = rec.begin_connection("robot-1")
    assert isinstance(conn, ConnectionRecorder)
    assert conn.label == "robot-1" and conn.dir == rec.run_dir / "robot-1"
    assert conn.dir.is_dir(), "the connection directory is claimed eagerly"

    conn.begin_episode()
    assert conn.episode_open and conn.episode_dir is None, "the episode directory appears with the first step"
    assert list(conn.dir.iterdir()) == []

    raw_jpeg = encode_jpeg_bytes(_rgb(value=200), quality=60)
    conn.record_step(**_step_kwargs(0, instruction="", image=raw_jpeg))
    conn.record_step(**_step_kwargs(1, dt_ms=100.0, image=_rgb(value=50)))
    conn.record_step(**_step_kwargs(2, dt_ms=250.0, waypoints=None, stop=True, visible=None,
                                    pointing=None, instruction="go to the sofa "))

    ep = conn.episode_dir
    assert ep == rec.run_dir / "robot-1" / "episode_000"
    assert conn.steps_recorded == 3
    assert sorted(p.name for p in ep.iterdir()) == [
        "actions.jsonl", "image_000000.jpg", "image_000001.jpg", "image_000002.jpg", "manifest.json",
    ]
    # Client bytes are stored verbatim; arrays are JPEG-encoded.
    assert (ep / "image_000000.jpg").read_bytes() == raw_jpeg
    assert decode_rgb_bytes((ep / "image_000001.jpg").read_bytes()).shape == (6, 8, 3)

    manifest = json.loads((ep / "manifest.json").read_text())
    assert manifest == {
        "schema": 1,
        "created_at": manifest["created_at"],
        "conn": "robot-1",
        "episode": 0,
        "task": "tracking",
        "model_path": "/path/to/model",
        "video_fps": 15,
        "video_timeline": "realtime",
        "waypoint_dt_s": 0.2,
        "overlay_hfov_deg": 112.0,
        "overlay_cam_height": 0.5,
        "overlay_forward_offset": None,
        "frame_size": [8, 6],
        "instruction": "go to the sofa",  # first NON-EMPTY instruction
        "extra": {"note": "unit test"},
    }
    datetime.fromisoformat(manifest["created_at"])  # ISO 8601

    lines = [json.loads(line) for line in (ep / "actions.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [0, 1, 2]

    conn.end_episode()
    assert not (ep / "actions.jsonl").exists()
    assert not (ep / "actions.json.tmp").exists()
    records = json.loads((ep / "actions.json").read_text())
    assert records == lines
    assert load_records(ep) == records

    r0, r1, r2 = records
    assert set(r0) == RECORD_KEYS
    assert r0["seq"] == 10 and r1["seq"] == 11
    assert [r["step_dt_ms"] for r in records] == [0.0, 100.0, 150.0]
    assert r0["step_fps"] is None
    assert r1["step_fps"] == pytest.approx(10.0)
    assert r2["step_fps"] == pytest.approx(1000.0 / 150.0)
    assert r0["received_at"] == "2026-08-28T12:00:00.000+00:00"
    assert r1["waypoints"] == [[0.25, 0.0, pytest.approx(0.05)]] * 10
    assert r2["waypoints"] is None and r2["stop"] is True and r2["visible"] is None
    assert r0["visible"] is True and r0["stop"] is False
    assert r0["instruction"] == "" and r2["instruction"] == "go to the sofa "
    assert r0["raw_text"] == "<traj_7>" and r0["latency_ms"] == 12.5
    assert r0["pointing"]["apos_px"] == [4.0, 3.0] and r2["pointing"] is None
    assert r0["frame_size"] == [8, 6] and r1["frame_size"] == [8, 6]

    assert find_episode_dirs([tmp_path]) == [ep]


def test_extra_record_keys_and_numpy_scalars_are_json_safe(tmp_path):
    conn = _recorder(tmp_path).begin_connection()
    conn.record_step(**_step_kwargs(0, latency_ms=np.float32(3.5), seq=np.int64(4)),
                     episode_id="episode_007", score=np.float32(0.25), tag=Path("x"))
    conn.end_episode()

    (rec,) = load_records(conn.dir / "episode_000")
    assert rec["episode_id"] == "episode_007"
    assert rec["score"] == pytest.approx(0.25) and rec["tag"] == "x"
    assert rec["latency_ms"] == 3.5 and rec["seq"] == 4
    assert set(rec) == RECORD_KEYS | {"episode_id", "score", "tag"}


def test_received_at_accepts_iso_strings_epochs_and_defaults_to_now(tmp_path):
    conn = _recorder(tmp_path).begin_connection()
    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    conn.record_step(**_step_kwargs(0, received_at=None))
    conn.record_step(**_step_kwargs(1, received_at=T0.isoformat()))
    conn.record_step(**_step_kwargs(2, received_at=(T0 + timedelta(seconds=2)).timestamp()))
    conn.record_step(**_step_kwargs(3, received_at=T0))  # clock went backwards: clamp, no crash
    conn.end_episode()

    r0, r1, r2, r3 = load_records(conn.dir / "episode_000")
    assert datetime.fromisoformat(r0["received_at"]) >= before
    assert r1["received_at"].startswith("2026-08-28T12:00:00")
    assert r2["step_dt_ms"] == 2000.0 and r2["step_fps"] == pytest.approx(0.5)
    assert r3["step_dt_ms"] == 0.0 and r3["step_fps"] is None


def test_per_step_timeline_still_records_step_timing(tmp_path):
    conn = _recorder(tmp_path, timeline="per_step").begin_connection()
    conn.record_step(**_step_kwargs(0))
    conn.record_step(**_step_kwargs(1, dt_ms=300.0))
    conn.end_episode()

    ep = conn.dir / "episode_000"
    assert json.loads((ep / "manifest.json").read_text())["video_timeline"] == "per_step"
    assert [r["step_dt_ms"] for r in load_records(ep)] == [0.0, 300.0]


def test_save_images_false_records_steps_without_frames(tmp_path):
    conn = _recorder(tmp_path, save_images=False).begin_connection()
    conn.record_step(**_step_kwargs(0))
    conn.end_episode()

    ep = conn.dir / "episode_000"
    assert not list(ep.glob("*.jpg"))
    (rec,) = load_records(ep)
    assert rec["frame_size"] == [8, 6]  # size is still known from the bytes


def test_image_none_is_allowed(tmp_path):
    conn = _recorder(tmp_path).begin_connection()
    conn.record_step(**_step_kwargs(0, image=None))
    conn.end_episode()
    (rec,) = load_records(conn.dir / "episode_000")
    assert rec["frame_size"] is None
    assert json.loads((conn.dir / "episode_000" / "manifest.json").read_text())["frame_size"] is None


# -- connections and episodes ----------------------------------------------------------------


def test_connection_labels_default_sanitise_and_deduplicate(tmp_path):
    rec = _recorder(tmp_path)
    assert rec.begin_connection().label == "conn001"
    assert rec.begin_connection(None).label == "conn002"
    assert rec.begin_connection("").label == "conn003"
    assert rec.begin_connection("robot 7/../x").label == "robot_7_.._x"
    assert rec.begin_connection("robot 7/../x").label == "robot_7_.._x_2"
    assert rec.begin_connection("robot 7/../x").label == "robot_7_.._x_3"
    assert rec.begin_connection("---").label == "conn007"  # nothing safe survives


def test_begin_episode_numbers_episodes_and_finishes_the_open_one(tmp_path):
    conn = _recorder(tmp_path).begin_connection("c")
    conn.begin_episode()
    conn.record_step(**_step_kwargs(0))
    conn.begin_episode()  # ends episode_000 first
    first = conn.dir / "episode_000"
    assert (first / "actions.json").exists() and not (first / "actions.jsonl").exists()
    assert conn.episode_dir is None and conn.episode_open

    conn.record_step(**_step_kwargs(0))
    conn.record_step(**_step_kwargs(1, dt_ms=50.0))
    assert conn.episode_dir == conn.dir / "episode_001"
    conn.end_episode()

    assert sorted(p.name for p in conn.dir.iterdir()) == ["episode_000", "episode_001"]
    assert len(load_records(conn.dir / "episode_001")) == 2
    # The second episode's timing does not carry over from the first.
    assert load_records(conn.dir / "episode_001")[0]["step_dt_ms"] == 0.0


def test_episodes_without_steps_leave_no_episode_directory(tmp_path):
    rec = _recorder(tmp_path)
    conn = rec.begin_connection("idle")
    conn.begin_episode()
    conn.begin_episode()
    conn.end_episode()
    conn.end_episode()  # twice is fine
    conn.close()
    assert conn.dir.is_dir()  # the connection dir is claimed up front ...
    assert list(conn.dir.iterdir()) == []  # ... but no episode_* dir was ever created
    assert list(rec.run_dir.iterdir()) == [conn.dir]


def test_same_run_name_in_one_root_gets_exclusive_run_dirs(tmp_path):
    first = _recorder(tmp_path)
    second = _recorder(tmp_path)
    third = _recorder(tmp_path)

    assert first.run_dir == tmp_path / "run_test"
    assert second.run_dir == tmp_path / "run_test_2"
    assert third.run_dir == tmp_path / "run_test_3"
    assert all(r.run_dir.is_dir() for r in (first, second, third))

    # The same label on two recorders lands in two distinct, already-existing directories.
    a, b = first.begin_connection("robot"), second.begin_connection("robot")
    assert a.dir == tmp_path / "run_test" / "robot" and a.dir.is_dir()
    assert b.dir == tmp_path / "run_test_2" / "robot" and b.dir.is_dir()
    assert a.dir != b.dir

    a.record_step(**_step_kwargs(0))
    b.record_step(**_step_kwargs(0))
    for rec in (first, second, third):
        rec.close()
    assert sorted(e.relative_to(tmp_path) for e in find_episode_dirs([tmp_path])) == [
        Path("run_test/robot/episode_000"),
        Path("run_test_2/robot/episode_000"),
    ]


def test_a_stale_connection_dir_on_disk_is_never_reused(tmp_path):
    rec = _recorder(tmp_path)
    (rec.run_dir / "robot").mkdir()  # left behind by another process sharing the run dir
    conn = rec.begin_connection("robot")
    assert conn.label == "robot_2" and conn.dir == rec.run_dir / "robot_2" and conn.dir.is_dir()


def test_record_step_without_begin_episode_opens_one(tmp_path):
    conn = _recorder(tmp_path).begin_connection("c")
    conn.record_step(**_step_kwargs(0))
    assert conn.episode_dir == conn.dir / "episode_000"
    assert (conn.episode_dir / "actions.jsonl").exists()


def test_jsonl_is_flushed_per_step_so_an_unfinished_episode_is_readable(tmp_path):
    conn = _recorder(tmp_path).begin_connection("c")
    conn.record_step(**_step_kwargs(0))
    conn.record_step(**_step_kwargs(1, dt_ms=100.0))

    ep = conn.episode_dir
    assert not (ep / "actions.json").exists()
    assert [r["step"] for r in load_records(ep)] == [0, 1]
    assert find_episode_dirs([tmp_path]) == [ep]


def test_close_ends_the_episode_and_further_steps_are_ignored(tmp_path):
    rec = _recorder(tmp_path)
    conn = rec.begin_connection("c")
    conn.record_step(**_step_kwargs(0))
    conn.close()

    ep = conn.dir / "episode_000"
    assert (ep / "actions.json").exists() and not (ep / "actions.jsonl").exists()
    conn.record_step(**_step_kwargs(1))
    conn.begin_episode()
    conn.close()  # idempotent
    assert sorted(p.name for p in conn.dir.iterdir()) == ["episode_000"]
    assert not conn.episode_open


def test_recorder_close_closes_every_connection(tmp_path):
    rec = _recorder(tmp_path)
    a, b = rec.begin_connection("a"), rec.begin_connection("b")
    a.record_step(**_step_kwargs(0))
    b.record_step(**_step_kwargs(0))
    b.record_step(**_step_kwargs(1, dt_ms=100.0))
    rec.close()
    rec.close()

    assert len(load_records(rec.run_dir / "a" / "episode_000")) == 1
    assert len(load_records(rec.run_dir / "b" / "episode_000")) == 2
    assert not list(rec.run_dir.rglob("actions.jsonl"))
    # A connection label can be reused once its recorder is gone.
    assert rec.begin_connection("a").label == "a_2"  # the directory still exists on disk


def test_context_managers_close(tmp_path):
    with _recorder(tmp_path) as rec:
        with rec.begin_connection("c") as conn:
            conn.record_step(**_step_kwargs(0))
    assert (rec.run_dir / "c" / "episode_000" / "actions.json").exists()


# -- failures never propagate ----------------------------------------------------------------


@pytest.mark.parametrize("make_root", [
    pytest.param(lambda tmp: tmp / "not-a-dir", id="root-is-a-file"),
    pytest.param(lambda tmp: tmp / "not-a-dir" / "nested" / "rec", id="parent-is-a-file"),
])
def test_unusable_root_is_a_startup_error(tmp_path, make_root):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where the root should be")

    with pytest.raises(OSError):
        EpisodeRecorder(make_root(tmp_path), task="t", model_path="m", **CAM)
    assert blocker.is_file() and blocker.read_text() == "a file where the root should be"


def test_per_step_write_failures_are_swallowed_and_logged(tmp_path, caplog):
    rec = _recorder(tmp_path)
    conn = rec.begin_connection("robot")
    # Sabotage the claimed connection directory: the episode dir can no longer be created.
    conn.dir.rmdir()
    conn.dir.write_text("not a directory any more")

    with caplog.at_level(logging.WARNING, logger="lightnav.viz.recorder"):
        conn.begin_episode()
        conn.record_step(**_step_kwargs(0))
        conn.record_step(**_step_kwargs(1, dt_ms=100.0))
        conn.end_episode()
        conn.close()
        rec.close()

    assert conn.steps_recorded == 0
    assert conn.dir.is_file()
    failures = [r for r in caplog.records if "record_step" in r.message and "failed" in r.message]
    assert len(failures) == 2
    assert not list(rec.run_dir.rglob("actions.json*"))


def test_connection_dir_failure_is_logged_and_recording_continues_best_effort(tmp_path, caplog):
    rec = _recorder(tmp_path)
    rec.run_dir.rmdir()
    rec.run_dir.write_text("run dir replaced by a file")

    with caplog.at_level(logging.WARNING, logger="lightnav.viz.recorder"):
        conn = rec.begin_connection("robot")
        conn.record_step(**_step_kwargs(0))
        rec.close()

    assert conn.label == "robot" and conn.steps_recorded == 0
    assert any("cannot create connection dir" in r.message for r in caplog.records)
    assert any("record_step" in r.message for r in caplog.records)


def test_end_episode_write_failure_is_swallowed(tmp_path, monkeypatch, caplog):
    conn = _recorder(tmp_path).begin_connection("c")
    conn.record_step(**_step_kwargs(0))

    def boom(path, obj, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(recorder_mod, "_write_json_atomic", boom)
    with caplog.at_level(logging.WARNING, logger="lightnav.viz.recorder"):
        conn.end_episode()
    assert any("end_episode failed" in r.message for r in caplog.records)
    assert not conn.episode_open


def test_unencodable_image_only_drops_the_frame(tmp_path, caplog):
    conn = _recorder(tmp_path).begin_connection("c")
    with caplog.at_level(logging.WARNING, logger="lightnav.viz.recorder"):
        conn.record_step(**_step_kwargs(0, image=np.zeros((2, 2, 7))))  # not an image layout
    conn.end_episode()

    ep = conn.dir / "episode_000"
    assert not list(ep.glob("*.jpg"))
    assert len(load_records(ep)) == 1
    assert any("not written" in r.message for r in caplog.records)


@pytest.mark.parametrize("bad_waypoints", ["junk", [["a", "b"]], np.zeros(3), np.zeros((0, 3))])
def test_unusable_waypoints_are_recorded_as_null(tmp_path, bad_waypoints):
    conn = _recorder(tmp_path).begin_connection("c")
    conn.record_step(**_step_kwargs(0, waypoints=bad_waypoints))
    conn.end_episode()
    (rec,) = load_records(conn.dir / "episode_000")
    assert rec["waypoints"] is None
