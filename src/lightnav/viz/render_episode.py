"""Render recorded episode directories (see :mod:`lightnav.viz.recorder`) to mp4."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from lightnav.viz.render import DEFAULT_WAYPOINT_DT_S, TRAJ_WIDTH_FRAC, render_frame
from lightnav.viz.video import (
    decode_rgb_bytes,
    open_video_writer,
    pad_to_even_dimensions,
    step_repeats,
    upscale_to_height,
)

IMAGE_PREFIX = "image_"
DEFAULT_OUT_NAME = "traj_pointing.mp4"
DEFAULT_VIDEO_FPS = 10
# Used only when a manifest is missing or predates a key.
FALLBACK_HFOV_DEG = 90.0
FALLBACK_CAM_HEIGHT = 0.5
FALLBACK_FORWARD_OFFSET = 0.0


def _is_episode_dir(path: Path) -> bool:
    return (path / "actions.json").is_file() or (path / "actions.jsonl").is_file()


def find_episode_dirs(roots: list[Path]) -> list[Path]:
    """Episode directories (holding ``actions.json`` or an unfinished ``actions.jsonl``).

    Each root may be an episode directory itself or any ancestor of one. The
    result is deduplicated and sorted.
    """
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"  ! not found: {root}", file=sys.stderr)
            continue
        if _is_episode_dir(root):
            found.append(root)
            continue
        dirs = {p.parent for p in root.rglob("actions.json")}
        dirs |= {p.parent for p in root.rglob("actions.jsonl")}
        found.extend(sorted(dirs))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return sorted(out)


def load_manifest(episode_dir: Path) -> dict:
    """The episode's ``manifest.json`` as a dict, or ``{}`` when absent or unreadable."""
    path = Path(episode_dir) / "manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ! unreadable manifest ({e}); falling back to defaults", file=sys.stderr)
        return {}


def load_records(episode_dir: Path) -> list[dict]:
    """Step records from ``actions.json`` (array) or ``actions.jsonl`` (one per line).

    Unparsable jsonl lines are skipped; non-dict entries are dropped.
    """
    episode_dir = Path(episode_dir)
    array_path = episode_dir / "actions.json"
    if array_path.is_file():
        try:
            data = json.loads(array_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! unreadable actions.json: {e}", file=sys.stderr)
            data = []
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []
    lines_path = episode_dir / "actions.jsonl"
    if not lines_path.is_file():
        return []
    out: list[dict] = []
    try:
        with open(lines_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError as e:
        print(f"  ! unreadable actions.jsonl: {e}", file=sys.stderr)
    return out


def _wall_clock_span_s(records: list[dict]) -> float | None:
    """Span of ``received_at`` across the episode, for the timing self-check."""
    stamps = []
    for r in records:
        ts = r.get("received_at")
        if not ts:
            continue
        try:
            stamps.append(datetime.fromisoformat(str(ts)))
        except ValueError:
            continue
    if len(stamps) < 2:
        return None
    try:
        return (max(stamps) - min(stamps)).total_seconds()
    except TypeError:  # mixed naive/aware stamps
        return None


def _manifest_float(manifest: dict, key: str, fallback: float) -> float:
    value = manifest.get(key)
    if value is None:
        return float(fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def render_episode_dir(episode_dir: Path, *, out_name: str = DEFAULT_OUT_NAME,
                       fps: int | None = None, timeline: str | None = None,
                       dt_s: float | None = None, height: int = 0,
                       forward_offset: str | float = "auto", hud: bool = True,
                       pointing: bool = True, traj_width: float = TRAJ_WIDTH_FRAC,
                       overwrite: bool = False, min_steps: int = 0) -> bool:
    """Render one recorded episode to ``<episode_dir>/<out_name>``.

    The manifest supplies ``fps`` / ``timeline`` / camera geometry / ``dt_s`` unless
    overridden. Images are joined to records by ``step``; steps without a usable
    image are skipped and reported. ``forward_offset="auto"`` resolves the ribbon
    offset per frame from the bottom-edge depth; a number is added to the
    manifest's ``overlay_forward_offset``. Returns True on success (or when the
    episode was legitimately skipped: too short, or output exists without
    ``overwrite``), False when nothing could be rendered.
    """
    episode_dir = Path(episode_dir)
    print(f"== {episode_dir}")
    records = load_records(episode_dir)
    if not records:
        print("  ! no step records -- skipped", file=sys.stderr)
        return False
    if len(records) < min_steps:
        print(f"  - only {len(records)} step(s) < min_steps {min_steps}, skipped")
        return True

    manifest = load_manifest(episode_dir)
    fps = int(fps or manifest.get("video_fps") or DEFAULT_VIDEO_FPS)
    if timeline:
        realtime = timeline == "realtime"
    else:
        realtime = str(manifest.get("video_timeline", "realtime")) == "realtime"
    if dt_s is None:
        dt_s = _manifest_float(manifest, "waypoint_dt_s", DEFAULT_WAYPOINT_DT_S)
    hfov = _manifest_float(manifest, "overlay_hfov_deg", FALLBACK_HFOV_DEG)
    cam_h = _manifest_float(manifest, "overlay_cam_height", FALLBACK_CAM_HEIGHT)
    fwd_recorded = _manifest_float(manifest, "overlay_forward_offset", FALLBACK_FORWARD_OFFSET)
    if isinstance(forward_offset, str) and forward_offset.strip().lower() == "auto":
        fwd: float | None = None  # resolved per frame from its own geometry
    else:
        fwd = fwd_recorded + float(forward_offset)

    out_path = episode_dir / out_name
    if out_path.exists() and not overwrite:
        print(f"  - exists, skipped (use overwrite): {out_path.name}")
        return True

    by_step: dict[int, dict] = {}
    for r in records:
        try:
            by_step[int(r["step"])] = r
        except (KeyError, TypeError, ValueError):
            continue

    written = 0
    used_steps = 0
    missing: list[int] = []
    writer = None
    # Keep the real extension: imageio picks its backend from the suffix.
    tmp_path = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    try:
        for step in sorted(by_step):
            img = episode_dir / f"{IMAGE_PREFIX}{step:06d}.jpg"
            if not img.is_file():
                missing.append(step)
                continue
            rec = by_step[step]
            try:
                rgb = decode_rgb_bytes(img.read_bytes())
            except Exception as e:  # noqa: BLE001
                print(f"  ! step {step}: undecodable {img.name}: {e}", file=sys.stderr)
                missing.append(step)
                continue
            rgb = upscale_to_height(rgb, height)
            frame = render_frame(
                rgb,
                waypoints=rec.get("waypoints"),
                instruction=str(rec.get("instruction") or ""),
                step=rec.get("step"),
                step_fps=rec.get("step_fps"),
                stop=bool(rec.get("stop", False)),
                pointing=rec.get("pointing"),
                hfov_deg=hfov,
                cam_height=cam_h,
                forward_offset=fwd,
                dt_s=dt_s,
                hud=hud,
                draw_pointing=pointing,
                traj_width=traj_width,
            )
            frame, _, _ = pad_to_even_dimensions(frame)
            if writer is None:
                writer = open_video_writer(tmp_path, fps)
            n = step_repeats(rec.get("step_dt_ms"), fps, realtime)
            for _ in range(n):
                writer.append_data(frame)
            written += n
            used_steps += 1
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        print("  ! no renderable frames -- nothing written", file=sys.stderr)
        tmp_path.unlink(missing_ok=True)
        return False

    tmp_path.replace(out_path)
    video_s = written / fps
    span = _wall_clock_span_s(records)
    line = (f"  -> {out_path.name}  steps={used_steps}/{len(by_step)}  frames={written}"
            f"  fps={fps}  timeline={'realtime' if realtime else 'per_step'}"
            f"  {video_s:.1f}s")
    if hud:
        line += f"  hud(dt={dt_s}s)"
    line += "  fwd=" + ("auto" if fwd is None else f"{fwd:.2f}m")
    if height:
        line += f"  {height}p"
    if span is not None and span >= 1.0:
        line += f"  (wall {span:.1f}s, {video_s / span * 100:.0f}%)"
    print(line)
    if missing:
        head = ", ".join(str(s) for s in missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        print(f"  ! {len(missing)} step(s) had no usable image, skipped: {head}{more}",
              file=sys.stderr)
    return True
