"""Frame overlay: ground-plane trajectory ribbon, pointing markers, telemetry HUD.

All drawing works on HWC uint8 RGB arrays and returns a new array. ``cv2`` is
imported inside the functions that need it (``video`` extra); Pillow is used for
TrueType text with a cv2 Hershey fallback when no font is installed.
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Iterable

import numpy as np

from lightnav.viz.projection import bottom_edge_depth, project_waypoints_to_image
from lightnav.viz.video import _optional_import

_log = logging.getLogger(__name__)

# Seconds per waypoint step assumed by the HUD velocity readout. The waypoint chunk
# is a sequence of per-step displacements with no time base of its own; this is a
# display convention, overridable through ``dt_s`` everywhere it is used.
DEFAULT_WAYPOINT_DT_S = 0.1

# ── trajectory ribbon ───────────────────────────────────────────────────────
TRAJ_WIDTH_FRAC = 0.25       # ribbon span where it crosses the frame's bottom edge
TRAJ_FILL_ALPHA = 0.90       # opacity over the held (near) section
# Opacity is flat until TRAJ_FADE_START of the corridor length, then falls to zero.
TRAJ_FADE_START = 0.5
TRAJ_FADE_POW = 1.15         # curve shape inside the fading tail
# The chunk is resampled to this many poses before building quads, so the colour ramp
# reads as continuous rather than as one band per waypoint segment.
TRAJ_SUBDIV = 72
_TRAJ_NEAR = (48, 176, 255)   # robot end: saturated azure
_TRAJ_FAR = (214, 246, 255)   # far end: pale ice
_TRAJ_RAIL = (236, 252, 255)  # rails, paling to _TRAJ_FAR with the fill
_TRAJ_RAIL_BOOST = 1.30       # rails carry a bit more alpha than the fill they edge


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _traj_metric_half_width(h: int, w: int, cam_height: float, width_frac: float) -> float:
    """Lateral half-width in metres for a ribbon spanning ``width_frac`` of the frame
    width where it crosses the bottom edge.

    At the bottom-edge depth ``d = 2 * fy * cam_height / h`` a half-width ``Wm`` covers
    ``fx * Wm / d`` pixels; setting that to ``width_frac * w / 2`` gives
    ``Wm = width_frac * (w / h) * cam_height`` (the focal length cancels). The ribbon
    therefore has a constant real width and tapers with distance like the ground.
    """
    if h <= 0:
        return 0.0
    return width_frac * (float(w) / float(h)) * float(cam_height)


def _resample_path(wps: np.ndarray, n_out: int) -> np.ndarray:
    """Densify the waypoint polyline by linear interpolation on index."""
    n_in = wps.shape[0]
    if n_in >= n_out or n_in < 2:
        return wps
    src = np.arange(n_in, dtype=np.float64)
    dst = np.linspace(0.0, n_in - 1, n_out)
    return np.stack([np.interp(dst, src, wps[:, k]) for k in range(3)], axis=1)


def _as_waypoints(waypoints) -> np.ndarray | None:
    """Coerce to an ``(N, 3)`` float64 array, or None when the shape is unusable."""
    if waypoints is None:
        return None
    try:
        wps = np.asarray(waypoints, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if wps.ndim != 2 or wps.shape[0] < 1 or wps.shape[1] < 3:
        return None
    return wps[:, :3]


def draw_traj_ribbon(rgb: np.ndarray, waypoints, *, hfov_deg: float, cam_height: float,
                     forward_offset: float | None = None,
                     width_frac: float = TRAJ_WIDTH_FRAC,
                     alpha: float = TRAJ_FILL_ALPHA) -> np.ndarray:
    """Draw the planned path as a translucent ground-plane corridor.

    The ribbon starts under the robot (a ``(0, 0, 0)`` pose is prepended), is
    saturated azure at the near end and fades to pale ice in colour and opacity
    towards the far end, and is edged with thin rails. Both edges go through the
    same projector as the centre line, so it bends with the path and narrows with
    distance. When ``forward_offset`` is None the whole chunk is pushed out by the
    frame's bottom-edge depth so near waypoints stay visible; pass ``0.0`` for exact
    placement. Returns the input unchanged when the chunk cannot be drawn.
    """
    cv2 = _optional_import("cv2")

    wps = _as_waypoints(waypoints)
    if wps is None or wps.shape[0] < 2:
        return rgb
    h, w = rgb.shape[:2]

    # The waypoints are future poses; the path itself begins under the camera.
    wps = np.vstack([np.zeros((1, 3), dtype=np.float64), wps])

    if forward_offset is None:
        forward_offset = bottom_edge_depth(h, w, cam_height, hfov_deg)
    depth = wps[:, 0] + float(forward_offset)
    if np.flatnonzero(np.isfinite(depth) & (depth > 0.05)).size < 2:
        return rgb
    half_m = _traj_metric_half_width(h, w, cam_height, width_frac)
    if not math.isfinite(half_m) or half_m <= 0:
        return rgb
    wps = _resample_path(wps, TRAJ_SUBDIV)

    def project(lateral_shift: float) -> np.ndarray:
        shifted = wps.copy()
        shifted[:, 1] = shifted[:, 1] + lateral_shift
        return project_waypoints_to_image(
            shifted, image_size=(h, w), hfov_deg=hfov_deg,
            cam_height=cam_height, forward_offset=forward_offset,
        )

    try:
        left, right = project(+half_m), project(-half_m)
    except Exception as e:  # noqa: BLE001 -- a bad chunk must not kill the frame
        _log.warning("ribbon projection failed: %s", e)
        return rgb

    # Projected u can be enormous near the camera plane; clamp before int32.
    lim = 8 * max(h, w)

    def pt(row: np.ndarray):
        if not np.all(np.isfinite(row)):
            return None
        return (int(np.clip(row[0], -lim, lim)), int(np.clip(row[1], -lim, lim)))

    n = wps.shape[0]
    segs = []
    for i in range(n - 1):
        a_l, b_l, a_r, b_r = pt(left[i]), pt(left[i + 1]), pt(right[i]), pt(right[i + 1])
        if None in (a_l, b_l, a_r, b_r):
            continue
        segs.append((i, np.array([a_l, b_l, b_r, a_r], dtype=np.int32)))
    if not segs:
        return rgb

    # One label image plus a lookup table: each segment paints its own index once and
    # colour/alpha are resolved for the whole frame in a single gather. LINE_8, not
    # LINE_AA: an intermediate value in a label image would be a different segment.
    label = np.zeros((h, w), dtype=np.int32)
    denom = max(1, n - 2)

    def ramp(t: float) -> tuple[np.ndarray, float]:
        col = np.array([_lerp(_TRAJ_NEAR[k], _TRAJ_FAR[k], t) for k in range(3)],
                       dtype=np.float32)
        if t <= TRAJ_FADE_START:
            return col, alpha
        tail = (t - TRAJ_FADE_START) / max(1e-6, 1.0 - TRAJ_FADE_START)
        return col, alpha * (1.0 - tail) ** TRAJ_FADE_POW

    n_seg = len(segs)
    colour_lut = np.zeros((2 * n_seg + 1, 3), dtype=np.float32)
    alpha_lut = np.zeros(2 * n_seg + 1, dtype=np.float32)
    for k, (i, poly) in enumerate(segs, start=1):
        col, a = ramp(i / denom)
        colour_lut[k], alpha_lut[k] = col, a
        cv2.fillPoly(label, [poly], k)

    # Rails share the label image at indices above the fill so they overwrite it and
    # fade with it, instead of hanging as opaque wires past the faded corridor.
    rail_t = max(1, int(round(2 * h / 270.0)))
    for k, (i, _poly) in enumerate(segs, start=1):
        t = i / denom
        _, a = ramp(t)
        colour_lut[n_seg + k] = [_lerp(_TRAJ_RAIL[c], _TRAJ_FAR[c], t) for c in range(3)]
        alpha_lut[n_seg + k] = min(1.0, a * _TRAJ_RAIL_BOOST)
        for series in (left, right):
            p0, p1 = pt(series[i]), pt(series[i + 1])
            if p0 and p1:
                cv2.line(label, p0, p1, n_seg + k, rail_t)

    sel = label > 0
    if not sel.any():
        return rgb
    out = rgb.astype(np.float32)
    lab = label[sel]
    a = alpha_lut[lab][:, None]
    out[sel] = out[sel] * (1.0 - a) + colour_lut[lab] * a
    return out.astype(np.uint8)


# ── pointing markers ────────────────────────────────────────────────────────
_C_OPOS = (255, 96, 172)   # opos (target pixel): warm magenta
_C_APOS = (128, 255, 214)  # apos (affordance to move towards): mint
_GLOW_RINGS = 3            # halo layers
_POINT_COLOURS = {"apos": _C_APOS, "opos": _C_OPOS}


def _glow_disc(out: np.ndarray, cx: int, cy: int, r: int, colour, *,
               filled: bool = True, thick: int = 1) -> None:
    """Draw a disc with a fading halo (a few alpha-blended rings) in place."""
    cv2 = _optional_import("cv2")

    # Blend inside the halo's bounding box only, not the whole frame.
    h, w = out.shape[:2]
    reach = r + int(round(r * 0.55 * _GLOW_RINGS)) + 2
    x0, x1 = max(0, cx - reach), min(w, cx + reach + 1)
    y0, y1 = max(0, cy - reach), min(h, cy + reach + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = out[y0:y1, x0:x1]
    lcx, lcy = cx - x0, cy - y0
    for i in range(_GLOW_RINGS, 0, -1):
        rr = r + int(round(r * 0.55 * i))
        a = 0.16 / i
        layer = roi.copy()
        cv2.circle(layer, (lcx, lcy), rr, colour, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, a, roi, 1.0 - a, 0.0, dst=roi)
    if filled:
        cv2.circle(out, (cx, cy), r, colour, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
    else:
        cv2.circle(out, (cx, cy), r, colour, thick, cv2.LINE_AA)


def draw_pointing_soft(rgb: np.ndarray, points: Iterable[tuple[float, float, str]]) -> np.ndarray:
    """Draw pointing markers as haloed discs: mint for ``apos``, magenta for ``opos``.

    ``points`` yields ``(u_px, v_px, channel)`` already in the pixel space of ``rgb``.
    Unknown channels are ignored. opos markers are drawn last so the target pixel
    stays legible where both channels land on the same spot.
    """
    h, w = rgb.shape[:2]
    out = rgb.copy()
    r = max(4, w // 72)
    thick = max(1, r // 3)
    lim = 8 * max(h, w)

    pts = [p for p in points if len(p) >= 3 and p[2] in _POINT_COLOURS]
    order = {"apos": 0, "opos": 1}
    for u, v, channel in sorted(pts, key=lambda p: order[p[2]]):
        try:
            u, v = float(u), float(v)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        cx = int(round(max(-lim, min(lim, u))))
        cy = int(round(max(-lim, min(lim, v))))
        _glow_disc(out, cx, cy, r, _POINT_COLOURS[channel], filled=True, thick=thick + 1)
    return out


def pointing_points(pointing: dict | None, width: int, height: int) -> list[tuple[float, float, str]]:
    """Pixel markers to draw from a ``pointing_payload`` dict, rescaled to ``(width, height)``.

    A channel is kept only when ``<ch>_state == "point"`` and ``<ch>_px`` is a pixel.
    Pixels are rescaled from ``pointing["frame_size"] = [W0, H0]`` to the target size;
    a missing or invalid ``frame_size`` is taken to equal the target size.
    """
    if not isinstance(pointing, dict):
        return []
    w0, h0 = float(width), float(height)
    fs = pointing.get("frame_size")
    try:
        if fs is not None and float(fs[0]) > 0 and float(fs[1]) > 0:
            w0, h0 = float(fs[0]), float(fs[1])
    except (TypeError, ValueError, IndexError):
        pass
    sx, sy = float(width) / w0, float(height) / h0
    out: list[tuple[float, float, str]] = []
    for channel in ("apos", "opos"):
        if pointing.get(f"{channel}_state") != "point":
            continue
        px = pointing.get(f"{channel}_px")
        if px is None:
            continue
        try:
            u, v = float(px[0]), float(px[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        out.append((u * sx, v * sy, channel))
    return out


# ── telemetry HUD ───────────────────────────────────────────────────────────
# Every dimension scales off the frame height so different recording sizes get the
# same layout.
_C_ACCENT = (0, 231, 255)      # cyan: chrome, labels, brackets
_C_ACCENT_DIM = (0, 116, 145)
_C_VALUE = (255, 216, 112)     # amber: numeric readouts
_C_TEXT = (231, 244, 250)
_C_NEG = (255, 104, 132)       # rose: negative velocity
_C_PANEL = (2, 10, 18)
_C_GO = (72, 232, 160)     # status pill: go
_C_STOP = (255, 96, 120)   # status pill: stop
_PANEL_ALPHA = 0.62
_VIGNETTE = 0.28  # corner darkening; see _vignette_mask
_HERSHEY_SCALE = 0.38  # cv2 fallback text scale at 270p, near the 11 px TrueType sizes

# Full scale for the velocity bars. Fixed rather than auto-ranged so bars stay
# comparable across frames.
_VX_FULL = 3.0    # m/s
_VY_FULL = 0.5    # m/s
_VYAW_FULL = 3.5  # rad/s

# Condensed (regular, bold) for chrome and the instruction, monospace bold for
# numerals so readouts do not jitter as digits change width.
_FONT_SETS = (
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
)


@lru_cache(maxsize=1)
def _font_set() -> tuple[str, str, str] | None:
    """First fully-installed (regular, bold, mono-bold) triple, or None."""
    for paths in _FONT_SETS:
        if all(os.path.exists(f) for f in paths):
            return paths
    return None


@lru_cache(maxsize=64)
def _font(role: str, px: int):
    """A PIL font for ``role`` in {"reg", "bold", "mono"}, or None if unavailable."""
    paths = _font_set()
    if paths is None:
        return None
    try:
        from PIL import ImageFont

        return ImageFont.truetype(paths[{"reg": 0, "bold": 1, "mono": 2}[role]], px)
    except Exception:  # noqa: BLE001 -- any font failure falls back to cv2 text
        return None


_SENTENCE_END = ".!?"


def normalize_instruction(text: str) -> str:
    """Collapse whitespace, capitalise the first letter, and add a terminal period.

    Both steps are no-ops when the text already complies.
    """
    t = " ".join((text or "").split())
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    if t[-1] not in _SENTENCE_END:
        t += "."
    return t


def body_velocity(waypoints, dt_s: float):
    """``(vx, vy, vyaw)`` from the first waypoint, or None when there is none.

    The chunk carries per-step displacements in metres and radians; dividing the
    first row by ``dt_s`` gives m/s and rad/s. Only waypoint 0 is used because it
    is the only pose that is commanded before the next replan.
    """
    if waypoints is None:
        return None
    try:
        wp = np.asarray(waypoints, dtype=float)
    except (TypeError, ValueError):
        return None
    if wp.ndim != 2 or wp.shape[0] < 1 or wp.shape[1] < 3 or dt_s <= 0:
        return None
    if not np.all(np.isfinite(wp[0, :3])):
        return None
    fwd, lat, yaw = (float(x) for x in wp[0, :3])
    return fwd / dt_s, lat / dt_s, yaw / dt_s


@lru_cache(maxsize=8)
def _vignette_mask(h: int, w: int) -> np.ndarray:
    """Radial falloff in ``[1 - _VIGNETTE, 1]`` as an ``(h, w, 1)`` multiplier.

    Darkening the corners keeps the translucent chrome legible over bright frames
    without raising the panel alpha. Cached per frame size.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny = (yy / max(1, h - 1) - 0.5) * 2.0
    nx = (xx / max(1, w - 1) - 0.5) * 2.0
    r = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)
    fall = np.clip((r - 0.55) / 0.45, 0.0, 1.0) ** 1.6
    return (1.0 - _VIGNETTE * fall)[:, :, None]


def _apply_vignette(out: np.ndarray) -> np.ndarray:
    h, w = out.shape[:2]
    return np.clip(out.astype(np.float32) * _vignette_mask(h, w), 0, 255).astype(np.uint8)


def _blend_panel(out: np.ndarray, x0: int, y0: int, x1: int, y1: int, rgb, alpha: float) -> None:
    """Alpha-blend a filled axis-aligned rectangle in place."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(out.shape[1], x1), min(out.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = out[y0:y1, x0:x1].astype(np.float32)
    tint = np.asarray(rgb, dtype=np.float32)
    out[y0:y1, x0:x1] = (roi * (1.0 - alpha) + tint * alpha).astype(np.uint8)


def _chamfer_panel(out: np.ndarray, x0: int, y0: int, x1: int, y1: int, cut: int) -> None:
    """Translucent panel with the top-right corner cut off, blended in place."""
    cv2 = _optional_import("cv2")

    poly = np.array([(x0, y0), (x1 - cut, y0), (x1, y0 + cut), (x1, y1), (x0, y1)],
                    dtype=np.int32)
    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    sel = mask.astype(bool)
    if not sel.any():
        return
    tint = np.asarray(_C_PANEL, dtype=np.float32)
    out[sel] = (out[sel].astype(np.float32) * (1.0 - _PANEL_ALPHA)
                + tint * _PANEL_ALPHA).astype(np.uint8)


def _corner_brackets(out: np.ndarray, s: float, top: int) -> None:
    """Viewport L-brackets in three corners; ``top`` keeps the upper pair below the header.

    The bottom-left corner is left free for the telemetry block.
    """
    cv2 = _optional_import("cv2")

    h, w = out.shape[:2]
    arm = max(9, int(round(22 * s)))
    t = max(1, int(round(2 * s)))
    m = max(3, int(round(7 * s)))
    for cx, cy, dx, dy in ((m, top + m, 1, 1), (w - m, top + m, -1, 1),
                           (w - m, h - m, -1, -1)):
        cv2.line(out, (cx, cy), (cx + dx * arm, cy), _C_ACCENT, t, cv2.LINE_AA)
        cv2.line(out, (cx, cy), (cx, cy + dy * arm), _C_ACCENT, t, cv2.LINE_AA)


def _bar(out: np.ndarray, x: int, y: int, w: int, h: int, frac: float) -> None:
    """Clamped bar growing from a centre tick, so sign is length as well as colour."""
    cv2 = _optional_import("cv2")

    frac = 0.0 if frac is None else max(-1.0, min(1.0, frac))
    cv2.rectangle(out, (x, y), (x + w, y + h), _C_ACCENT_DIM, 1, cv2.LINE_AA)
    ix, iy, iw, ih = x + 1, y + 1, w - 2, h - 2
    if iw <= 0 or ih <= 0:
        return
    colour = _C_VALUE if frac >= 0 else _C_NEG
    mid = ix + iw // 2
    span = int(round(abs(frac) * iw / 2))
    if span:
        a, b = (mid, mid + span) if frac >= 0 else (mid - span, mid)
        cv2.rectangle(out, (a, iy), (b, iy + ih), colour, -1)
    cv2.line(out, (mid, y + 1), (mid, y + h - 1), _C_ACCENT, 1, cv2.LINE_AA)


def draw_scifi_hud(rgb: np.ndarray, *, instruction: str, step, fps, vel,
                   stop: bool = False) -> np.ndarray:
    """Header (instruction | GO/STOP | step | rate) plus a body-velocity readout.

    ``vel`` is ``(vx, vy, vyaw)`` or None; ``fps`` is the step rate in Hz or None.
    Text is placed with PIL anchors on real font metrics; when no TrueType face is
    installed the HUD falls back to cv2's Hershey font.
    """
    cv2 = _optional_import("cv2")
    from PIL import Image, ImageDraw

    # Vignette before the chrome so the panels and text are not dimmed.
    out = _apply_vignette(rgb)
    h, w = out.shape[:2]
    s = h / 270.0
    pad = max(4, int(round(9 * s)))
    f_instr = max(10, int(round(12 * s)))
    f_head = max(9, int(round(11 * s)))
    f_lab = max(8, int(round(9 * s)))
    f_val = max(10, int(round(12 * s)))
    f_unit = max(7, int(round(8 * s)))

    instr_f = _font("bold", f_instr)
    head_lab_f, head_num_f = _font("bold", f_head), _font("mono", f_head)
    lab_f, val_f, unit_f = _font("bold", f_lab), _font("mono", f_val), _font("reg", f_unit)
    pil_ok = all(f is not None for f in (instr_f, head_lab_f, head_num_f, lab_f,
                                        val_f, unit_f))

    hershey_scale = _HERSHEY_SCALE * s

    def width(text: str, font) -> int:
        if pil_ok:
            x0, _, x1, _ = font.getbbox(text)
            return x1 - x0
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, hershey_scale, 1)
        return tw

    # ---- header ------------------------------------------------------------
    bar_h = max(16, int(round(24 * s)))
    _blend_panel(out, 0, 0, w, bar_h, _C_PANEL, _PANEL_ALPHA)
    cv2.line(out, (0, bar_h), (w, bar_h), _C_ACCENT, max(1, int(round(s))), cv2.LINE_AA)
    mid_y = bar_h // 2

    # GO / STOP reflects the decoded action (the ``stop`` flag), not a pointing directive.
    pill_txt = "STOP" if stop else "GO"
    pill_bg = _C_STOP if stop else _C_GO
    pill_pad = max(3, int(round(6 * s)))
    pill_h = max(11, bar_h - 2 * max(2, int(round(4 * s))))
    step_txt = f"STEP {int(step):04d}" if step is not None else "STEP ----"
    try:
        fps_txt = f"{float(fps):.2f} Hz"
    except (TypeError, ValueError):
        fps_txt = "--.-- Hz"
    w_step, w_fps = width(step_txt, head_num_f), width(fps_txt, head_num_f)
    w_sep = width("  |  ", head_lab_f)
    w_pill = width(pill_txt, head_lab_f) + 2 * pill_pad
    right_w = w_pill + pill_pad + w_step + w_sep + w_fps

    # A chevron marks the instruction as a command being executed.
    chev_w = max(4, int(round(5 * s)))
    instr_x = pad + chev_w + max(3, int(round(5 * s)))
    instr = " ".join((instruction or "").split())
    avail = w - instr_x - pad - right_w - pad
    ellipsis = "…" if pil_ok else "..."
    if instr and width(instr, instr_f) > avail:
        while instr and width(instr + ellipsis, instr_f) > avail:
            instr = instr[:-1]
        instr = instr.rstrip() + ellipsis
    cv2.fillPoly(out, [np.array([(pad, mid_y - chev_w), (pad + chev_w, mid_y),
                                 (pad, mid_y + chev_w)], dtype=np.int32)],
                 _C_ACCENT, cv2.LINE_AA)

    # ---- telemetry block ---------------------------------------------------
    rows = (("VX", "m/s", 0, _VX_FULL), ("VY", "m/s", 1, _VY_FULL),
            ("VYAW", "rad/s", 2, _VYAW_FULL))
    lab_w = max(width(r[0], lab_f) for r in rows)
    val_w = width("-0.00", val_f)  # mono: one measurement fits every value
    unit_w = max(width(r[1], unit_f) for r in rows)
    bar_w = max(20, int(round(34 * s)))
    row_h = max(10, int(round(13 * s)))
    gap = max(2, int(round(4 * s)))
    ipad = max(3, int(round(6 * s)))
    blk_w = ipad + lab_w + gap + val_w + gap + unit_w + gap + bar_w + ipad
    blk_h = gap + len(rows) * row_h + gap
    bx, by = pad, h - pad - blk_h
    _chamfer_panel(out, bx, by, bx + blk_w, by + blk_h, max(3, int(round(7 * s))))
    cv2.line(out, (bx, by + 1), (bx, by + blk_h - 1), _C_ACCENT,
             max(2, int(round(2 * s))), cv2.LINE_AA)

    text_ops = []  # (x, y, text, colour, font, anchor)
    text_ops.append((instr_x, mid_y, instr, _C_TEXT, instr_f, "lm"))
    rx = w - pad
    text_ops.append((rx, mid_y, fps_txt, _C_VALUE, head_num_f, "rm"))
    text_ops.append((rx - w_fps, mid_y, "  |  ", _C_ACCENT_DIM, head_lab_f, "rm"))
    text_ops.append((rx - w_fps - w_sep, mid_y, step_txt, _C_ACCENT, head_num_f, "rm"))
    # Status pill: filled chip with dark glyphs, the one element meant to be seen first.
    px1 = rx - w_fps - w_sep - w_step - pill_pad
    px0 = px1 - w_pill
    py0 = mid_y - pill_h // 2
    cv2.rectangle(out, (px0, py0), (px1, py0 + pill_h), pill_bg, -1, cv2.LINE_AA)
    text_ops.append(((px0 + px1) // 2, mid_y, pill_txt, _C_PANEL, head_lab_f, "mm"))

    for i, (label, unit, idx, full) in enumerate(rows):
        cy = by + gap + i * row_h + row_h // 2
        val = None if vel is None else vel[idx]
        vtxt = "--.--" if val is None else f"{val:.2f}"
        colour = _C_VALUE if (val is None or val >= 0) else _C_NEG
        lx = bx + ipad
        text_ops.append((lx, cy, label, _C_ACCENT, lab_f, "lm"))
        vx_right = lx + lab_w + gap + val_w
        text_ops.append((vx_right, cy, vtxt, colour, val_f, "rm"))
        text_ops.append((vx_right + gap, cy, unit, _C_ACCENT_DIM, unit_f, "lm"))
        bh = max(4, int(round(6 * s)))
        _bar(out, vx_right + gap + unit_w + gap, cy - bh // 2, bar_w, bh,
             None if val is None else val / full)

    if pil_ok:
        img = Image.fromarray(out)
        d = ImageDraw.Draw(img)
        for x, y, txt, colour, font, anchor in text_ops:
            if txt:
                d.text((x, y), txt, font=font, fill=colour, anchor=anchor)
        out = np.array(img, dtype=np.uint8)  # np.array copies -> writable for cv2
    else:
        for x, y, txt, colour, _unused_font, anchor in text_ops:
            if not txt:
                continue
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, hershey_scale, 1)
            if anchor and anchor[0] == "r":
                x0 = x - tw
            elif anchor and anchor[0] == "m":
                x0 = x - tw // 2
            else:
                x0 = x
            cv2.putText(out, txt, (x0, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        hershey_scale, colour, 1, cv2.LINE_AA)

    return out


def _as_rgb_u8(rgb) -> np.ndarray:
    """Coerce to a fresh HWC uint8 RGB array (grayscale is stacked, alpha dropped)."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an HWC RGB frame, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr).copy()


def render_frame(rgb: np.ndarray, *, waypoints=None, instruction: str = "", step=None,
                 step_fps=None, stop: bool = False, pointing: dict | None = None,
                 hfov_deg: float, cam_height: float, forward_offset: float | None = None,
                 dt_s: float = DEFAULT_WAYPOINT_DT_S, hud: bool = True,
                 draw_pointing: bool = True, traj_width: float = TRAJ_WIDTH_FRAC) -> np.ndarray:
    """Compose the full overlay: trajectory ribbon, pointing markers, then the HUD.

    ``waypoints`` is an ``(N, 3)`` chunk of ``[forward, lateral, yaw]`` displacements
    (the ribbon needs at least two rows); ``pointing`` is a ``pointing_payload`` dict
    whose pixels are rescaled to this frame; ``step_fps`` is the step rate shown in
    the header. Bad content is logged and skipped rather than raised; the only
    exception propagated is ImportError when cv2 is not installed.
    """
    _optional_import("cv2")

    try:
        out = _as_rgb_u8(rgb)
    except Exception as e:  # noqa: BLE001
        _log.warning("render_frame: unusable frame (%s)", e)
        return rgb

    wps = _as_waypoints(waypoints)
    if wps is not None and wps.shape[0] >= 2:
        try:
            out = draw_traj_ribbon(out, wps, hfov_deg=hfov_deg, cam_height=cam_height,
                                   forward_offset=forward_offset, width_frac=traj_width)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            _log.warning("render_frame: ribbon skipped (%s)", e)

    if draw_pointing and pointing is not None:
        try:
            h, w = out.shape[:2]
            pts = pointing_points(pointing, w, h)
            if pts:
                out = draw_pointing_soft(out, pts)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            _log.warning("render_frame: pointing skipped (%s)", e)

    if hud:
        try:
            out = draw_scifi_hud(
                out,
                instruction=normalize_instruction(str(instruction or "")),
                step=step,
                fps=step_fps,
                vel=body_velocity(wps, dt_s),
                stop=bool(stop),
            )
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            _log.warning("render_frame: HUD skipped (%s)", e)

    return out
