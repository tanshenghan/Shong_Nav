"""Token encode/decode helpers for the trajectory, target-position and pointing vocabularies.

Standard library only (``math``, ``re``) so it can be imported by lightweight clients
without torch. Every regex and bin layout here is part of the trained-in contract: a
decoder that disagrees with the label encoder by one bin reads every token wrong.
"""

import math
import re
from typing import Dict, List, Sequence, Tuple

# ── Trajectory vocabulary ─────────────────────────────────────────────────────
# A flat trajectory checkpoint emits one ``<traj_k>`` token per step whose id indexes a
# precomputed centroid array of shape (K, H, 3). K and the horizon H are fixed when the
# checkpoint is built; the defaults below match the released checkpoints.
DEFAULT_TRAJ_K: int = 256
DEFAULT_TRAJ_HORIZON: int = 10
# Legacy aliases kept for callers that still import the old names.
TRACKING_TRAJ_K: int = DEFAULT_TRAJ_K
TRACKING_TRAJ_HORIZON: int = DEFAULT_TRAJ_HORIZON


def traj_token(cluster_id: int) -> str:
    """Format a cluster id as a trajectory token literal."""
    return f"<traj_{int(cluster_id)}>"


def parse_traj_token(text: str) -> int:
    """Extract the integer id from a `<traj_K>` token (whitespace tolerated)."""
    m = re.search(r"<traj_(\d+)>", text)
    if not m:
        raise ValueError(f"No traj token found in {text!r}")
    return int(m.group(1))


# ── RVQ multi-level action vocabulary ─────────────────────────────────────────
# RVQ checkpoints emit one ``<act_l{level}_{code}>`` token per residual level
# (coarse -> fine); the per-level codebook sizes come from the bundle manifest.


def rvq_action_token(level: int, code: int) -> str:
    """Format one RVQ level's code as an ``<act_l{level}_{code}>`` token literal."""
    return f"<act_l{int(level)}_{int(code)}>"


def parse_rvq_action_tokens(text: str) -> List[int]:
    """Extract per-level codes ``[c0, c1, ...]`` from a run of ``<act_l{lvl}_{code}>``
    tokens (ordered by level). Raises if none are present."""
    by_level = {}
    for m in re.finditer(r"<act_l(\d+)_(\d+)>", text):
        by_level[int(m.group(1))] = int(m.group(2))
    if not by_level:
        raise ValueError(f"No rvq action tokens found in {text!r}")
    missing = [lvl for lvl in range(max(by_level) + 1) if lvl not in by_level]
    if missing:
        raise ValueError(f"Missing rvq act levels {missing} in {text!r}")
    return [by_level[lvl] for lvl in range(max(by_level) + 1)]


# ── Tracking target-position vocabulary ───────────────────────────────────────
# Tracking checkpoints emit ONE ``<tpos_K>`` token BEFORE the trajectory token that
# coarsely encodes where the followed person is.
#
# Layout (joint encoding, single token):
#   id = 0                                            -> invisible
#   id = 1 + az_bin * TPOS_NUM_DIST_BINS + d_bin       -> visible at (az, d)
#
# Bins (fixed analytical bins, no clustering):
#   az_bin in [0, 14] -- 15 equal bins over [-60deg, +60deg] (8deg each); |az| > 60deg
#                        clamps to the edge bin.
#   d_bin  in [0, 6]  -- 7 non-uniform distance bins (dense in the ~1-2.5 m follow
#                        band, coarse at the tails); < 0.75 m and > 5 m clamp to the
#                        end bins. See _TPOS_DIST_EDGES.
#
# Vocab size = 1 + 15 * 7 = 106.
TPOS_NUM_AZ_BINS: int = 15
TPOS_AZ_MAX_RAD: float = math.pi / 3.0  # +-60deg azimuth range (camera FOV)
# Non-uniform distance edges (meters): fine in the follow band, coarse outside.
_TPOS_DIST_EDGES: Tuple[float, ...] = (0.0, 1.0, 1.4, 1.7, 2.0, 2.5, 3.5, 5.0)
TPOS_NUM_DIST_BINS: int = len(_TPOS_DIST_EDGES) - 1  # 7
TPOS_DIST_MAX: float = _TPOS_DIST_EDGES[-1]  # 5.0m; >5m clamps to the last distance bin
TPOS_INVISIBLE_ID: int = 0
TPOS_VOCAB_SIZE: int = 1 + TPOS_NUM_AZ_BINS * TPOS_NUM_DIST_BINS

# 16 edges spanning [-TPOS_AZ_MAX_RAD, +TPOS_AZ_MAX_RAD] -> 15 equal-width bins (8deg each).
_TPOS_AZ_EDGES: Tuple[float, ...] = tuple(
    -TPOS_AZ_MAX_RAD + i * (2 * TPOS_AZ_MAX_RAD / TPOS_NUM_AZ_BINS) for i in range(TPOS_NUM_AZ_BINS + 1)
)


def tpos_token(token_id: int) -> str:
    """Format a target-position id as a `<tpos_K>` token literal."""
    return f"<tpos_{int(token_id)}>"


def safe_parse_tpos_token(text: str) -> "int | None":
    """Extract the ``<tpos_K>`` id, or ``None`` if absent or out of vocab.

    Checkpoints without the target-position channel emit ``<traj_K>`` only; callers
    treat ``None`` as "no visibility/pose information".
    """
    m = re.search(r"<tpos_(\d+)>", text)
    if not m:
        return None
    tid = int(m.group(1))
    if not (0 <= tid < TPOS_VOCAB_SIZE):
        return None
    return tid


def _bin_index(value: float, edges: Sequence[float], num_bins: int) -> int:
    """Return the bin index in [0, num_bins-1] for ``value`` given monotonically
    increasing ``edges`` of length ``num_bins + 1``. Values outside the range
    clamp to the closest edge bin (no exception).
    """
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return num_bins - 1
    # Linear scan; num_bins is tiny so this is fast and avoids numpy.
    for i in range(num_bins):
        if value < edges[i + 1]:
            return i
    return num_bins - 1


def encode_target_pos(visible: float, azimuth: float, distance: float) -> int:
    """Encode (visible, azimuth[rad], distance[m]) into a single tpos token id.

    Returns ``TPOS_INVISIBLE_ID`` (==0) when ``visible`` is falsy.
    """
    if not visible:
        return TPOS_INVISIBLE_ID
    az_bin = _bin_index(float(azimuth), _TPOS_AZ_EDGES, TPOS_NUM_AZ_BINS)
    d_bin = _bin_index(float(distance), _TPOS_DIST_EDGES, TPOS_NUM_DIST_BINS)
    return 1 + az_bin * TPOS_NUM_DIST_BINS + d_bin


def decode_target_pos(token_id: int) -> Dict[str, float]:
    """Decode a tpos id into {visible, az_bin, d_bin, az_center, d_center}.

    Centers are the geometric midpoints of the bin and are useful for
    visualization or as a fallback target prediction during inference.
    """
    tid = int(token_id)
    if tid == TPOS_INVISIBLE_ID:
        return {
            "visible": 0.0,
            "az_bin": -1,
            "d_bin": -1,
            "az_center": float("nan"),
            "d_center": float("nan"),
        }
    if not (1 <= tid < TPOS_VOCAB_SIZE):
        raise ValueError(f"tpos id {tid} out of range [0, {TPOS_VOCAB_SIZE})")
    offset = tid - 1
    az_bin = offset // TPOS_NUM_DIST_BINS
    d_bin = offset % TPOS_NUM_DIST_BINS
    az_center = 0.5 * (_TPOS_AZ_EDGES[az_bin] + _TPOS_AZ_EDGES[az_bin + 1])
    d_center = 0.5 * (_TPOS_DIST_EDGES[d_bin] + _TPOS_DIST_EDGES[d_bin + 1])
    return {
        "visible": 1.0,
        "az_bin": int(az_bin),
        "d_bin": int(d_bin),
        "az_center": float(az_center),
        "d_center": float(d_center),
    }


# ── Dual-channel grid pointing ────────────────────────────────────────────────
# Independent token family (never reuses ``<tpos_*>``). Two per-frame grounding
# channels emitted BEFORE the action tokens:
#   apos -- affordance point ("go here"), or a rotate-left / rotate-right / stop
#           directive when nothing projects; 0 = no label.
#   opos -- object/target point: the goal's pixel whenever visible; 0 = not visible.
# The grid is 48x27 = 1296 cells and frame-relative, so ids decode straight to a
# pixel fraction of the frame (no camera model). Sentinel ids 1297-1299 are apos-only.
POINT_GRID_W: int = 48
POINT_GRID_H: int = 27
POINT_CELLS: int = POINT_GRID_W * POINT_GRID_H  # 1296
APOS_ROTL_ID: int = 1 + POINT_CELLS  # 1297 -- rotate left
APOS_ROTR_ID: int = 2 + POINT_CELLS  # 1298 -- rotate right
APOS_STOP_ID: int = 3 + POINT_CELLS  # 1299 -- path exhausted: stop here
APOS_VOCAB_SIZE: int = 4 + POINT_CELLS  # 1300 (0=none, 1..1296 grid, rot-L/rot-R/stop)
OPOS_VOCAB_SIZE: int = 1 + POINT_CELLS  # 1297 (0=not visible, 1..1296 grid)


def apos_token(token_id: int) -> str:
    """Format an affordance-point id as an ``<apos_K>`` token literal."""
    return f"<apos_{int(token_id)}>"


def opos_token(token_id: int) -> str:
    """Format an object/target-point id as an ``<opos_K>`` token literal."""
    return f"<opos_{int(token_id)}>"


def encode_point_pixel(u: float, v: float, width: float, height: float) -> int:
    """(u, v) pixel -> fine-grid point id (1..1296); clamps to the edge cell."""
    col = int(min(max(float(u) / max(float(width), 1.0), 0.0), 0.999999) * POINT_GRID_W)
    row = int(min(max(float(v) / max(float(height), 1.0), 0.0), 0.999999) * POINT_GRID_H)
    return 1 + row * POINT_GRID_W + col


def decode_point_pixel_center(token_id: int, width: float, height: float) -> "tuple[float, float] | None":
    """Fine-grid point id -> (u, v) cell-center pixel, or None for 0/rot/out-of-range."""
    tid = int(token_id)
    if not (1 <= tid <= POINT_CELLS):
        return None
    k = tid - 1
    col, row = k % POINT_GRID_W, k // POINT_GRID_W
    return ((col + 0.5) / POINT_GRID_W * float(width), (row + 0.5) / POINT_GRID_H * float(height))


# ── posxy: dual-token axis coordinates (the high-resolution alternative to the
# single-token <apos_K>/<opos_K> grid). A point is TWO shared axis tokens
# ``<pos_x><pos_y>`` (x then y, 0..999 per axis), optionally prefixed by a channel
# marker ``<apos>`` / ``<opos>`` so a dual-channel output stays parseable when only
# one channel is present (tracking = opos only).
# nav 6-token output = ``<apos><pos_ax><pos_ay><opos><pos_ox><pos_oy>``.
#
# Must match the encoder used to build the checkpoint labels -- a decoder that
# disagrees with the encoder by one bin puts every marker in the wrong place.
POS_AXIS_BINS: int = 1000
POS_TOKENS: list[str] = [f"<pos_{i}>" for i in range(POS_AXIS_BINS)]
POSXY_CHANNEL_TOKEN: dict[str, str] = {"opos": "<opos>", "apos": "<apos>"}
POSXY_MARKERS: list[str] = list(POSXY_CHANNEL_TOKEN.values())
POSXY_TOKENS: list[str] = POS_TOKENS + POSXY_MARKERS
# The bare markers never collide with the grid family: <apos_K>/<opos_K> require
# ``_<digits>`` before the closing bracket.
_POSXY_RE = re.compile(r"(?:<(apos|opos)>)?<pos_(\d+)><pos_(\d+)>")

# Non-positional sentinels: an affordance can be a rotate/stop directive and a target can
# be out of view -- none of those has an (x, y), so they replace the ``<pos_x><pos_y>``
# pair after the channel marker.
POSXY_SENTINEL: dict[str, str] = {
    "rotl": "<rotl>",  # apos: rotate left in place (no forward-visible waypoint)
    "rotr": "<rotr>",  # apos: rotate right in place
    "stop": "<stop>",  # apos: arrived -> stop
    "novis": "<novis>",  # opos: target not visible this frame
}
POSXY_SENTINELS: list[str] = list(POSXY_SENTINEL.values())


def pos_token(bin_id: int) -> str:
    """Format one posxy axis bin as a ``<pos_K>`` token literal."""
    return f"<pos_{int(bin_id)}>"


def encode_point_axis(u: float, v: float, width: float, height: float) -> "tuple[int, int]":
    """(u, v) pixel -> ``(x_bin, y_bin)``, each 0..999; clamps to the edge bin."""
    xb = int(min(max(float(u) / max(float(width), 1.0), 0.0), 0.999999) * POS_AXIS_BINS)
    yb = int(min(max(float(v) / max(float(height), 1.0), 0.0), 0.999999) * POS_AXIS_BINS)
    return xb, yb


def decode_posxy_center(
    x_bin: int, y_bin: int, width: float, height: float,
) -> "tuple[float, float]":
    """``(x_bin, y_bin)`` -> ``(u, v)`` bin-center pixel against the given frame dims."""
    return (
        (int(x_bin) + 0.5) / POS_AXIS_BINS * float(width),
        (int(y_bin) + 0.5) / POS_AXIS_BINS * float(height),
    )


def render_posxy(x_bin: int, y_bin: int, channel: "str | None" = None) -> str:
    """``[<channel>]<pos_x><pos_y>``. ``channel`` in {None, 'opos', 'apos'}."""
    marker = POSXY_CHANNEL_TOKEN[channel] if channel else ""
    return f"{marker}{pos_token(x_bin)}{pos_token(y_bin)}"


def parse_posxy(text: str) -> "list[tuple[str | None, int, int]]":
    """Extract channel-marked ``<pos_x><pos_y>`` pairs as ``(channel, x_bin, y_bin)``."""
    out: list[tuple[str | None, int, int]] = []
    for m in _POSXY_RE.finditer(text):
        xb, yb = int(m.group(2)), int(m.group(3))
        if 0 <= xb < POS_AXIS_BINS and 0 <= yb < POS_AXIS_BINS:
            out.append((m.group(1), xb, yb))
    return out


def posxy_channels(text: str) -> "dict[str, tuple[int, int]]":
    """``{channel: (x_bin, y_bin)}`` for the channel-marked pairs in ``text``.

    Unmarked pairs are ignored: without a marker there is nothing to say which channel a
    pair belongs to, and guessing by position is how an apos ends up drawn as the target.
    A repeated channel keeps the FIRST pair, matching the output order the model was
    trained to emit.
    """
    found: dict[str, tuple[int, int]] = {}
    for channel, xb, yb in parse_posxy(text):
        if channel and channel not in found:
            found[channel] = (xb, yb)
    return found


# A sentinel is only a sentinel after its channel marker. ``<stop>`` is ALSO a
# discrete-action token in older checkpoints, so matching it bare would read an action
# as a pointing directive.
_POSXY_SENTINEL_RE = re.compile(r"<(apos|opos)>(?:<(rotl|rotr|stop|novis)>)")


def posxy_sentinels(text: str) -> "dict[str, str]":
    """``{channel: 'rotl'|'rotr'|'stop'|'novis'}`` for the marker-prefixed sentinels.

    These replace the ``<pos_x><pos_y>`` pair when a channel has no position at all, so a
    text can carry a sentinel for one channel and a point for the other. First occurrence
    per channel wins, matching ``posxy_channels``.
    """
    found: dict[str, str] = {}
    for m in _POSXY_SENTINEL_RE.finditer(text):
        if m.group(1) not in found:
            found[m.group(1)] = m.group(2)
    return found


def posxy_is_clamped(x_bin: int, y_bin: int) -> bool:
    """True when either axis bin sits at its boundary, i.e. the point may be CLAMPED.

    ``encode_point_axis`` clamps before binning, exactly like the grid encoder, so bin 0
    or 999 means "at that edge OR beyond it".
    """
    return int(x_bin) in (0, POS_AXIS_BINS - 1) or int(y_bin) in (0, POS_AXIS_BINS - 1)


def point_clamp_direction(token_id: int) -> "tuple[int, int]":
    """``(dx, dy)`` in {-1, 0, 1} for a boundary cell, ``(0, 0)`` when interior.

    A property of the ENCODING, not of any renderer: the encode clamps u/v into range
    before quantising, so a point outside the frame collapses onto an edge cell and the
    id alone cannot say which happened. Consumers that need the distinction (an overlay,
    a client drawing a marker) read the direction from the cell rather than from a pixel.
    """
    k = int(token_id) - 1
    if not (0 <= k < POINT_CELLS):
        return 0, 0
    col, row = k % POINT_GRID_W, k // POINT_GRID_W
    dx = -1 if col == 0 else (1 if col == POINT_GRID_W - 1 else 0)
    dy = -1 if row == 0 else (1 if row == POINT_GRID_H - 1 else 0)
    return dx, dy


def point_cell_is_clamped(token_id: int) -> bool:
    """True when a pointing id sits on the grid boundary, i.e. it may be CLAMPED.

    id 1296 means "off the bottom-right" just as much as it means "genuinely in the
    bottom-right corner", and the two are indistinguishable from the id alone.
    """
    return point_clamp_direction(token_id) != (0, 0)


def safe_parse_apos_token(text: str) -> "int | None":
    """Extract the ``<apos_K>`` affordance-point id, or ``None`` if absent / out of vocab.

    Grounding token emitted before the action tokens by dual-pointing checkpoints;
    absent on non-pointing checkpoints -> None.
    """
    m = re.search(r"<apos_(\d+)>", text)
    if not m:
        return None
    aid = int(m.group(1))
    return aid if 0 <= aid < APOS_VOCAB_SIZE else None


def safe_parse_opos_token(text: str) -> "int | None":
    """Extract the ``<opos_K>`` object/target-point id, or ``None`` if absent / out of vocab."""
    m = re.search(r"<opos_(\d+)>", text)
    if not m:
        return None
    oid = int(m.group(1))
    return oid if 0 <= oid < OPOS_VOCAB_SIZE else None
