"""Overlay primitives on synthetic frames: ``render_frame`` and its parts, the
pointing rescale, the HUD, the velocity readout, the projection maths and the
video helpers (timebase, padding, resize, JPEG codec).

Needs cv2 (the ``video`` extra); the module itself imports without it.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from lightnav.viz import (
    DEFAULT_WAYPOINT_DT_S,
    MAX_STEP_REPEATS,
    body_velocity,
    bottom_edge_depth,
    decode_rgb_bytes,
    draw_pointing_soft,
    draw_scifi_hud,
    draw_traj_ribbon,
    encode_jpeg_bytes,
    normalize_instruction,
    open_video_writer,
    pad_to_even_dimensions,
    pointing_points,
    project_waypoints_to_image,
    render_frame,
    step_repeats,
    upscale_to_height,
)

pytest.importorskip("cv2")

H, W = 270, 480
CAM = dict(hfov_deg=112.0, cam_height=0.5)


def _frame(h: int = H, w: int = W) -> np.ndarray:
    """A deterministic mid-grey gradient so overlays are visible in every region."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = (60 + 120 * xx / max(1, w - 1)).astype(np.uint8)
    g = (60 + 120 * yy / max(1, h - 1)).astype(np.uint8)
    b = np.full((h, w), 110, np.uint8)
    return np.stack([r, g, b], axis=-1)


def _chunk(n: int = 10) -> np.ndarray:
    """A gently curving 10-row chunk of ``[forward, lateral, yaw]`` poses."""
    return np.column_stack(
        [np.linspace(0.25, 2.5, n), np.linspace(0.0, 0.6, n), np.linspace(0.0, 0.3, n)]
    )


def _pointing(**overrides) -> dict:
    payload = {
        "mode": "grid",
        "frame_size": [W, H],
        "apos_px": [240.0, 135.0],
        "opos_px": [60.0, 200.0],
        "apos_state": "point",
        "opos_state": "point",
    }
    payload.update(overrides)
    return payload


# -- render_frame --------------------------------------------------------------------------


def test_render_frame_keeps_shape_and_dtype_and_draws_something():
    rgb = _frame()
    before = rgb.copy()

    out = render_frame(rgb, waypoints=_chunk(), instruction="go to the sofa", step=3,
                       step_fps=9.5, pointing=_pointing(), **CAM)

    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    assert not np.array_equal(out, rgb)
    assert np.array_equal(rgb, before), "the input frame must not be modified in place"


def test_render_frame_is_deterministic():
    kwargs = dict(waypoints=_chunk(), instruction="turn left at the door", step=12,
                  step_fps=10.0, stop=True, pointing=_pointing(), **CAM)
    a = render_frame(_frame(), **kwargs)
    b = render_frame(_frame(), **kwargs)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "waypoints",
    [
        None,
        [],
        np.zeros((0, 3)),
        np.zeros((1, 3)),
        [[0.5, 0.0, 0.0]],
        np.full((10, 3), np.nan),
        np.array([[np.inf, 0.0, 0.0], [1.0, np.nan, 0.0]]),
        np.zeros((10, 3)),
        "junk",
        [[1.0, 2.0], [3.0, 4.0]],
        np.zeros((2, 3, 4)),
        [["a", "b", "c"], ["d", "e", "f"]],
    ],
    ids=["none", "empty-list", "empty-array", "one-row", "one-row-list", "nan", "inf-nan",
         "zeros", "string", "two-cols", "3d", "letters"],
)
def test_render_frame_never_raises_on_an_unusable_chunk(waypoints):
    out = render_frame(_frame(), waypoints=waypoints, instruction="go", step=1, **CAM)
    assert out.shape == (H, W, 3) and out.dtype == np.uint8


@pytest.mark.parametrize(
    "pointing",
    [{}, {"frame_size": "wide"}, {"apos_state": "point", "apos_px": "x"},
     {"apos_state": "point", "apos_px": [None, 3]}, {"apos_state": "point", "apos_px": [1e12, 1e12]},
     ["apos"], "apos"],
)
def test_render_frame_never_raises_on_a_junk_pointing_payload(pointing):
    out = render_frame(_frame(), pointing=pointing, **CAM)
    assert out.shape == (H, W, 3) and out.dtype == np.uint8


def test_render_frame_with_nothing_to_draw_returns_an_equal_copy():
    rgb = _frame()
    out = render_frame(rgb, hud=False, **CAM)
    assert out is not rgb
    assert np.array_equal(out, rgb)


def test_render_frame_ribbon_needs_two_rows():
    rgb = _frame()
    one = render_frame(rgb, waypoints=np.array([[1.0, 0.0, 0.0]]), hud=False, **CAM)
    two = render_frame(rgb, waypoints=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), hud=False, **CAM)
    assert np.array_equal(one, rgb)
    assert not np.array_equal(two, rgb)


def test_render_frame_pointing_markers_land_where_the_payload_says():
    rgb = _frame()
    out = render_frame(rgb, pointing=_pointing(opos_state="none", opos_px=None), hud=False, **CAM)

    assert not np.array_equal(out[135, 240], rgb[135, 240])  # apos disc centre
    assert np.array_equal(out[5, 5], rgb[5, 5])  # far corner untouched
    assert np.array_equal(out[200, 60], rgb[200, 60])  # opos was not "point"


def test_render_frame_can_switch_pointing_and_hud_off():
    rgb = _frame()
    out = render_frame(rgb, pointing=_pointing(), hud=False, draw_pointing=False, **CAM)
    assert np.array_equal(out, rgb)


def test_render_frame_accepts_grayscale_float_and_rgba_input():
    grey = render_frame(np.zeros((H, W), np.uint8), **CAM)
    rgba = render_frame(np.zeros((H, W, 4), np.uint8), **CAM)
    flt = render_frame(np.zeros((H, W, 3), np.float32) + 300.0, **CAM)
    for out in (grey, rgba, flt):
        assert out.shape == (H, W, 3) and out.dtype == np.uint8


def test_render_frame_returns_an_unusable_frame_untouched():
    bad = np.zeros(7)
    assert render_frame(bad, **CAM) is bad


def test_render_frame_without_cv2_raises_the_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match=r"lightnav\[video\]"):
        render_frame(_frame(), **CAM)


def test_open_video_writer_without_imageio_raises_the_install_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "imageio.v2", None)
    with pytest.raises(ImportError, match=r"lightnav\[video\]"):
        open_video_writer(tmp_path / "x.mp4", 10)


# -- ribbon ----------------------------------------------------------------------------------


def test_draw_traj_ribbon_draws_near_the_bottom_centre_and_leaves_the_sky_alone():
    rgb = _frame()
    out = draw_traj_ribbon(rgb, _chunk(), **CAM)

    changed = np.any(out != rgb, axis=-1)
    assert changed[H - 3, W // 2], "the corridor starts under the robot at the bottom edge"
    assert not changed[:20].any(), "nothing above the horizon"
    assert out.dtype == np.uint8 and out.shape == rgb.shape


def test_draw_traj_ribbon_with_exact_placement_differs_from_auto_offset():
    rgb = _frame()
    auto = draw_traj_ribbon(rgb, _chunk(), **CAM)
    exact = draw_traj_ribbon(rgb, _chunk(), forward_offset=0.0, **CAM)
    assert not np.array_equal(auto, exact)


@pytest.mark.parametrize("waypoints", [None, [], np.zeros((1, 3)), np.full((5, 3), np.nan), "x"])
def test_draw_traj_ribbon_returns_the_input_for_an_unusable_chunk(waypoints):
    rgb = _frame()
    assert draw_traj_ribbon(rgb, waypoints, **CAM) is rgb


# -- pointing --------------------------------------------------------------------------------


def test_pointing_points_rescales_from_the_payload_frame_size():
    pts = pointing_points(_pointing(), width=960, height=540)
    assert pts == [(480.0, 270.0, "apos"), (120.0, 400.0, "opos")]


def test_pointing_points_keeps_only_channels_in_the_point_state():
    pts = pointing_points(_pointing(opos_state="rot"), W, H)
    assert pts == [(240.0, 135.0, "apos")]

    pts = pointing_points(_pointing(apos_state="none", apos_px=None), W, H)
    assert pts == [(60.0, 200.0, "opos")]

    assert pointing_points(_pointing(apos_state="point", apos_px=None, opos_state="stop"), W, H) == []


def test_pointing_points_without_a_frame_size_assumes_the_target_size():
    payload = _pointing()
    del payload["frame_size"]
    assert pointing_points(payload, 960, 540) == [(240.0, 135.0, "apos"), (60.0, 200.0, "opos")]
    assert pointing_points(_pointing(frame_size=[0, 0]), 960, 540) == pointing_points(payload, 960, 540)


@pytest.mark.parametrize("payload", [None, [], "apos", 3, {"apos_state": "point", "apos_px": [float("nan"), 1]}])
def test_pointing_points_ignores_junk(payload):
    assert pointing_points(payload, W, H) == []


def test_draw_pointing_soft_draws_opos_over_apos_and_ignores_unknown_channels():
    rgb = _frame()
    out = draw_pointing_soft(rgb, [(240, 135, "opos"), (240, 135, "apos"), (30, 30, "tpos")])

    assert tuple(int(c) for c in out[135, 240]) == (255, 96, 172)  # magenta centre: opos wins
    assert np.array_equal(out[30, 30], rgb[30, 30])  # unknown channel not drawn
    assert not np.array_equal(out, rgb)
    assert np.array_equal(rgb, _frame())

    apos_only = draw_pointing_soft(rgb, [(240, 135, "apos")])
    assert tuple(int(c) for c in apos_only[135, 240]) == (128, 255, 214)  # mint


def test_draw_pointing_soft_tolerates_off_frame_and_non_finite_points():
    rgb = _frame()
    out = draw_pointing_soft(rgb, [(-5000, 20, "apos"), (float("nan"), 1, "opos"),
                                   ("x", 1, "opos"), (1e9, 1e9, "apos")])
    assert out.shape == rgb.shape and out.dtype == np.uint8


# -- instruction / velocity ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  go   to the\n sofa ", "Go to the sofa."),
        ("go!", "Go!"),
        ("Turn around?", "Turn around?"),
        ("Already fine.", "Already fine."),
        ("", ""),
        ("   ", ""),
        (None, ""),
        ("a", "A."),
        ("éclair", "Éclair."),
    ],
)
def test_normalize_instruction(raw, expected):
    assert normalize_instruction(raw) == expected


def test_body_velocity_uses_the_first_waypoint_only():
    wps = [[0.25, -0.05, 0.1], [9.0, 9.0, 9.0]]
    vx, vy, vyaw = body_velocity(wps, 0.1)
    assert (vx, vy, vyaw) == (pytest.approx(2.5), pytest.approx(-0.5), pytest.approx(1.0))


def test_body_velocity_accepts_ndarray_input_and_extra_columns():
    arr = np.array([[0.5, 0.0, 0.2, 99.0]], dtype=np.float32)
    vx, vy, vyaw = body_velocity(arr, DEFAULT_WAYPOINT_DT_S)
    assert vx == pytest.approx(5.0) and vy == 0.0 and vyaw == pytest.approx(2.0, abs=1e-6)
    assert body_velocity(np.zeros((10, 3)), 1.0) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("waypoints", "dt"),
    [
        (None, 0.1),
        ([], 0.1),
        (np.zeros((0, 3)), 0.1),
        ([[1.0, 2.0]], 0.1),
        (np.zeros(3), 0.1),
        ([[np.nan, 0.0, 0.0]], 0.1),
        ([[1.0, 0.0, 0.0]], 0.0),
        ([[1.0, 0.0, 0.0]], -1.0),
        ("junk", 0.1),
        ([["a", "b", "c"]], 0.1),
    ],
)
def test_body_velocity_returns_none_when_undefined(waypoints, dt):
    assert body_velocity(waypoints, dt) is None


# -- HUD -------------------------------------------------------------------------------------


@pytest.mark.parametrize("stop", [False, True])
def test_draw_scifi_hud_renders_for_both_states(stop):
    rgb = _frame()
    out = draw_scifi_hud(rgb, instruction="Go to the sofa.", step=7, fps=9.87,
                         vel=(1.2, -0.1, 0.3), stop=stop)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert not np.array_equal(out, rgb)
    assert np.array_equal(rgb, _frame())


def test_draw_scifi_hud_stop_and_go_differ_only_in_the_header():
    kwargs = dict(instruction="Follow the person.", step=7, fps=10.0, vel=(0.5, 0.0, 0.0))
    go = draw_scifi_hud(_frame(), stop=False, **kwargs)
    stop = draw_scifi_hud(_frame(), stop=True, **kwargs)

    changed = np.any(go != stop, axis=-1)
    assert changed.any()
    assert not changed[40:].any(), "only the header pill differs between GO and STOP"


def test_draw_scifi_hud_tolerates_missing_telemetry_and_long_text():
    rgb = _frame()
    out = draw_scifi_hud(rgb, instruction="", step=None, fps=None, vel=None)
    assert out.shape == rgb.shape
    long = draw_scifi_hud(rgb, instruction="walk " * 200, step=123456, fps="fast", vel=(-9.0, 9.0, -9.0))
    assert long.shape == rgb.shape and long.dtype == np.uint8


def test_draw_scifi_hud_scales_to_other_frame_sizes():
    for h, w in ((90, 160), (1080, 1920), (33, 47)):
        out = draw_scifi_hud(_frame(h, w), instruction="go", step=1, fps=1.0, vel=(0.0, 0.0, 0.0))
        assert out.shape == (h, w, 3)


def test_draw_scifi_hud_falls_back_to_hershey_text_without_truetype(monkeypatch):
    from lightnav.viz import render as render_mod

    monkeypatch.setattr(render_mod, "_font", lambda role, px: None)
    rgb = _frame()
    out = draw_scifi_hud(rgb, instruction="go to the sofa", step=3, fps=10.0, vel=(1.0, 0.0, 0.0))
    assert out.shape == rgb.shape and not np.array_equal(out, rgb)


# -- video helpers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step_dt_ms", "fps", "realtime", "expected"),
    [
        (None, 10, True, 1),
        (0, 10, True, 1),
        (0.0, 10, True, 1),
        (-50, 10, True, 1),
        (300, 10, True, 3),
        (300.0, 10, True, 3),
        ("300", 10, True, 3),
        (1000, 10, True, 10),
        (10_000_000, 10, True, MAX_STEP_REPEATS),
        (300, 10, False, 1),
        (10_000_000, 10, False, 1),
        ("junk", 10, True, 1),
        (float("nan"), 10, True, 1),
        (10, 10, True, 1),  # rounds to zero frames -> still one
    ],
)
def test_step_repeats_table(step_dt_ms, fps, realtime, expected):
    assert step_repeats(step_dt_ms, fps, realtime) == expected


def test_step_repeats_clamps_an_infinite_duration():
    assert step_repeats(float("inf"), 10, True) == MAX_STEP_REPEATS


def test_max_step_repeats_is_twenty():
    assert MAX_STEP_REPEATS == 20


def test_pad_to_even_dimensions_replicates_the_edge():
    frame = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    out, pad_right, pad_bottom = pad_to_even_dimensions(frame)

    assert out.shape == (8, 10, 3) and (pad_right, pad_bottom) == (1, 1)
    assert np.array_equal(out[:7, :9], frame)
    assert np.array_equal(out[7], out[6])  # bottom row replicated
    assert np.array_equal(out[:, 9], out[:, 8])  # right column replicated


def test_pad_to_even_dimensions_leaves_even_frames_alone():
    frame = np.zeros((8, 10, 3), np.uint8)
    out, pad_right, pad_bottom = pad_to_even_dimensions(frame)
    assert out is frame and (pad_right, pad_bottom) == (0, 0)

    tall, r, b = pad_to_even_dimensions(np.zeros((9, 10, 3), np.uint8))
    assert tall.shape == (10, 10, 3) and (r, b) == (0, 1)
    wide, r, b = pad_to_even_dimensions(np.zeros((10, 9), np.uint8))
    assert wide.shape == (10, 10) and (r, b) == (1, 0)


def test_upscale_to_height_keeps_aspect_and_even_width():
    rgb = _frame(270, 480)
    up = upscale_to_height(rgb, 540)
    assert up.shape == (540, 960, 3) and up.dtype == np.uint8
    down = upscale_to_height(rgb, 90)
    assert down.shape == (90, 160, 3)
    odd = upscale_to_height(_frame(100, 75), 30)
    assert odd.shape[0] == 30 and odd.shape[1] % 2 == 0
    assert upscale_to_height(rgb, 0) is rgb
    assert upscale_to_height(rgb, 270) is rgb


def test_jpeg_roundtrip_preserves_size_and_approximate_colour():
    rgb = np.zeros((36, 64, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[..., 2] = 40
    data = encode_jpeg_bytes(rgb)
    assert data[:2] == b"\xff\xd8"

    back = decode_rgb_bytes(data)
    assert back.shape == rgb.shape and back.dtype == np.uint8
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 8

    rgba = encode_jpeg_bytes(np.zeros((4, 4, 4), np.float32) + 999.0)
    assert decode_rgb_bytes(rgba).shape == (4, 4, 3)


# -- projection ------------------------------------------------------------------------------


def test_bottom_edge_depth_numeric():
    assert bottom_edge_depth(270, 480, 0.5, 112.0) == pytest.approx(0.600, abs=1e-3)
    assert bottom_edge_depth(0, 480, 0.5, 112.0) == 0.0
    # Scales linearly with camera height and with the focal length (narrower FOV -> deeper).
    assert bottom_edge_depth(270, 480, 1.0, 112.0) == pytest.approx(2 * 0.5995631, abs=1e-6)
    assert bottom_edge_depth(270, 480, 0.5, 90.0) > bottom_edge_depth(270, 480, 0.5, 112.0)


@pytest.mark.parametrize(("h", "w", "cam_height", "hfov"), [(270, 480, 0.5, 112.0),
                                                             (480, 848, 0.88, 120.0),
                                                             (1080, 1920, 0.3, 69.4)])
def test_bottom_edge_depth_projects_onto_the_bottom_edge(h, w, cam_height, hfov):
    d = bottom_edge_depth(h, w, cam_height, hfov)
    uv = project_waypoints_to_image(np.array([[d, 0.0, 0.0]]), (h, w), hfov, cam_height)
    assert uv.shape == (1, 2)
    assert uv[0, 1] == pytest.approx(h, abs=1e-9)
    assert uv[0, 0] == pytest.approx(w / 2.0, abs=1e-9)


def test_project_waypoints_geometry():
    hfov, cam_h = 90.0, 0.5
    # fx = (480/2)/tan(45deg) = 240; a point 1 m ahead, 0.5 m to the left -> u = 240 - 120.
    uv = project_waypoints_to_image(np.array([[1.0, 0.5, 0.0]]), (270, 480), hfov, cam_h)
    assert uv[0] == pytest.approx([240.0 - 120.0, 135.0 + 240.0 * 0.5], abs=1e-9)

    # Farther points converge on the principal point (the horizon).
    far = project_waypoints_to_image(np.array([[1000.0, 0.0, 0.0]]), (270, 480), hfov, cam_h)
    assert far[0, 1] == pytest.approx(135.0, abs=0.2)

    # Depth at or below min_depth is NaN; a forward_offset rescues it.
    near = project_waypoints_to_image(np.array([[0.0, 0.0, 0.0]]), (270, 480), hfov, cam_h)
    assert np.isnan(near).all()
    pushed = project_waypoints_to_image(np.array([[0.0, 0.0, 0.0]]), (270, 480), hfov, cam_h,
                                        forward_offset=1.0)
    assert np.isfinite(pushed).all()


def test_project_waypoints_yaw_only_rows_inherit_the_previous_position():
    wps = np.array([[1.0, 0.2, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    uv = project_waypoints_to_image(wps, (270, 480), 90.0, 0.5)
    assert np.array_equal(uv[1], uv[0])
    assert np.isnan(uv[2]).all()  # no yaw, no displacement: below min depth


@pytest.mark.parametrize("bad", [np.zeros((0, 3)), np.zeros((3, 2)), np.zeros(3), np.zeros((2, 3, 1))])
def test_project_waypoints_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        project_waypoints_to_image(bad, (270, 480), 90.0, 0.5)


@pytest.mark.parametrize("offset", [-0.1, math.nan, math.inf])
def test_project_waypoints_rejects_bad_forward_offsets(offset):
    with pytest.raises(ValueError):
        project_waypoints_to_image(np.ones((2, 3)), (270, 480), 90.0, 0.5, forward_offset=offset)
