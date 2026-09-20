"""Cross-generation checkpoint compatibility: one matrix over every output layout.

Every checkpoint generation this server can be pointed at emits a different token
string. This file drives the FULL server-side decode chain for each of them --
waypoint decode, wire signals and the generation-cap probe -- so a change that fixes
the newest family cannot silently break an older one (or vice versa).

Layouts, oldest to newest:

    v1 flat            <traj_K>
    v2 flat            <tpos_K><traj_K>
    v2 RVQ             <tpos_K><act_l0_*>..<act_lD_*>
    vln RVQ, legacy    <act_l0_*>..<act_lD_*>                    (no grounding)
    tracking pointing  <opos_K><act_l0_*>..                      (opos replaces tpos)
    nav pointing       <apos_A><opos_O><act_l0_*>..
"""

from __future__ import annotations

import numpy as np
import pytest

from lightnav.serving.protocol import decode_prediction_signals
from lightnav.serving.token_budget import decode_token_budget, probe_grounding_tokens
from lightnav.tracking import TrackingAgent
from lightnav.vln_utils import (
    APOS_STOP_ID,
    encode_point_pixel,
    encode_target_pos,
    rvq_action_token,
    tpos_token,
    traj_token,
)

H, K, LEVELS = 10, 256, [256, 256, 256]


class _StubRvq:
    """Minimal RVQBundle surface used by TrackingAgent.decode_waypoints."""

    levels = LEVELS
    horizon = H

    @staticmethod
    def is_stop(codes) -> bool:
        return list(codes) == [0, 0, 0]

    @staticmethod
    def decode_waypoints(codes) -> np.ndarray:
        wp = np.zeros((H, 3), dtype=np.float32)
        wp[:, 0] = np.linspace(0.3, 3.0, H) * (1 + sum(codes) / 768.0)
        return wp


def _act(codes=(1, 2, 3)) -> str:
    return "".join(rvq_action_token(lvl, c) for lvl, c in enumerate(codes))


VISIBLE_TPOS = encode_target_pos(1.0, 0.2, 3.0)
INVISIBLE_TPOS = encode_target_pos(0.0, 0.0, 0.0)
OPOS_VISIBLE = encode_point_pixel(200, 300, 640, 480)
APOS_POINT = encode_point_pixel(400, 260, 640, 480)

# (id, raw_text, is_rvq, expect_visible, expect_tpos, expect_apos, expect_opos)
CASES = [
    ("v1_flat", traj_token(7), False, None, None, None, None),
    ("v1_flat_stop", traj_token(0), False, None, None, None, None),
    ("v2_flat_visible", tpos_token(VISIBLE_TPOS) + traj_token(7), False, True, VISIBLE_TPOS, None, None),
    ("v2_flat_lost", tpos_token(INVISIBLE_TPOS) + traj_token(7), False, False, INVISIBLE_TPOS, None, None),
    ("v2_rvq", tpos_token(VISIBLE_TPOS) + _act(), True, True, VISIBLE_TPOS, None, None),
    ("vln_rvq_legacy", _act(), True, None, None, None, None),
    ("tracking_pointing", f"<opos_{OPOS_VISIBLE}>" + _act(), True, True, None, None, OPOS_VISIBLE),
    (
        "nav_pointing",
        f"<apos_{APOS_POINT}><opos_{OPOS_VISIBLE}>" + _act(),
        True,
        True,
        None,
        APOS_POINT,
        OPOS_VISIBLE,
    ),
    ("nav_pointing_lost", f"<apos_{APOS_POINT}><opos_0>" + _act(), True, False, None, APOS_POINT, 0),
    (
        "nav_pointing_stop",
        f"<apos_{APOS_STOP_ID}><opos_0>" + _act((0, 0, 0)),
        True,
        False,
        None,
        APOS_STOP_ID,
        0,
    ),
]
IDS = [c[0] for c in CASES]


@pytest.fixture
def flat_agent():
    centroids = np.zeros((K, H, 3), dtype=np.float32)
    centroids[7, :, 0] = np.linspace(0.3, 3.0, H)  # id 0 stays all-zero == stop
    return TrackingAgent(engine=object(), centroids=centroids, num_history_frames=8)


@pytest.fixture
def rvq_agent():
    return TrackingAgent(engine=object(), num_history_frames=8, rvq_bundle=_StubRvq())


@pytest.mark.parametrize(
    ("name", "raw", "is_rvq", "visible", "tpos", "apos", "opos"),
    CASES,
    ids=IDS,
)
def test_signals_decode_for_every_checkpoint_generation(
    name, raw, is_rvq, visible, tpos, apos, opos, flat_agent, rvq_agent
):
    agent = rvq_agent if is_rvq else flat_agent
    waypoints, echoed = agent.decode_waypoints(raw)

    assert echoed == raw
    assert waypoints.shape == (H, 3)
    assert waypoints.dtype == np.float32

    signals = decode_prediction_signals(
        raw, vocab_size=None if is_rvq else K, is_rvq=is_rvq, waypoints=waypoints
    )
    assert signals.visible is visible, f"{name}: visible"
    assert signals.tpos_id == tpos, f"{name}: tpos_id"
    assert signals.apos_id == apos, f"{name}: apos_id"
    assert signals.opos_id == opos, f"{name}: opos_id"


@pytest.mark.parametrize(("name", "raw"), [(c[0], c[1]) for c in CASES], ids=IDS)
def test_stop_is_reported_for_the_stop_action_of_each_generation(name, raw, flat_agent, rvq_agent):
    is_rvq = dict((c[0], c[2]) for c in CASES)[name]
    agent = rvq_agent if is_rvq else flat_agent
    waypoints, _ = agent.decode_waypoints(raw)
    signals = decode_prediction_signals(
        raw, vocab_size=None if is_rvq else K, is_rvq=is_rvq, waypoints=waypoints
    )

    expect_stop = name in ("v1_flat_stop", "nav_pointing_stop")
    assert signals.stop is expect_stop, f"{name}: stop"
    if expect_stop:
        assert not waypoints.any(), f"{name}: a stop must zero the waypoints"


def test_generation_cap_probe_covers_every_family():
    """The probed cap must be >= what each family actually emits (it may overshoot)."""

    class Tok:
        unk_token_id = 0

        def __init__(self, present):
            self._p = set(present)

        def convert_tokens_to_ids(self, t):
            return 100 if t in self._p else 0

    rvq = _StubRvq()
    emitted = {  # vocab present -> tokens a step really emits
        ("<tpos_0>",): 1 + len(LEVELS),  # v2 rvq tracking
        ("<apos_0>",): 2 + len(LEVELS),  # nav pointing
        ("<apos_0>", "<tpos_0>"): 2 + len(LEVELS),  # mixed vocab, nav source
        (): len(LEVELS),  # legacy vln rvq
    }
    for present, actual in emitted.items():
        cap = decode_token_budget(probe_grounding_tokens(Tok(present)), rvq)
        assert cap >= actual, f"{present}: cap {cap} would truncate {actual} tokens"


def test_rvq_missing_level_is_a_loud_error_not_a_silent_short_decode(rvq_agent):
    """A truncated generation (cap too small) must raise, never yield 2-level waypoints."""
    from lightnav.vln_utils import parse_rvq_action_tokens

    with pytest.raises(ValueError, match="Missing rvq act levels"):
        parse_rvq_action_tokens("<act_l0_1><act_l2_3>")
    with pytest.raises(ValueError, match="expected 3"):
        rvq_agent.decode_waypoints("<act_l0_1><act_l1_2>")


def test_rvq_code_out_of_range_is_rejected(rvq_agent):
    with pytest.raises(ValueError, match="out of range"):
        rvq_agent.decode_waypoints(_act((1, 2, 300)))


def test_rvq_near_zero_decode_is_reported_as_an_exact_stop():
    """A decode within the RVQ stop tolerance becomes exact zeros (= stop on the wire)."""

    class NearZeroRvq(_StubRvq):
        @staticmethod
        def is_stop(codes) -> bool:
            return False

        @staticmethod
        def decode_waypoints(codes) -> np.ndarray:
            return np.full((H, 3), 1e-3, dtype=np.float32)

    agent = TrackingAgent(engine=object(), num_history_frames=8, rvq_bundle=NearZeroRvq())
    waypoints, _ = agent.decode_waypoints(_act())
    assert not waypoints.any()
    assert decode_prediction_signals(_act(), is_rvq=True, waypoints=waypoints).stop is True

    strict = TrackingAgent(
        engine=object(), num_history_frames=8, rvq_bundle=NearZeroRvq(), stop_atol=1e-6
    )
    waypoints, _ = strict.decode_waypoints(_act())
    assert waypoints.any()
