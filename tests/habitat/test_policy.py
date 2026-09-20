"""TrajVocabVLNCEPolicy: trajectory tokens -> Habitat ``velocity_control`` action dicts.

CPU-only: the engine is a stub that returns canned text, and frame preprocessing is
replaced by a tiny zero tensor so the buffer plumbing is exercised without a resize.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.habitat.policy import (
    TrajVocabVLNCEPolicy,
    extract_instruction,
    select_action_waypoint,
)
from lightnav.inference.policies import NavigationPolicy
from lightnav.traj_vocab import RVQBundle


class StubEngine:
    """Only the surface the policy touches: ``bundle``, ``reset_episode_state`` and
    ``generate_from_frames`` (which must be called with ``task_type="vlnce_traj"``)."""

    def __init__(self, text: str = "", action_method: str | None = None) -> None:
        bundle = SimpleNamespace(video_size=(224, 224))
        if action_method is not None:
            bundle.action_method = action_method
        self.bundle = bundle
        self._text = text
        self.calls: list = []

    def reset_episode_state(self) -> None:
        self.calls.append(("reset",))

    def generate_from_frames(
        self, video, instruction, predict_horizon=1, frame_ids=None, task_type="vlnce"
    ):
        self.calls.append(
            (
                "gen_frames",
                tuple(video.shape),
                instruction,
                predict_horizon,
                list(frame_ids or []),
                task_type,
            )
        )
        return self._text, 1.0


@pytest.fixture(autouse=True)
def _stub_convert_frame(monkeypatch):
    # Skip the real (slow) frame preprocessing; shape must be (3, H, W).
    monkeypatch.setattr(NavigationPolicy, "_convert_frame", lambda self, frame: torch.zeros(3, 4, 4))


def _rgb_frame() -> np.ndarray:
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _build_centroids() -> np.ndarray:
    """K=4, H=10, 3. [0]=STOP zeros, [1]=pure forward 0.25m, [2]/[3] arbitrary."""
    centroids = np.zeros((4, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25  # all 10 waypoints = forward 0.25m
    centroids[2, :, 2] = 0.1  # small yaw
    centroids[3, :, 0] = 0.5
    return centroids


def _build_policy(engine: StubEngine, centroids: np.ndarray | None = None) -> TrajVocabVLNCEPolicy:
    return TrajVocabVLNCEPolicy(
        engine,
        _build_centroids() if centroids is None else centroids,
        dt=1.0,
        lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0),
        num_history_frames=4,
    )


def _reset_and_act(policy: TrajVocabVLNCEPolicy, instruction: str = "go") -> dict:
    policy.reset({"rgb": _rgb_frame(), "instruction": {"text": instruction}})
    return policy.act({"rgb": _rgb_frame()}, info={})


# -- helpers -----------------------------------------------------------------------------


def test_extract_instruction_accepts_dict_str_and_missing():
    assert extract_instruction(None) == ""
    assert extract_instruction({}) == ""
    assert extract_instruction({"instruction": None}) == ""
    assert extract_instruction({"instruction": {"text": "turn left"}}) == "turn left"
    assert extract_instruction({"instruction": {}}) == ""
    assert extract_instruction({"instruction": "go straight"}) == "go straight"


def test_select_action_waypoint_skips_leading_origin_rows():
    wp = np.zeros((4, 3), dtype=np.float32)
    wp[2, 0] = 0.2
    idx, row = select_action_waypoint(wp)
    assert idx == 2 and row[0] == pytest.approx(0.2)


def test_select_action_waypoint_ignores_lateral_only_rows():
    """Only forward / yaw can drive a unicycle; a pure lateral row is still 'origin'."""
    wp = np.zeros((3, 3), dtype=np.float32)
    wp[0, 1] = 0.5  # lateral only
    wp[1, 2] = 0.01  # yaw
    idx, _ = select_action_waypoint(wp)
    assert idx == 1


def test_select_action_waypoint_falls_back_to_row_zero_when_all_zero():
    wp = np.zeros((5, 3), dtype=np.float32)
    idx, row = select_action_waypoint(wp)
    assert idx == 0
    np.testing.assert_array_equal(row, wp[0])


# -- flat vocabulary ---------------------------------------------------------------------


def test_zero_cluster_returns_velocity_stop_command() -> None:
    """centroids[0] is all zeros -> policy returns a zero-speed velocity dict."""
    engine = StubEngine("<traj_0>")
    policy = _build_policy(engine)

    action = _reset_and_act(policy)

    assert isinstance(action, dict)
    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == pytest.approx(-1.0)
    assert action["action_args"]["angular_velocity"] == pytest.approx(0.0)
    info = policy.get_info()
    assert info["cluster_id"] == 0
    assert info["rvq_codes"] is None
    assert info["predicted_traj"].shape == (10, 3)
    assert info["action_waypoint_index"] == 0
    # The engine saw the VLN-CE trajectory prompt family with the buffered frame.
    gen = engine.calls[-1]
    assert gen[0] == "gen_frames"
    assert gen[1] == (1, 3, 4, 4)
    assert gen[2] == "go"
    assert gen[3] == 1
    assert gen[4] == [0]
    assert gen[5] == "vlnce_traj"


def test_near_zero_but_nonzero_centroid_returns_velocity() -> None:
    """A legitimate small-motion cluster (~0 but not literally zero) must NOT be
    routed to a stop -- it produces a (small) velocity command."""
    centroids = np.zeros((4, 10, 3), dtype=np.float32)
    # 5mm forward + 0.5 deg yaw: the kind of cluster that would trip Habitat's
    # min_abs_* thresholds under the default 0.025 m/s / 1 deg/s settings.
    centroids[0, :, 0] = 0.005
    centroids[0, :, 2] = np.deg2rad(0.5)
    centroids[1, :, 0] = 0.25

    engine = StubEngine("<traj_0>")
    policy = TrajVocabVLNCEPolicy(
        engine,
        centroids,
        dt=1.0,
        lin_vel_range=(0.0, 1.5),
        ang_vel_range=(-45.0, 45.0),
        num_history_frames=4,
    )
    action = _reset_and_act(policy, "creep forward")

    assert action["action"] == "velocity_control"
    # 0.005 / 1.0 = 0.005 m/s -> raw close to -1 but > -1.
    assert action["action_args"]["linear_velocity"] > -1.0
    assert action["action_args"]["linear_velocity"] < -0.99


def test_pure_forward_cluster_maps_to_expected_raw() -> None:
    """centroids[1] is pure forward 0.25m per step; check raw normalization math."""
    policy = _build_policy(StubEngine("<traj_1>"))

    action = _reset_and_act(policy)

    # forward_m=0.25, dt=1 -> lin_mps=0.25; range (0, 0.3): raw = 2*(0.25-0)/0.3 - 1 = 2/3
    assert action["action_args"]["linear_velocity"] == pytest.approx(2.0 / 3.0)
    assert action["action_args"]["angular_velocity"] == pytest.approx(0.0)
    assert policy.get_info()["action_waypoint_index"] == 0


def test_leading_zero_waypoint_is_skipped() -> None:
    """Some vocab entries start at the origin before motion; use the first moving waypoint."""
    centroids = np.zeros((4, 10, 3), dtype=np.float32)
    centroids[1, 1, 0] = 0.2
    centroids[1, 2:, 0] = 0.4

    policy = _build_policy(StubEngine("<traj_1>"), centroids)

    action = _reset_and_act(policy)

    assert action["action_args"]["linear_velocity"] == pytest.approx(1.0 / 3.0)
    assert action["action_args"]["angular_velocity"] == pytest.approx(0.0)
    info = policy.get_info()
    assert info["action_waypoint_index"] == 1
    np.testing.assert_array_equal(info["action_waypoint"], centroids[1, 1])


@pytest.mark.parametrize("text", ["<traj_999>", "blah blah", ""])
def test_unparseable_or_out_of_range_output_falls_back_to_zero_command(text) -> None:
    """<traj_999> for K=4 / plain text -> zero-velocity fallback (Habitat treats it as stop)."""
    policy = _build_policy(StubEngine(text))

    action = _reset_and_act(policy)

    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == pytest.approx(-1.0)
    assert action["action_args"]["angular_velocity"] == pytest.approx(0.0)
    assert policy._last_cluster_id == -1
    assert policy.get_info().get("cluster_id", None) in (-1, None)
    assert "predicted_traj" not in policy.get_info()


def test_get_info_after_successful_act() -> None:
    """get_info() returns predicted_traj of the right shape and the cluster id."""
    policy = _build_policy(StubEngine("<traj_2>"))
    policy.reset({"rgb": _rgb_frame(), "instruction": {"text": "go"}})

    # Before any act, get_info has no traj info.
    pre = policy.get_info()
    assert "predicted_traj" not in pre
    assert pre["num_history_frames"] == 4

    policy.act({"rgb": _rgb_frame()}, info={})
    info = policy.get_info()
    assert info["cluster_id"] == 2
    assert info["raw_text"] == "<traj_2>"
    assert info["predicted_traj"].shape == (10, 3)
    assert info["action_waypoint_index"] == 0
    expected = np.zeros((10, 3), dtype=np.float32)
    expected[:, 2] = 0.1
    np.testing.assert_array_equal(info["predicted_traj"], expected)


def test_predicted_traj_is_a_copy_of_the_centroid() -> None:
    policy = _build_policy(StubEngine("<traj_3>"))
    _reset_and_act(policy)
    policy.get_info()["predicted_traj"][:] = 0.0
    assert policy.centroids[3, 0, 0] == pytest.approx(0.5)


def test_reset_clears_last_state() -> None:
    """A successful act() then reset() should clear cached cluster/traj."""
    engine = StubEngine("<traj_1>")
    policy = _build_policy(engine)
    policy.reset({"rgb": _rgb_frame()})
    policy.act({"rgb": _rgb_frame()}, info={})
    assert policy.get_info().get("cluster_id") == 1

    policy.reset({"rgb": _rgb_frame()})
    info = policy.get_info()
    assert "cluster_id" not in info
    assert "predicted_traj" not in info
    assert "action_waypoint_index" not in info
    assert policy.instruction == ""  # no instruction in that obs
    assert engine.calls.count(("reset",)) == 2


def test_reset_seeds_the_instruction_and_frame_ids_restart() -> None:
    engine = StubEngine("<traj_1>")
    policy = _build_policy(engine)
    policy.reset({"rgb": _rgb_frame(), "instruction": {"text": "first"}})
    policy.act({"rgb": _rgb_frame()}, info={})
    policy.act({"rgb": _rgb_frame()}, info={})
    assert engine.calls[-1][4] == [0, 1]

    policy.reset({"rgb": _rgb_frame(), "instruction": {"text": "second"}})
    policy.act({"rgb": _rgb_frame()}, info={})
    assert policy.instruction == "second"
    assert engine.calls[-1][2] == "second"
    assert engine.calls[-1][4] == [0]


def test_history_window_is_bounded_by_num_history_frames() -> None:
    engine = StubEngine("<traj_1>")
    policy = _build_policy(engine)  # num_history_frames=4
    policy.reset({"rgb": _rgb_frame(), "instruction": {"text": "go"}})
    for _ in range(6):
        policy.act({"rgb": _rgb_frame()}, info={})

    gen = engine.calls[-1]
    assert gen[1] == (4, 3, 4, 4)
    assert gen[4] == [2, 3, 4, 5]


def test_observe_does_not_run_inference() -> None:
    """observe() must push the frame into history but never call the engine."""
    engine = StubEngine("<traj_1>")
    policy = _build_policy(engine)
    policy.reset({"rgb": _rgb_frame()})

    policy.observe({"rgb": _rgb_frame()}, info={})
    policy.observe({"rgb": _rgb_frame()}, info={})

    assert all(call[0] != "gen_frames" for call in engine.calls)
    assert policy.agent._history_frame_ids == [0, 1]


def test_centroids_shape_validation() -> None:
    bad = np.zeros((4, 10), dtype=np.float32)  # missing last dim
    with pytest.raises(ValueError, match="shape"):
        _build_policy(StubEngine(), bad)


def test_centroids_min_k_validation() -> None:
    bad = np.zeros((1, 10, 3), dtype=np.float32)  # K must be >= 2
    with pytest.raises(ValueError, match="K>=2"):
        _build_policy(StubEngine(), bad)


def test_centroids_can_be_loaded_from_an_npy_path(tmp_path) -> None:
    path = tmp_path / "centroids_whole_chunk_K4_h10.npy"
    np.save(path, _build_centroids())
    policy = _build_policy(StubEngine("<traj_1>"), path)
    assert policy.K == 4 and policy.H == 10
    action = _reset_and_act(policy)
    assert action["action_args"]["linear_velocity"] == pytest.approx(2.0 / 3.0)


def test_non_positive_dt_is_rejected() -> None:
    with pytest.raises(ValueError, match="dt"):
        TrajVocabVLNCEPolicy(
            StubEngine(),
            _build_centroids(),
            dt=0.0,
            lin_vel_range=(0.0, 0.3),
            ang_vel_range=(-30.0, 30.0),
        )


# -- RVQ decode path: D <act_l*> tokens -> codeword sum -> SE(2) compose -----------------


def _build_rvq_bundle(stop_l0: int | None = None) -> RVQBundle:
    """levels [2,2], H=10, jac=ones. l0[1] = pure-forward 0.25m/step diff; everything
    else zero. So codes [1,0] -> constant forward, codes [0,0] -> stop."""
    H, feat = 10, 30
    cb0 = np.zeros((2, feat), dtype=np.float32)
    cb0[1] = np.tile(np.array([0.25, 0.0, 0.0], dtype=np.float32), H)
    cb1 = np.zeros((2, feat), dtype=np.float32)
    return RVQBundle(
        path=None,
        manifest={},
        levels=[2, 2],
        horizon=H,
        representation="se2_diff",
        objective="ade_v1",
        encode={"type": "ade", "heading_weight": 0.3},
        feature_space="weighted_diff",
        codebooks=[cb0, cb1],
        jacobian_weights=np.ones(feat, dtype=np.float32),
        alpha={},
        stop_l0=stop_l0,
    )


def _build_rvq_policy(engine: StubEngine, bundle: RVQBundle | None = None) -> TrajVocabVLNCEPolicy:
    return TrajVocabVLNCEPolicy(
        engine,
        None,
        dt=1.0,
        lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0),
        num_history_frames=4,
        rvq_bundle=bundle or _build_rvq_bundle(),
    )


def test_rvq_forward_tokens_produce_forward_velocity():
    engine = StubEngine("<act_l0_1><act_l1_0>")
    policy = _build_rvq_policy(engine)

    action = _reset_and_act(policy)

    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] > -1.0  # moving, not the stop sentinel
    assert action["action_args"]["linear_velocity"] == pytest.approx(2.0 / 3.0)
    assert action["action_args"]["angular_velocity"] == pytest.approx(0.0)
    info = policy.get_info()
    assert info["rvq_codes"] == [1, 0]
    assert info["cluster_id"] == 1  # level-0 code
    assert info["predicted_traj"].shape == (10, 3)
    assert info["predicted_traj"][0, 0] == pytest.approx(0.25, abs=1e-5)
    assert engine.calls[-1][-1] == "vlnce_traj"
    assert policy.method == "rvq" and policy.K is None and policy.H == 10


def test_rvq_explicit_stop_code_forces_stop():
    """codes[0]==stop_l0 forces a clean stop even if that codeword would decode nonzero."""
    bundle = _build_rvq_bundle(stop_l0=1)  # stop_l0=1 decodes to forward!
    engine = StubEngine("<act_l0_1><act_l1_0>")
    policy = _build_rvq_policy(engine, bundle)

    action = _reset_and_act(policy, "stop")

    assert action["action_args"]["linear_velocity"] == pytest.approx(-1.0)  # forced stop
    assert np.all(policy.get_info()["predicted_traj"] == 0.0)


def test_rvq_zero_tokens_produce_stop():
    policy = _build_rvq_policy(StubEngine("<act_l0_0><act_l1_0>"))
    action = _reset_and_act(policy, "stop")
    assert action["action_args"]["linear_velocity"] == pytest.approx(-1.0)
    assert action["action_args"]["angular_velocity"] == pytest.approx(0.0)


def test_rvq_parse_failure_falls_back_to_zero():
    policy = _build_rvq_policy(StubEngine("i am not an action token"))
    action = _reset_and_act(policy, "")
    assert action["action_args"]["linear_velocity"] == pytest.approx(-1.0)
    assert policy.get_info().get("cluster_id", None) in (-1, None)


def test_rvq_wrong_level_count_falls_back():
    policy = _build_rvq_policy(StubEngine("<act_l0_1>"))  # only 1 level, bundle expects 2
    action = _reset_and_act(policy, "")
    assert action["action_args"]["linear_velocity"] == pytest.approx(-1.0)


def test_rvq_out_of_range_code_falls_back():
    policy = _build_rvq_policy(StubEngine("<act_l0_1><act_l1_5>"))  # level 1 has 2 codes
    action = _reset_and_act(policy, "")
    assert action["action_args"]["linear_velocity"] == pytest.approx(-1.0)


def test_policy_rejects_both_or_neither_source():
    eng = StubEngine()
    with pytest.raises(ValueError):
        TrajVocabVLNCEPolicy(
            eng,
            _build_centroids(),
            dt=1.0,
            lin_vel_range=(0.0, 0.3),
            ang_vel_range=(-30.0, 30.0),
            rvq_bundle=_build_rvq_bundle(),
        )
    with pytest.raises(ValueError):
        TrajVocabVLNCEPolicy(
            eng, None, dt=1.0, lin_vel_range=(0.0, 0.3), ang_vel_range=(-30.0, 30.0)
        )


# -- decoder must agree with the checkpoint's action_method -------------------------------


def test_decoder_must_match_the_checkpoint_action_method():
    with pytest.raises(ValueError, match="rvq"):
        _build_policy(StubEngine(action_method="rvq"))
    with pytest.raises(ValueError, match="flat"):
        _build_rvq_policy(StubEngine(action_method="flat"))


def test_matching_action_method_is_accepted():
    assert _build_policy(StubEngine(action_method="flat")).method == "flat"
    assert _build_rvq_policy(StubEngine(action_method="rvq")).method == "rvq"


def test_stop_action_is_the_zero_velocity_command_without_touching_the_model() -> None:
    engine = StubEngine("<traj_3>")
    policy = _build_policy(engine)
    action = policy.stop_action()
    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == pytest.approx(-1.0)
    assert action["action_args"]["angular_velocity"] == pytest.approx(0.0)
    assert engine.calls == []  # no inference, no frame consumed
