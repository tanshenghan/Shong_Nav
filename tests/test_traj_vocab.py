"""RVQ bundle loading / decoding and the SE(2) composition it relies on (synthetic bundles)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lightnav.traj_vocab import RVQBundle, compose_to_abs, load_rvq_bundle, wrap_to_pi

LEVELS = [4, 8, 8]
HORIZON = 10
FEAT = 3 * HORIZON


def test_load_rvq_bundle_ok(tmp_path, rvq_bundle_writer):
    rvq_bundle_writer(tmp_path)
    # The training-side kwargs are accepted for signature compatibility and ignored.
    b = load_rvq_bundle(tmp_path, HORIZON, 20, load_cluster_ids=True, load_distances=True)

    assert isinstance(b, RVQBundle)
    assert b.levels == LEVELS
    assert [c.shape for c in b.codebooks] == [(4, FEAT), (8, FEAT), (8, FEAT)]
    assert all(c.dtype == np.float32 for c in b.codebooks)
    assert b.jacobian_weights.shape == (FEAT,)
    assert b.horizon == HORIZON
    assert b.path == tmp_path
    assert b.representation == "se2_diff" and b.encode["heading_weight"] == 0.3
    assert b.objective == "ade_v1" and b.feature_space == "weighted_diff"
    assert b.alpha == {"mode": "none"}
    assert b.stop_l0 == 0
    assert b.manifest["method"] == "rvq"
    assert not hasattr(b, "cluster_ids")


def test_load_accepts_a_string_path(tmp_path, rvq_bundle_writer):
    rvq_bundle_writer(tmp_path)
    assert load_rvq_bundle(str(tmp_path), HORIZON).levels == LEVELS


def test_bundle_is_stop(tmp_path, rvq_bundle_writer):
    rvq_bundle_writer(tmp_path)
    b = load_rvq_bundle(tmp_path, HORIZON)
    assert b.stop_l0 == 0
    assert b.is_stop([0, 3, 5]) is True  # level-0 code == stop_l0
    assert b.is_stop([1, 0, 0]) is False  # different coarse code
    b.stop_l0 = None
    assert b.is_stop([0, 0, 0]) is False  # no stop code declared


def test_no_stop_block_means_no_stop_code(tmp_path, rvq_bundle_writer):
    rvq_bundle_writer(tmp_path, stop_l0=None)
    b = load_rvq_bundle(tmp_path, HORIZON)
    assert b.stop_l0 is None
    assert b.is_stop([0, 0, 0]) is False


def test_decode_waypoints_shape_and_level_check(tmp_path, rvq_bundle_writer):
    rvq_bundle_writer(tmp_path)
    b = load_rvq_bundle(tmp_path, HORIZON)
    wp = b.decode_waypoints([0, 0, 0])  # levels [4,8,8] -> 3 codes
    assert wp.shape == (HORIZON, 3) and wp.dtype == np.float32
    assert wp.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        b.decode_waypoints([0, 0])  # wrong number of codes


def test_horizon_mismatch_raises(tmp_path, rvq_bundle_writer):
    rvq_bundle_writer(tmp_path, horizon=HORIZON)
    with pytest.raises(RuntimeError):
        load_rvq_bundle(tmp_path, HORIZON + 1)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rvq_bundle(tmp_path / "nope", HORIZON)


def test_non_rvq_method_raises(tmp_path, rvq_bundle_writer):
    man = rvq_bundle_writer(tmp_path)
    man["method"] = "flat"
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="rvq"):
        load_rvq_bundle(tmp_path, HORIZON)


def test_codebook_shape_mismatch_raises(tmp_path, rvq_bundle_writer):
    rvq_bundle_writer(tmp_path)
    np.save(tmp_path / "codebook_l0.npy", np.zeros((4, FEAT + 1), np.float32))
    with pytest.raises(RuntimeError, match="codebook"):
        load_rvq_bundle(tmp_path, HORIZON)


def test_jacobian_shape_mismatch_raises(tmp_path, rvq_bundle_writer):
    rvq_bundle_writer(tmp_path)
    np.save(tmp_path / "jacobian_weights.npy", np.ones(FEAT - 1, np.float32))
    with pytest.raises(RuntimeError, match="jacobian"):
        load_rvq_bundle(tmp_path, HORIZON)


# -- SE(2) composition ---------------------------------------------------------------


def test_wrap_to_pi():
    theta = np.array([0.0, np.pi, -np.pi, 3 * np.pi / 2, -3 * np.pi / 2, 2 * np.pi, 0.5])
    np.testing.assert_allclose(
        wrap_to_pi(theta), [0.0, -np.pi, -np.pi, -np.pi / 2, np.pi / 2, 0.0, 0.5], atol=1e-12
    )


def test_compose_to_abs_pure_forward_accumulates():
    deltas = np.tile(np.array([0.25, 0.0, 0.0]), (10, 1))
    out = compose_to_abs(deltas)
    assert out.dtype == np.float32 and out.shape == (10, 3)
    np.testing.assert_allclose(out[:, 0], 0.25 * np.arange(1, 11), rtol=1e-6)
    assert np.all(out[:, 1:] == 0)


def test_compose_to_abs_rotates_subsequent_steps():
    deltas = np.array([[1.0, 0.0, math.pi / 2], [1.0, 0.0, 0.0]])
    out = compose_to_abs(deltas)
    # First delta is wp1's ego pose directly; the second step moves along the new heading.
    np.testing.assert_allclose(out[0], [1.0, 0.0, math.pi / 2], atol=1e-6)
    np.testing.assert_allclose(out[1], [1.0, 1.0, math.pi / 2], atol=1e-6)


def test_compose_to_abs_wraps_heading_and_supports_batches():
    deltas = np.zeros((2, 4, 3))
    deltas[..., 2] = math.pi / 2  # four quarter turns -> back to 0 (wrapped)
    out = compose_to_abs(deltas)
    assert out.shape == (2, 4, 3)
    np.testing.assert_allclose(out[:, 1, 2], -math.pi, atol=1e-6)  # pi wraps to -pi
    np.testing.assert_allclose(out[:, 3, 2], 0.0, atol=1e-6)
    with pytest.raises(ValueError):
        compose_to_abs(np.zeros((5, 2)))


def test_compose_to_abs_accumulates_in_float64():
    deltas = np.full((1000, 3), 1e-3, dtype=np.float32)
    deltas[:, 1:] = 0.0
    out = compose_to_abs(deltas)
    assert out[-1, 0] == pytest.approx(1.0, rel=1e-6)


# -- decode math --------------------------------------------------------------------


def _forward_codebooks(h: int = HORIZON):
    feat = 3 * h
    cb0 = np.zeros((2, feat), np.float32)
    cb0[1] = np.tile([0.25, 0.0, 0.0], h)  # constant forward diff per step
    cb1 = np.zeros((2, feat), np.float32)
    cb1[1] = np.tile([0.0, 0.0, 0.1], h)  # constant per-step yaw
    return cb0, cb1


def test_decode_se2_diff_composes_the_codeword_sum(tmp_path, rvq_bundle_writer):
    cb0, cb1 = _forward_codebooks()
    rvq_bundle_writer(tmp_path, levels=(2, 2), codebooks=[cb0, cb1])
    b = load_rvq_bundle(tmp_path, HORIZON)

    wp = b.decode_waypoints([1, 0])
    np.testing.assert_allclose(wp[:, 0], 0.25 * np.arange(1, 11), rtol=1e-6)
    assert np.all(wp[:, 1:] == 0)

    wp2 = b.decode_waypoints([1, 1])
    np.testing.assert_allclose(wp2[:, 2], 0.1 * np.arange(1, 11), atol=1e-6)  # heading accumulates
    assert wp2[1, 1] > 0  # turning left bends the path to +y (+lateral = left)
    expected = compose_to_abs(((cb0[1] + cb1[1]) / b.jacobian_weights).reshape(HORIZON, 3))
    np.testing.assert_allclose(wp2, expected, atol=1e-6)

    assert not b.decode_waypoints([0, 0]).any()  # all-zero codewords -> stationary


def test_decode_ego_abs_is_a_passthrough(tmp_path, rvq_bundle_writer):
    cb0, cb1 = _forward_codebooks()
    rvq_bundle_writer(tmp_path, levels=(2, 2), codebooks=[cb0, cb1], representation="ego_abs")
    b = load_rvq_bundle(tmp_path, HORIZON)
    wp = b.decode_waypoints([1, 1])
    np.testing.assert_allclose(wp, (cb0[1] + cb1[1]).reshape(HORIZON, 3), atol=1e-7)
    assert wp.dtype == np.float32


def test_jacobian_weights_deweight_the_codeword_sum(tmp_path, rvq_bundle_writer):
    cb0, cb1 = _forward_codebooks()
    rvq_bundle_writer(
        tmp_path, levels=(2, 2), codebooks=[cb0, cb1], representation="ego_abs",
        jacobian=np.full(FEAT, 2.0, np.float32),
    )
    b = load_rvq_bundle(tmp_path, HORIZON)
    wp = b.decode_waypoints([1, 0])
    np.testing.assert_allclose(wp[:, 0], 0.125, atol=1e-7)


def test_rvq_bundle_can_be_constructed_directly():
    cb0, cb1 = _forward_codebooks()
    b = RVQBundle(
        path=None,
        manifest={},
        levels=[2, 2],
        horizon=HORIZON,
        representation="se2_diff",
        objective="ade_v1",
        encode={"type": "ade", "heading_weight": 0.3},
        feature_space="weighted_diff",
        codebooks=[cb0, cb1],
        jacobian_weights=np.ones(FEAT, np.float32),
        alpha={},
        stop_l0=1,
    )
    assert b.is_stop([1, 0]) is True
    assert b.decode_waypoints([1, 0])[0, 0] == pytest.approx(0.25)
