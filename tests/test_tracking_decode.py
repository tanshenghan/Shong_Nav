"""decode_waypoints: parse '<tpos_k><traj_k>' text -> centroid waypoints. CPU-only."""

from __future__ import annotations

import numpy as np
import pytest

from lightnav.tracking import TrackingAgent


class _NullEngine:
    """TrackingAgent calls no engine method during decode."""


def _agent(centroids: np.ndarray) -> TrackingAgent:
    # num_history_frames is irrelevant to decode; buffer is untouched here.
    return TrackingAgent(engine=_NullEngine(), centroids=centroids, num_history_frames=4)


def test_decode_picks_centroid_by_traj_id(fake_centroids):
    agent = _agent(fake_centroids)
    wp, raw = agent.decode_waypoints("<tpos_0><traj_3>")
    assert raw == "<tpos_0><traj_3>"
    # fake_centroids[k][:,0] == k  -> traj_3 means forward_m all == 3.0
    assert np.allclose(wp[:, 0], 3.0)
    assert wp.shape == (fake_centroids.shape[1], 3)


def test_decode_v1_traj_only(fake_centroids):
    agent = _agent(fake_centroids)
    wp, raw = agent.decode_waypoints("<traj_1>")
    assert np.allclose(wp[:, 0], 1.0)


def test_decode_returns_a_copy_of_the_centroid(fake_centroids):
    agent = _agent(fake_centroids)
    wp, _ = agent.decode_waypoints("<traj_2>")
    wp[:] = 0.0
    assert np.allclose(agent.centroids[2, :, 0], 2.0)


def test_decode_out_of_range_raises(fake_centroids):
    agent = _agent(fake_centroids)
    with pytest.raises(ValueError):
        agent.decode_waypoints("<traj_99>")  # K=4, 99 is out of range


def test_decode_without_traj_token_raises(fake_centroids):
    agent = _agent(fake_centroids)
    with pytest.raises(ValueError):
        agent.decode_waypoints("i am not an action token")


def test_agent_requires_exactly_one_decoder(fake_centroids):
    with pytest.raises(ValueError):
        TrackingAgent(engine=_NullEngine(), num_history_frames=4)
    with pytest.raises(ValueError):
        TrackingAgent(
            engine=_NullEngine(), centroids=fake_centroids, num_history_frames=4, rvq_bundle=object()
        )


def test_agent_rejects_malformed_centroids():
    with pytest.raises(ValueError, match="shape"):
        TrackingAgent(engine=_NullEngine(), centroids=np.zeros((4, 10), np.float32))
