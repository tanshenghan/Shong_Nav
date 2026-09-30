from concurrent.futures import Future
from types import SimpleNamespace

from vln_mujoco.mpc import MpcTracker
from vln_mujoco.server import VlnMujocoServer


def test_stop_clears_path_command_and_rejects_inflight_mpc():
    tracker = MpcTracker.__new__(MpcTracker)
    tracker._generation = 1
    tracker._future = Future()
    tracker._future.set_result(SimpleNamespace(generation=1, command=(.8, .5)))
    calls = []
    server = VlnMujocoServer.__new__(VlnMujocoServer)
    server.simulation = SimpleNamespace(
        set_velocity=lambda *v: calls.append(('velocity', v)),
        set_path=lambda: calls.append(('path', ())),
    )
    server.vln = SimpleNamespace(stop=lambda: calls.append(('vln', ())))
    server.mpc = tracker
    server.capture_poses = {1: (0, 0, 0)}
    server.auto_owner = object()
    server.stop_vln()
    assert server.auto_owner is None
    assert server.auto_command == (0, 0)
    assert server.waypoints == ()
    assert server.capture_poses == {}
    assert calls[0] == ('velocity', (0, 0))
    assert ('path', ()) in calls
    assert tracker.poll() is None
    assert tracker.command == (0, 0)
