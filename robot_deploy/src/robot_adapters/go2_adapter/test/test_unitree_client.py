from types import SimpleNamespace

from go2_adapter.unitree_client import (
    UnitreeClient,
    battery_from_message,
    normalized_mode,
    state_from_message,
)


class _FakeObstacles:
    def __init__(self):
        self.calls = []

    def UseRemoteCommandFromApi(self, enabled):
        self.calls.append(("api", enabled))
        return 0

    def Move(self, linear, lateral, angular):
        self.calls.append(("move", linear, lateral, angular))
        return 0


def _control_client():
    client = UnitreeClient.__new__(UnitreeClient)
    client._api_enabled = False
    client._obstacles = _FakeObstacles()
    client.events = []
    client.errors = []
    client.sent = []
    client._event_callback = client.events.append
    client._error_callback = client.errors.append
    client._sent_callback = client.sent.append
    return client


def test_state_from_message_converts_wxyz_to_xyzw():
    message = SimpleNamespace(
        position=[1.0, 2.0, 0.3],
        velocity=[0.4, 0.1, 0.0],
        yaw_speed=-0.2,
        body_height=0.31,
        mode=3,
        gait_type=1,
        error_code=0,
        imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0]),
    )
    state = state_from_message(
        message, received_ns=123, received_s=4.5
    )
    assert state.received_ns == 123
    assert state.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert state.position == (1.0, 2.0, 0.3)
    assert state.velocity == (0.4, 0.1, 0.0)


def test_normalized_modes_match_shared_interface():
    assert normalized_mode(0) == "STAND"
    assert normalized_mode(1) == "WALK"
    assert normalized_mode(3) == "WALK"
    assert normalized_mode(7) == "DAMPING"
    assert normalized_mode(10) == "SIT"
    assert normalized_mode(99) == "UNKNOWN"
    assert (
        normalized_mode(0, error_code=1001, body_height=0.07)
        == "DAMPING"
    )
    assert normalized_mode(0, error_code=100, body_height=0.31) == "WALK"


def test_battery_from_message():
    message = SimpleNamespace(
        bms_state=SimpleNamespace(
            soc=73,
            status=1,
            current=0,
            cycle=0,
            cell_vol=[3900] * 15,
        ),
        motor_state=[SimpleNamespace(mode=1, temperature=32)] * 12,
        power_v=29.4,
    )
    battery = battery_from_message(message, received_s=1.0)
    assert battery.percentage == 73.0
    assert battery.voltage == 29.4
    assert battery.motors_ok is True


def test_unpopulated_bms_does_not_report_false_zero_percentage():
    message = SimpleNamespace(
        bms_state=SimpleNamespace(
            soc=0,
            status=0,
            current=0,
            cycle=0,
            cell_vol=[0] * 15,
        ),
        motor_state=[SimpleNamespace(mode=1, temperature=32)] * 12,
        power_v=28.6,
    )
    battery = battery_from_message(message, received_s=1.0)
    assert battery.percentage is None


def test_api_control_is_acquired_lazily_and_released_after_zero():
    client = _control_client()
    assert client._enable_api_control() is True
    assert client._api_enabled is True
    client._release_api_control()
    assert client._api_enabled is False
    assert client._obstacles.calls == [
        ("api", True),
        ("move", 0.0, 0.0, 0.0),
        ("api", False),
    ]
    assert client.sent == [(0.0, 0.0, 0.0)]
    assert client.errors == []
