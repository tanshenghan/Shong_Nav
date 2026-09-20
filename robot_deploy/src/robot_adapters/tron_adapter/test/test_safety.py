from tron_adapter.safety import select_command


def arguments(**overrides):
    values = {
        "source": "manual",
        "robot_mode": "WALK",
        "manual_command": (0.4, -0.3),
        "manual_age_s": 0.1,
        "auto_command": (0.2, 0.1),
        "auto_age_s": 0.1,
        "watchdog_s": 0.35,
    }
    values.update(overrides)
    return values


def test_manual_command_requires_walk_and_fresh_input():
    assert select_command(**arguments()) == (0.4, -0.3)
    assert select_command(**arguments(robot_mode="DAMPING")) is None
    assert select_command(**arguments(manual_age_s=0.36)) is None


def test_auto_command_is_forwarded_without_clamping():
    assert select_command(
        **arguments(source="auto", auto_command=(2.0, -3.0))
    ) == (2.0, -3.0)


def test_disabled_or_non_finite_command_is_rejected():
    assert select_command(**arguments(source="disabled")) is None
    assert select_command(
        **arguments(manual_command=(float("nan"), 0.0))
    ) is None
