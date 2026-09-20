from go2_adapter.safety import select_command


def _arguments():
    return {
        "source": "disabled",
        "robot_mode": "WALK",
        "manual_command": (0.0, 0.0),
        "manual_age_s": 0.0,
        "auto_command": (0.0, 0.0),
        "auto_age_s": 0.0,
        "watchdog_s": 0.35,
    }


def test_selects_auto_command_without_clamping():
    arguments = _arguments()
    arguments.update(source="auto", auto_command=(3.0, -2.0))
    assert select_command(**arguments) == (3.0, -2.0)


def test_rejects_command_when_not_walk_or_stale():
    arguments = _arguments()
    arguments.update(source="manual", manual_command=(0.5, 0.2))
    arguments["robot_mode"] = "DAMPING"
    assert select_command(**arguments) is None
    arguments["robot_mode"] = "WALK"
    arguments["manual_age_s"] = 0.5
    assert select_command(**arguments) is None


def test_accepts_command_while_upright_default_stand_mode():
    arguments = _arguments()
    arguments.update(
        source="manual",
        robot_mode="STAND",
        manual_command=(0.5, 0.2),
    )
    assert select_command(**arguments) == (0.5, 0.2)
