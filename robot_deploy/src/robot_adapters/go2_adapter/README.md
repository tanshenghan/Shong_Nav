# go2_adapter

Connects the Go2 to the project's common ROS 2 robot interface through
`unitree_sdk2py`.

## Interface

- Subscribes: `web/cmd_vel`, `mpc/cmd_vel`
- Publishes: `odom`, `cmd_vel`, `control/source`, `diagnostics`, `robot/events`
- Services: `control/set_manual`, `control/set_auto`, `control/stop`
- Services: `robot/stand`, `robot/walk`, `robot/sit`, `robot/emergency_stop`

On the Go2, `robot/sit` first releases velocity control, then calls
`StandDown()` for a controlled lie-down.

`odom` directly forwards the Go2's `rt/lf/sportmodestate`; low-level state comes
from `rt/lowstate`. Some firmware does not populate its BMS percentage, in which
case the web page shows an empty battery reading instead of misreporting an
invalid value as 0%. Velocities are sent through `ObstaclesAvoidClient.Move()`
and mode commands through `SportClient`.

While the control source is `disabled`, the adapter does not acquire Unitree API
control. It acquires control only when the first valid manual or MPC velocity
command arrives; when the control source turns off, it sends a zero velocity
first and then hands control back to the remote.

The robot may keep reporting Unitree `mode=0` after the remote's `START` unlock.
The adapter maps `mode=0` with a body height of at least 0.2 m to the common
interface's `WALK`; a low body height or `error_code=1001` still maps to
`DAMPING`.

## Running

The system needs Unitree's official `unitree_sdk2_python` and the
CycloneDDS 0.10.2 it requires installed beforehand. The default network
interface is `enP8p1s0`:

```bash
ros2 launch vln_bringup go2.launch.py
```

Override the default interface with `network_interface:=eth0`.
