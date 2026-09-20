"""Write a patched copy of an EVT-Bench task config with a different jaw camera.

EVT-Bench's ``run.py`` collects Hydra overrides after the named arguments but
the ``trackvla`` branch calls ``get_config(exp_config)`` without them, so the
only reliable way to change the Spot jaw camera is to write a patched COPY of
``track_infer_{dt,stt,at}.yaml`` and pass its absolute path as ``--exp-config``.
The shared checkout is never modified.

Usage::

    python patch_task_config.py <src.yaml> <dst.yaml> [--hfov 120] [--height 0.7]

``jaw_rgb_sensor`` and ``jaw_panoptic_sensor`` are patched together: the
benchmark's humanoid detector scores visibility on the panoptic mask, so the
two cameras must share one lens.  ``hfov`` is written as an int (habitat types
it as int; a float is rejected by the structured config), ``height`` becomes
``position: [0, height, 0]``.  The output keeps the ``# @package _global_``
Hydra directive as its first line; without it the config composes in the wrong
place.  Python 3.9 compatible; needs only PyYAML.
"""

from __future__ import annotations

import argparse
import sys

import yaml

SENSOR_NAMES = ("jaw_rgb_sensor", "jaw_panoptic_sensor")
PACKAGE_DIRECTIVE = "# @package _global_\n"


def patch_config(cfg: dict, hfov: float = None, height: float = None) -> dict:
    """Patch the agent_1 jaw sensors in a loaded task config in place and return it."""
    updates = {}
    if hfov is not None:
        updates["hfov"] = int(round(float(hfov)))
    if height is not None:
        updates["position"] = [0.0, float(height), 0.0]
    if not updates:
        return cfg
    sim = cfg.setdefault("habitat", {}).setdefault("simulator", {})
    agent_1 = sim.setdefault("agents", {}).setdefault("agent_1", {})
    sensors = agent_1.setdefault("sim_sensors", {})
    for name in SENSOR_NAMES:
        sensors.setdefault(name, {}).update(updates)
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", help="task config to read, e.g. .../track_infer_dt.yaml")
    parser.add_argument("dst", help="patched copy to write (pass its absolute path to run.py)")
    parser.add_argument("--hfov", type=float, default=None, help="jaw camera horizontal FOV, deg")
    parser.add_argument("--height", type=float, default=None, help="jaw camera mount height, m")
    args = parser.parse_args(argv)

    with open(args.src) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        print(f"[patch_task_config] ERROR: {args.src} did not parse to a mapping", file=sys.stderr)
        return 1

    patch_config(cfg, hfov=args.hfov, height=args.height)

    with open(args.dst, "w") as f:
        f.write(PACKAGE_DIRECTIVE)
        yaml.safe_dump(cfg, f, sort_keys=False)

    changes = []
    if args.hfov is not None:
        changes.append(f"hfov={int(round(args.hfov))}")
    if args.height is not None:
        changes.append(f"position=[0, {float(args.height)}, 0]")
    what = ", ".join(changes) if changes else "no changes (plain copy)"
    print(f"[patch_task_config] {args.src} -> {args.dst}: {'/'.join(SENSOR_NAMES)} {what}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
