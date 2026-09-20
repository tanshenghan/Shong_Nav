from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

SCENE_NAME = "val_2"
SCENE_ID = "procthor-val-2"
SCENE_DATASET = "MolmoSpaces ProcTHOR 10K val"
SPAWN = (6.5, 13.8, 0.0)
CAMERA_NAME = "robot_rgb"
THIRD_PERSON_CAMERA_NAME = "robot_third_person"
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 270
PHYSICS_DT = 0.005
WHEEL_RADIUS = 0.033
WHEEL_TRACK = 0.160


def scene_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "assets"
        / "scenes"
        / "procthor-10k-val"
        / "val_2_ceiling.xml"
    )


def _freeze_environment(root: ET.Element) -> None:
    for parent in root.iter():
        for child in tuple(parent):
            if child.tag in {"joint", "freejoint"}:
                parent.remove(child)


def _absolute_asset_paths(root: ET.Element, source: Path) -> None:
    asset_root = source.parents[2].resolve()
    for element in root.iter():
        filename = element.get("file")
        if not filename:
            continue
        resolved = (source.parent / filename).resolve()
        try:
            resolved.relative_to(asset_root)
        except ValueError as exc:
            raise RuntimeError(f"scene asset escapes the bundled asset root: {filename}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"bundled scene asset is missing: {resolved}")
        element.set("file", str(resolved))


def _robot_body() -> ET.Element:
    body = ET.fromstring(
        f"""
        <body name="base_link" pos="{SPAWN[0]} {SPAWN[1]} {WHEEL_RADIUS}">
          <freejoint name="base_joint" />
          <inertial pos="0 0 0.055" mass="2.5" diaginertia="0.018 0.018 0.025" />
          <geom name="base_collision" type="cylinder" size="0.092 0.028" pos="0 0 0.030"
                rgba="0.08 0.11 0.13 1" friction="0.9 0.02 0.002" />
          <geom name="lower_plate" type="cylinder" size="0.088 0.010" pos="0 0 0.063"
                rgba="0.05 0.08 0.09 1" contype="0" conaffinity="0" />
          <geom name="upper_plate" type="cylinder" size="0.078 0.010" pos="0 0 0.145"
                rgba="0.08 0.68 0.77 1" contype="0" conaffinity="0" />
          <geom name="mast" type="cylinder" size="0.010 0.052" pos="-0.020 0 0.105"
                rgba="0.65 0.72 0.74 1" contype="0" conaffinity="0" />
          <geom name="lidar" type="cylinder" size="0.042 0.018" pos="0.015 0 0.178"
                rgba="0.05 0.08 0.09 1" contype="0" conaffinity="0" />
          <geom name="camera_case" type="box" size="0.022 0.030 0.018" pos="0.067 0 0.162"
                rgba="0.14 0.18 0.19 1" contype="0" conaffinity="0" />
          <camera name="{CAMERA_NAME}" mode="fixed" pos="0.090 0 0.165"
                  xyaxes="0 -1 0 0 0 1" fovy="79.865" />
          <camera name="{THIRD_PERSON_CAMERA_NAME}" mode="fixed" pos="-0.65 0 0.50"
                  xyaxes="0 -1 0 0.38 0 0.925" fovy="65" />
          <body name="left_wheel" pos="0 0.080 0">
            <joint name="left_wheel_joint" type="hinge" axis="0 1 0" />
            <geom name="left_wheel_geom" type="cylinder" size="{WHEEL_RADIUS} 0.012"
                  euler="{math.pi / 2} 0 0" mass="0.08" rgba="0.035 0.045 0.048 1"
                  friction="2.0 0.01 0.001" solref="0.004 1" />
          </body>
          <body name="right_wheel" pos="0 -0.080 0">
            <joint name="right_wheel_joint" type="hinge" axis="0 1 0" />
            <geom name="right_wheel_geom" type="cylinder" size="{WHEEL_RADIUS} 0.012"
                  euler="{math.pi / 2} 0 0" mass="0.08" rgba="0.035 0.045 0.048 1"
                  friction="2.0 0.01 0.001" solref="0.004 1" />
          </body>
          <body name="caster" pos="-0.073 0 -0.018">
            <geom name="caster_geom" type="sphere" size="0.015" mass="0.025"
                  rgba="0.32 0.36 0.37 1" friction="0.05 0.001 0.0001" />
          </body>
        </body>
        """
    )
    return body


def build_model(source: Path | None = None) -> mujoco.MjModel:
    source = (source or scene_path()).resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"bundled MolmoSpaces scene is missing: {source}. "
            "Run scripts/vendor_molmospaces_scene.py during development."
        )
    root = ET.parse(source).getroot()
    _freeze_environment(root)
    _absolute_asset_paths(root, source)

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", str(PHYSICS_DT))

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(CAMERA_WIDTH))
    global_visual.set("offheight", str(CAMERA_HEIGHT))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MolmoSpaces scene has no worldbody")
    worldbody.append(_robot_body())

    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
