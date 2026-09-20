import mujoco

from vln_mujoco.model import CAMERA_NAME, build_model


def test_bundled_scene_and_robot_compile() -> None:
    model = build_model()
    assert model.nu == 0  # pure kinematics: no actuators by design
    assert model.camera(CAMERA_NAME).id >= 0
    assert model.body("base_link").id >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
