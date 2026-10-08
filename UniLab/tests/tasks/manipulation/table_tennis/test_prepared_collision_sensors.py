"""Solved robot contact selectors, without altering any collision geometry."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjbatch")

from unisim import MuJoCoBackend
from unisim.scene import SceneCfg

from unilab.tasks.manipulation.table_tennis.prepared_scene import (
    ARM_CONTACT_NAMES,
    prepared_scene_path,
)

JOINT_NAMES = ("rail_joint", "joint1", "joint2", "joint3", "joint4", "joint6")
# A valid-range folded pose has real carriage/link3 and table/link3 contacts.
# It is a detection fixture, not an admissible control trajectory.
COLLIDING_POSE = np.array(
    [
        0.46689334653699116,
        -0.8523592557223691,
        3.181171663646385,
        0.8152712930419048,
        0.1552098497292203,
        -1.4465958356656583,
    ]
)


def _model():
    return mujoco.MjModel.from_xml_path(prepared_scene_path())


def _sensor_counts(model, data):
    return np.array([data.sensordata[int(model.sensor(name).adr[0])] for name in ARM_CONTACT_NAMES])


def _native_counts(model, data):
    robot = model.body("rail_carriage").id
    table = model.body("table").id
    floor = model.geom("floor").id

    def belongs(body, root):
        while body:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return False

    counts = np.zeros(3)
    for contact in data.contact:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        robot1, robot2 = belongs(body1, robot), belongs(body2, robot)
        counts[0] += robot1 and robot2
        counts[1] += (robot1 and belongs(body2, table)) or (robot2 and belongs(body1, table))
        counts[2] += (robot1 and geom2 == floor) or (robot2 and geom1 == floor)
    return counts


def _set_pose(model, data, values):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    indices = np.array([int(model.joint(name).qposadr[0]) for name in JOINT_NAMES])
    data.qpos[indices] = values
    mujoco.mj_forward(model, data)


def test_arm_sensor_selectors_detect_real_self_and_full_table_contacts():
    model = _model()
    data = mujoco.MjData(model)
    _set_pose(model, data, COLLIDING_POSE)
    expected = _native_counts(model, data)
    np.testing.assert_array_equal(expected, [1, 1, 0])
    np.testing.assert_array_equal(_sensor_counts(model, data), expected)
    # The additional world/support contacts in this pose are not falsely
    # classified as a floor contact or silently relabelled as self contact.
    assert data.ncon > expected.sum()
    for name in ARM_CONTACT_NAMES:
        sensor = model.sensor(name).id
        assert model.sensor_dim[sensor] == 1
        assert model.sensor_needstage[sensor] == mujoco.mjtStage.mjSTAGE_ACC


@pytest.mark.parametrize("side", [-1, 1])
def test_home_and_height_preparation_baseline_have_no_robot_contacts(side):
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(_sensor_counts(model, data), 0)
    indices = np.array([int(model.joint(name).qposadr[0]) for name in JOINT_NAMES])
    home = data.qpos[indices].copy()
    for phase in (-0.1, 0, 0.3, 0.8):
        pose = home.copy()
        pose[1] += side * np.pi / 2
        pose[2:5] += phase * np.array([0.5, 1, -0.5])
        _set_pose(model, data, pose)
        np.testing.assert_array_equal(_sensor_counts(model, data), 0)
        np.testing.assert_array_equal(_native_counts(model, data), 0)


def test_bound_public_sensors_preserve_every_solved_physics_sample():
    model = _model()
    oracle = mujoco.MjData(model)
    _set_pose(model, oracle, COLLIDING_POSE)
    backend = MuJoCoBackend(
        SceneCfg(model_file=prepared_scene_path()),
        num_envs=1,
        sim_dt=model.opt.timestep,
        np_dtype=np.float64,
        base_name="rail_carriage",
    )
    backend.materialize()
    try:
        backend.set_state(np.array([0]), oracle.qpos[None], oracle.qvel[None])
        view = backend.bind_sensor_data(ARM_CONTACT_NAMES)
        samples = []
        backend.set_post_step_observer(lambda sim: samples.append(view.read().copy()))
        backend.step(np.zeros((1, model.nu)), nsteps=4)
        expected = []
        oracle.ctrl[:] = 0
        for _ in range(4):
            mujoco.mj_step(model, oracle)
            expected.append(_sensor_counts(model, oracle))
        assert len(samples) == 4
        np.testing.assert_array_equal(np.asarray(samples)[:, 0], expected)
        assert np.asarray(samples)[0, 0, 0] > 0
        assert np.asarray(samples)[0, 0, 1] > 0
    finally:
        backend.close()


def test_sensor_addition_leaves_dynamics_and_collision_model_unchanged():
    root = ET.parse(prepared_scene_path()).getroot()
    sensors = root.find("sensor")
    assert sensors is not None
    for sensor in list(sensors):
        if sensor.get("name") in ARM_CONTACT_NAMES:
            sensors.remove(sensor)
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    instrumented = _model()
    for field in (
        "body_mass",
        "body_inertia",
        "body_gravcomp",
        "dof_damping",
        "dof_armature",
        "jnt_range",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "geom_contype",
        "geom_conaffinity",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "pair_geom1",
        "pair_geom2",
    ):
        np.testing.assert_array_equal(getattr(original, field), getattr(instrumented, field))
    old_data, new_data = mujoco.MjData(original), mujoco.MjData(instrumented)
    _set_pose(original, old_data, COLLIDING_POSE)
    _set_pose(instrumented, new_data, COLLIDING_POSE)
    for _ in range(4):
        mujoco.mj_step(original, old_data)
        mujoco.mj_step(instrumented, new_data)
        np.testing.assert_array_equal(old_data.qpos, new_data.qpos)
        np.testing.assert_array_equal(old_data.qvel, new_data.qvel)
        assert old_data.ncon == new_data.ncon


@pytest.mark.parametrize("lateral", [-0.35, 0.35])
def test_zero_action_preparation_and_passive_hit_have_no_robot_contacts(lateral):
    pytest.importorskip("rail_alignment")
    from unilab.envs.manager_based_rl_env import make_manager_based_rl_env
    from unilab.tasks.manipulation.table_tennis.ball_launcher import ServeLauncherCfg
    from unilab.tasks.manipulation.table_tennis.training import build_panthera_return_env_cfg

    cfg = build_panthera_return_env_cfg()
    cfg.auto_reset = False
    cfg.events["launch_ball"].params["launcher_cfg"] = ServeLauncherCfg(
        lateral_range=(lateral, lateral),
        speed_range=(2.9, 2.9),
        receiver_bounce_range=(-0.94, -0.94),
        receiver_peak_range=(0.32, 0.32),
    )
    env = make_manager_based_rl_env(cfg, num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        view = env.physics_backend.bind_sensor_data(ARM_CONTACT_NAMES)
        original = control._observe_substep
        maxima = np.zeros(3)

        def observer(backend):
            maxima[:] = np.maximum(maxima, view.read()[0])
            original(backend)

        env.physics_backend.set_post_step_observer(observer)
        for _ in range(env.max_episode_length):
            state = env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            if state.terminated[0] or state.truncated[0]:
                break
        assert control.ready[0] and control.episodes[0].hit
        np.testing.assert_array_equal(maxima, 0)
    finally:
        env.close()
