from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco

from unisim import FakeBackend, MuJoCoBackend
from unisim.backend.base import PreStepControlOutput
from unisim.dr.types import ResetRandomizationPayload
from unisim.scene import SceneCfg

TRACKING_MODEL = """<mujoco>
  <compiler angle='radian'/>
  <option timestep='0.001'/>
  <worldbody><body name='base'><joint name='rail' type='slide' axis='1 0 0'
      limited='true' range='-1 1'/>
    <geom type='sphere' size='0.05' mass='1' contype='0' conaffinity='0'/>
    <body name='arm' pos='0 0 0.4'><joint name='hinge' axis='0 1 0'
        limited='true' range='-2 2' damping='0.4'/>
      <geom type='box' pos='0 0 0.2' size='0.05 0.05 0.2' mass='2'
        contype='0' conaffinity='0'/></body></body></worldbody>
  <actuator><motor name='rail_motor' joint='rail' ctrlrange='-100 100'
      forcerange='-80 80'/>
    <position name='hinge_position' joint='hinge' kp='100' kv='10'
      ctrlrange='-3 3' forcerange='-100 100'/></actuator>
</mujoco>"""


def _backend(tmp_path: Path, xml: str = TRACKING_MODEL, num_envs: int = 2):
    path = tmp_path / "model.xml"
    path.write_text(xml)
    backend = MuJoCoBackend(SceneCfg(model_file=str(path)), num_envs=num_envs, sim_dt=0.001)
    backend.materialize()
    return backend


def test_tracking_compensates_full_coupled_dynamics(tmp_path: Path):
    backend = _backend(tmp_path)
    qpos = np.array([[0.2, 0.5], [-0.2, -0.4]])
    qvel = np.array([[0.1, -0.3], [0.2, 0.4]])
    desired_acceleration = np.array([[0.2, -0.5], [-0.3, 0.7]])
    backend.set_state(np.arange(2), qpos, qvel)
    # Use the actual stored (default float32) state for both servo targets and
    # the independent double-precision forward-dynamics oracle.
    actual = backend.get_state(("qpos", "qvel"))
    qpos, qvel = actual["qpos"], actual["qvel"]
    view = backend.bind_tracking_control(["rail", "hinge"], ["rail_motor", "hinge_position"])
    assert view.kp == (0.0, 100.0)
    assert view.kd == (0.0, 10.0)
    controls = view.compute(qpos, qvel, desired_acceleration)
    model = mujoco.MjModel.from_xml_string(TRACKING_MODEL)
    scratch = mujoco.MjData(model)
    for env in range(2):
        scratch.qpos[:] = qpos[env]
        scratch.qvel[:] = qvel[env]
        scratch.ctrl[:] = controls[env]
        mujoco.mj_forward(model, scratch)
        np.testing.assert_allclose(scratch.qacc, desired_acceleration[env], atol=1e-10)
    np.testing.assert_array_equal(backend.get_state(("qpos",))["qpos"], qpos)


def test_tracking_servo_velocity_feedforward_and_limits(tmp_path: Path):
    backend = _backend(tmp_path)
    view = backend.bind_tracking_control(["rail", "hinge"], ["rail_motor", "hinge_position"])
    zeros = np.zeros((2, 2))
    desired_velocity = zeros.copy()
    desired_velocity[:, 1] = 0.5
    controls = view.compute(zeros, desired_velocity, zeros)
    np.testing.assert_allclose(controls[:, 1], 0.05, atol=1e-10)
    saturated = view.compute(zeros, zeros, np.full((2, 2), 1e5))
    np.testing.assert_allclose(saturated[:, 0], 80)
    np.testing.assert_allclose(saturated[:, 1], 1)
    with pytest.raises(ValueError, match="shape"):
        view.compute(zeros[:1], zeros, zeros)
    with pytest.raises(ValueError, match="NaN"):
        view.compute(np.full((2, 2), np.nan), zeros, zeros)
    with pytest.raises(ValueError, match="joint limits"):
        view.compute(np.full((2, 2), 3), zeros, zeros)
    with pytest.raises(NotImplementedError, match="transmission"):
        backend.bind_tracking_control(["hinge", "rail"], ["rail_motor", "hinge_position"])
    with pytest.raises(ValueError, match="unique"):
        backend.bind_tracking_control(["rail", "rail"], ["rail_motor", "hinge_position"])


@pytest.mark.parametrize("position_error", [-0.03, 0.01, 0.03])
def test_motor_acceleration_cancels_coupled_servo_feedback(tmp_path: Path, position_error: float):
    backend = _backend(tmp_path)
    backend.set_state(
        np.arange(2),
        np.array([[0.2, 0.5], [-0.2, -0.4]]),
        np.array([[0.1, -0.3], [0.2, 0.4]]),
    )
    actual = backend.get_state(("qpos", "qvel"))
    reference = actual["qpos"].astype(np.float64)
    reference[:, 1] += position_error
    reference_velocity = actual["qvel"].astype(np.float64)
    reference_velocity[:, 1] += 0.05
    acceleration = np.array([[0.2, -0.5], [-0.3, 0.7]])
    view = backend.bind_tracking_control(["rail", "hinge"], ["rail_motor", "hinge_position"])
    controls = view.compute(reference, reference_velocity, acceleration)
    model = mujoco.MjModel.from_xml_string(TRACKING_MODEL)
    scratch = mujoco.MjData(model)
    for env in range(2):
        scratch.qpos[:] = actual["qpos"][env]
        scratch.qvel[:] = actual["qvel"][env]
        scratch.ctrl[:] = controls[env]
        mujoco.mj_forward(model, scratch)
        np.testing.assert_allclose(scratch.qacc[0], acceleration[env, 0], atol=1e-10)
        # Servo feedback remains active; it is not cancelled with the rail residual.
        assert abs(scratch.qacc[1] - acceleration[env, 1]) > 0.1
        # Independently solve the old physical 0 N / 1 N rail response.
        scratch.ctrl[0] = 0
        mujoco.mj_forward(model, scratch)
        zero_acceleration = float(scratch.qacc[0])
        scratch.ctrl[0] = 1
        mujoco.mj_forward(model, scratch)
        response = float(scratch.qacc[0]) - zero_acceleration
        required = (acceleration[env, 0] - zero_acceleration) / response
        np.testing.assert_allclose(controls[env, 0], required, atol=1e-10)
    saturated = view.compute(reference, reference_velocity, np.full((2, 2), 1e5))
    np.testing.assert_allclose(saturated[:, 0], 80)
    scratch.ctrl[:] = saturated[0]
    mujoco.mj_forward(model, scratch)
    assert abs(scratch.qacc[0] - 1e5) > 1e4


def test_multiple_motors_cancel_servo_residual_together(tmp_path: Path):
    xml = (
        TRACKING_MODEL.replace(
            "<geom type='box' pos='0 0 0.2'",
            "<joint name='yaw' axis='0 0 1' limited='true' range='-2 2'/>"
            "<geom type='box' pos='0.1 0.1 0.2'",
        )
        .replace(
            "<position name='hinge_position' joint='hinge' kp='100' kv='10'",
            "<motor name='hinge_motor' joint='hinge'",
        )
        .replace("ctrlrange='-3 3'", "ctrlrange='-100 100'")
        .replace(
            "</actuator>",
            "<position name='yaw_position' joint='yaw' kp='100' kv='10' "
            "ctrlrange='-3 3' forcerange='-100 100'/></actuator>",
        )
    )
    backend = _backend(tmp_path, xml)
    backend.set_state(
        np.arange(2),
        np.array([[0.2, 0.5, 0.3], [-0.2, -0.4, -0.2]]),
        np.array([[0.1, -0.3, 0.1], [0.2, 0.4, -0.2]]),
    )
    actual = backend.get_state(("qpos", "qvel"))
    reference = actual["qpos"].astype(np.float64)
    reference[:, 2] += 0.01
    acceleration = np.array([[0.2, -0.5, 0.1], [-0.3, 0.7, 0.2]])
    view = backend.bind_tracking_control(
        ["rail", "hinge", "yaw"], ["rail_motor", "hinge_motor", "yaw_position"]
    )
    controls = view.compute(reference, actual["qvel"], acceleration)
    model = mujoco.MjModel.from_xml_string(xml)
    scratch = mujoco.MjData(model)
    for env in range(2):
        scratch.qpos[:] = actual["qpos"][env]
        scratch.qvel[:] = actual["qvel"][env]
        scratch.ctrl[:] = controls[env]
        mujoco.mj_forward(model, scratch)
        np.testing.assert_allclose(scratch.qacc[:2], acceleration[env, :2], atol=1e-10)
        assert abs(scratch.qacc[2] - acceleration[env, 2]) > 0.1


def test_motor_residual_correction_retains_joint_force_limit(tmp_path: Path):
    xml = TRACKING_MODEL.replace(
        "<joint name='rail'", "<joint actuatorfrcrange='-25 25' name='rail'"
    )
    backend = _backend(tmp_path, xml)
    view = backend.bind_tracking_control(["rail", "hinge"], ["rail_motor", "hinge_position"])
    zeros = np.zeros((2, 2))
    controls = view.compute(zeros, zeros, np.full((2, 2), 1e5))
    np.testing.assert_allclose(controls[:, 0], 25)
    model = mujoco.MjModel.from_xml_string(xml)
    scratch = mujoco.MjData(model)
    scratch.ctrl[:] = controls[0]
    mujoco.mj_forward(model, scratch)
    np.testing.assert_allclose(scratch.qfrc_actuator[0], 25)


def test_motor_only_binding_accounts_for_unbound_online_servo(tmp_path: Path):
    backend = _backend(tmp_path)
    backend.set_state(np.arange(2), np.array([[0.2, 0.5], [-0.2, -0.4]]), np.zeros((2, 2)))
    backend.step(np.array([[0, 0.51], [0, -0.39]]))
    actual = backend.get_state(("qpos", "qvel", "ctrl"))
    desired_acceleration = np.array([[0.2], [-0.3]])
    view = backend.bind_tracking_control(["rail"], ["rail_motor"])
    controls = view.compute(actual["qpos"][:, :1], actual["qvel"][:, :1], desired_acceleration)
    assert np.all(np.abs(controls) < 80)
    model = mujoco.MjModel.from_xml_string(TRACKING_MODEL)
    scratch = mujoco.MjData(model)
    for env in range(2):
        scratch.qpos[:] = actual["qpos"][env]
        scratch.qvel[:] = actual["qvel"][env]
        scratch.ctrl[:] = actual["ctrl"][env]
        scratch.ctrl[0] = controls[env, 0]
        mujoco.mj_forward(model, scratch)
        np.testing.assert_allclose(scratch.qacc[0], desired_acceleration[env, 0], atol=1e-10)


def test_cold_fk_uses_defaults_and_preserves_online_state(tmp_path: Path):
    backend = _backend(tmp_path)
    backend.set_state(np.arange(2), np.array([[0.7, 1], [-0.4, -1]]), np.zeros((2, 2)))
    before = backend.get_state(("qpos", "qvel"))
    poses, quaternions = backend.evaluate_body_poses(
        ["rail", "hinge"], np.array([[0, 0], [0.3, np.pi / 2]]), ["base", "arm"]
    )
    np.testing.assert_allclose(poses[:, 0], [[0, 0, 0], [0.3, 0, 0]], atol=1e-10)
    np.testing.assert_allclose(poses[:, 1], [[0, 0, 0.4], [0.3, 0, 0.4]], atol=1e-10)
    np.testing.assert_allclose(quaternions[1, 1], [np.sqrt(0.5), 0, np.sqrt(0.5), 0])
    for name, value in before.items():
        np.testing.assert_array_equal(backend.get_state((name,))[name], value)
    with pytest.raises(ValueError, match="unknown body"):
        backend.evaluate_body_poses(["rail"], np.zeros((1, 1)), ["missing"])


def test_tracking_rejects_randomized_online_model(tmp_path: Path):
    backend = _backend(tmp_path)
    view = backend.bind_tracking_control(["rail", "hinge"], ["rail_motor", "hinge_position"])
    zeros = np.zeros((2, 2))
    backend.set_state(
        np.arange(2),
        zeros,
        zeros,
        randomization=ResetRandomizationPayload(gravity=np.tile([0, 0, -5], (2, 1))),
    )
    with pytest.raises(NotImplementedError, match="unrandomized"):
        view.compute(zeros, zeros, zeros)
    with pytest.raises(NotImplementedError, match="unrandomized"):
        backend.bind_tracking_control(["rail"], ["rail_motor"])


@pytest.mark.parametrize("integrator", ["Euler", "implicitfast", "RK4"])
def test_post_observer_catches_transient_solved_contact_and_final_substep(
    tmp_path: Path, integrator: str
):
    xml = f"""<mujoco><option timestep='0.001' gravity='0 0 0' integrator='{integrator}'/>
      <worldbody><geom name='floor' type='plane' size='1 1 0.1' solref='-500000 -10'/>
        <body name='base' pos='0 0 0.15'><joint name='vertical' type='slide' axis='0 0 1'/>
          <geom name='ball' type='sphere' size='0.02' mass='1' solref='-500000 -10'/>
        </body></worldbody>
      <sensor><contact name='touch' geom1='ball' geom2='floor' data='found' num='1'/></sensor>
    </mujoco>"""
    backend = _backend(tmp_path, xml, num_envs=1)
    backend.set_state(np.array([0]), np.zeros((1, 1)), np.full((1, 1), -2))
    contact = backend.bind_sensor_data(["touch"])
    seen = []
    before = []

    def pre(sim, ctrl):
        before.append(float(sim.get_physics_state()[0, 0]))
        return ctrl

    def post(sim):
        seen.append((float(sim.get_physics_state()[0, 0]), contact.read().copy()))

    backend.set_pre_step_control(pre)
    backend.set_post_step_observer(post)
    backend.step(np.zeros((1, 0)), nsteps=150)
    assert len(seen) == len(before) == 150
    np.testing.assert_allclose([item[0] for item in seen], np.arange(1, 151) * 0.001)
    np.testing.assert_allclose(before, np.arange(150) * 0.001)
    assert any(item[1][0, 0] > 0 for item in seen)
    assert seen[-1][1][0, 0] == 0
    backend.set_post_step_observer(None)
    backend.step(np.zeros((1, 0)), nsteps=3)
    assert len(seen) == 150


@pytest.mark.parametrize("integrator", ["Euler", "implicitfast", "RK4"])
def test_pre_and_post_observers_preserve_kinematics_and_wrenches(tmp_path: Path, integrator: str):
    xml = f"""<mujoco><option timestep='0.001' gravity='0 0 0' integrator='{integrator}'/>
      <worldbody><body name='base'><joint name='rail' type='slide' axis='1 0 0'/>
        <geom type='sphere' size='0.05' mass='1' contype='0' conaffinity='0'/>
      </body></worldbody></mujoco>"""
    path = tmp_path / "model.xml"
    path.write_text(xml)
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(path)),
        num_envs=1,
        sim_dt=0.001,
        base_name="base",
        tracked_body_names=["base"],
        add_body_sensors=True,
        np_dtype=np.float64,
    )
    backend.materialize()
    body = backend.get_body_ids(["base"])
    seen = []

    def pre(sim, ctrl):
        # The current integrated qpos and body getters must agree even though
        # the scene integrator cannot use native Euler sensor-copyout hooks.
        np.testing.assert_allclose(
            sim.get_body_pos_w(body)[0, 0, 0],
            sim.get_state(("qpos",))["qpos"][0, 0],
            atol=1e-12,
        )
        return PreStepControlOutput(ctrl=ctrl, body_ids=body, force=np.array([[[2.0, 0, 0]]]))

    backend.set_pre_step_control(pre)
    backend.set_post_step_observer(lambda sim: seen.append(sim.get_dof_vel().copy()))
    backend.apply_body_force(body, np.array([[[3.0, 0, 0]]]))
    backend.step(np.zeros((1, 0)), nsteps=5)
    np.testing.assert_allclose([float(v[0, 0]) for v in seen], np.arange(1, 6) * 0.005)
    backend.set_pre_step_control(None)
    backend.step(np.zeros((1, 0)), nsteps=1)
    np.testing.assert_allclose(backend.get_dof_vel(), [[0.025]], atol=1e-12)


def test_observer_is_fail_closed_and_not_called_by_reset(tmp_path: Path):
    with pytest.raises(NotImplementedError):
        FakeBackend().set_post_step_observer(lambda backend: None)
    backend = _backend(tmp_path)
    seen = []
    backend.set_post_step_observer(lambda sim: seen.append(sim.get_physics_state().copy()))
    backend.reset(np.array([0]))
    assert not seen
    backend.step(np.zeros((2, 2)), nsteps=2)
    assert len(seen) == 2
    backend.set_post_step_observer(lambda sim: sim.step(np.zeros((2, 2))))
    with pytest.raises(RuntimeError, match="recursively"):
        backend.step(np.zeros((2, 2)))
