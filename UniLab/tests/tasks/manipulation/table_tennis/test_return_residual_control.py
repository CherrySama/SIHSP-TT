"""Independent checks of residual reference bounds and physical flight masking."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rail_alignment")
pytest.importorskip("mujoco")

from unilab.envs.manager_based_rl_env import make_manager_based_rl_env
from unilab.tasks.manipulation.table_tennis.ball_launcher import ServeLauncherCfg
from unilab.tasks.manipulation.table_tennis.training import build_panthera_return_env_cfg


@pytest.fixture(params=[-0.35, 0.35])
def prepared_env(request):
    cfg = build_panthera_return_env_cfg()
    cfg.auto_reset = False
    cfg.events["launch_ball"].params["launcher_cfg"] = ServeLauncherCfg(
        lateral_range=(request.param, request.param),
        speed_range=(2.9, 2.9),
        receiver_bounce_range=(-0.94, -0.94),
        receiver_peak_range=(0.32, 0.32),
    )
    env = make_manager_based_rl_env(cfg, num_envs=2)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        assert control.action_dim == 6
        for _ in range(90):
            env.step(np.zeros((2, 6)))
        assert control.ready.all()
        assert np.all(control.side == np.sign(request.param))
        assert not any(episode.hit for episode in control.episodes)
        yield env, control
    finally:
        env.close()


@pytest.mark.parametrize("command", [-1, 1])
def test_residual_limiter_bounds_and_repeated_absolute_targets(prepared_env, command):
    env, control = prepared_env
    joints = np.array([0, 2, 3, 4])
    base = control._preparation_base[0].copy()
    scales = control._offset_scales
    lower = np.maximum(control._limits[joints, 0], base[joints] - scales)
    upper = np.minimum(control._limits[joints, 1], base[joints] + scales)
    speed = np.array([0.2, 0.5, 0.5, 0.5])
    acceleration = np.array([2.0, 4.0, 4.0, 4.0])
    dt = env.physics_dt
    control._command[0, 2:] = command
    # The real physics establishes the base. Exercise the reference limiter at
    # its actual substep interval beyond the single incoming ball's duration.
    for _ in range(6000):
        previous_position = control._reference[0, joints].copy()
        previous_velocity = control._reference_velocity[0, joints].copy()
        control._advance_adjustments(0, dt, flight=False)
        control.elapsed[0] += dt
        position = control._reference[0, joints]
        velocity = control._reference_velocity[0, joints]
        assert np.all(position >= lower) and np.all(position <= upper)
        assert np.all(np.abs(velocity) <= speed)
        assert np.all(np.abs(velocity - previous_velocity) <= acceleration * dt + 1e-15)
        np.testing.assert_allclose(position - previous_position, velocity * dt, atol=1e-16)
        np.testing.assert_allclose(
            control._reference_acceleration[0, joints],
            (velocity - previous_velocity) / dt,
            atol=1e-15,
        )
    target = np.clip(base[joints] + scales * command, lower, upper)
    np.testing.assert_allclose(control._reference[0, joints], target, atol=2e-8)
    np.testing.assert_array_equal(control._preparation_base[0], base)
    assert np.all(control._peak_offset[0] <= scales + 1e-15)
    # Zero is a return to the original base, not a hold at the accumulated target.
    control._command[0, 2:] = 0
    for _ in range(6000):
        control._advance_adjustments(0, dt, flight=False)
    np.testing.assert_allclose(control._reference[0, joints], base[joints], atol=2e-8)


def test_real_outgoing_masks_opposite_actions_and_brakes_continuously(prepared_env):
    env, control = prepared_env
    command = np.tile([0.1, 0, 0.2, 0.2, 0.2, 0.2], (2, 1))
    for _ in range(50):
        env.step(command)
        if control.episodes[0].outgoing_time is not None:
            break
    assert all(episode.hit and episode.outgoing_time is not None for episode in control.episodes)
    assert not any(episode.done for episode in control.episodes)
    assert np.abs(control._reference_velocity[:, 1]).min() > 0.1
    initial_velocity = np.abs(control._reference_velocity.copy())
    records = []
    original = control._observe_substep

    def record(backend):
        original(backend)
        records.append((control._reference_velocity.copy(), control._reference_acceleration.copy()))

    env.physics_backend.set_post_step_observer(record)
    previous_velocity = control._reference_velocity.copy()
    for _ in range(5):
        env.step(np.array([[1] * 6, [-1] * 6], dtype=float))
        np.testing.assert_array_equal(control._executed_command, 0)
        np.testing.assert_array_equal(control._reference[0], control._reference[1])
        state = env.physics_backend.get_state(("qpos", "qvel"))
        np.testing.assert_array_equal(state["qpos"][0], state["qpos"][1])
        np.testing.assert_array_equal(state["qvel"][0], state["qvel"][1])
    bounds = np.array([2, 12, 4, 4, 4, 20])
    for velocity, acceleration in records:
        assert np.all(np.abs(velocity) <= initial_velocity + 1e-15)
        assert np.all(np.abs(velocity - previous_velocity) <= bounds * env.physics_dt + 1e-15)
        # Rail includes bounded feedback in addition to its trajectory acceleration.
        assert np.all(np.abs(acceleration[:, 0]) <= 10)
        assert np.all(np.abs(acceleration[:, 1:]) <= bounds[1:] + 1e-12)
        previous_velocity = velocity
    np.testing.assert_array_equal(control._reference_velocity, 0)
    assert not any(episode.done for episode in control.episodes)


def test_partial_reset_preserves_other_residual_reference_and_clears_selected(prepared_env):
    env, control = prepared_env
    for _ in range(5):
        env.step(np.tile([0.1, 0, 0.2, -0.2, 0.2, -0.2], (2, 1)))
    names = (
        "_reference",
        "_preparation_base",
        "_reference_velocity",
        "_reference_acceleration",
        "_command",
        "_executed_command",
        "_peak_offset",
        "_peak_arm_contacts",
        "_previous_action",
    )
    unreset = {name: getattr(control, name)[1].copy() for name in names}
    env.reset(np.array([0], dtype=np.int32))
    for name, expected in unreset.items():
        np.testing.assert_array_equal(getattr(control, name)[1], expected)
    np.testing.assert_array_equal(control._reference[0], control.home)
    np.testing.assert_array_equal(control._preparation_base[0], control.home)
    for name in names[2:]:
        np.testing.assert_array_equal(getattr(control, name)[0], 0)
    assert not control.ready[0] and control.ready[1]
    assert control.elapsed[0] == 0 and control.elapsed[1] > 0.9
    assert np.isnan(control.prepared_time[0])
    env.step(np.array([[1] * 6, [0] * 6], dtype=float))
    np.testing.assert_array_equal(control._executed_command[0], 0)


@pytest.mark.parametrize("lateral", [-0.35, 0.35])
def test_six_action_zero_reproduces_two_action_physical_hold(lateral):
    envs = []
    try:
        for residual in (False, True):
            cfg = build_panthera_return_env_cfg()
            cfg.auto_reset = False
            cfg.actions["return_control"].residual_control = residual
            cfg.events["launch_ball"].params["launcher_cfg"] = ServeLauncherCfg(
                lateral_range=(lateral, lateral),
                speed_range=(2.9, 2.9),
                receiver_bounce_range=(-0.94, -0.94),
                receiver_peak_range=(0.32, 0.32),
            )
            env = make_manager_based_rl_env(cfg, num_envs=1)
            envs.append(env)
            env.init_state()
        for _ in range(130):
            for env in envs:
                control = env.action_manager.get_term("return_control")
                env.step(np.zeros((1, control.action_dim)))
            states = [env.physics_backend.get_state(("qpos", "qvel")) for env in envs]
            for key in ("qpos", "qvel"):
                np.testing.assert_allclose(states[0][key], states[1][key], atol=2e-9, rtol=0)
        controls = [env.action_manager.get_term("return_control") for env in envs]
        assert all(control.episodes[0].outgoing_time is not None for control in controls)
        assert controls[0].prepared_time[0] == controls[1].prepared_time[0]
        assert controls[0].episodes[0].hit_time == controls[1].episodes[0].hit_time
        assert controls[0].episodes[0].outgoing_time == controls[1].episodes[0].outgoing_time
    finally:
        for env in envs:
            env.close()
