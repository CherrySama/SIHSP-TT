"""Physical preparation handoff and one-shot event rewards in the formal task."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rail_alignment")
pytest.importorskip("mujoco")

from unilab.envs.manager_based_rl_env import make_manager_based_rl_env
from unilab.tasks.manipulation.table_tennis.ball_launcher import ServeLauncherCfg
from unilab.tasks.manipulation.table_tennis.return_control import (
    return_guidance_reward,
    return_reward,
    return_shaping_reward,
)
from unilab.tasks.manipulation.table_tennis.return_events import ReturnEpisode
from unilab.tasks.manipulation.table_tennis.training import build_panthera_return_env_cfg


def _cfg():
    cfg = build_panthera_return_env_cfg()
    cfg.auto_reset = False
    cfg.events["launch_ball"].params["launcher_cfg"] = ServeLauncherCfg(
        lateral_range=(-0.35, -0.35),
        speed_range=(2.9, 2.9),
        receiver_bounce_range=(-0.94, -0.94),
        receiver_peak_range=(0.32, 0.32),
    )
    return cfg


def test_return_judgement_geometry_matches_materialized_physics_scene():
    import mujoco

    cfg = _cfg()
    model = mujoco.MjModel.from_xml_path(cfg.scene.model_file)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    episode = ReturnEpisode()
    table = model.geom("table_top").id
    net = model.geom("net_mesh").id
    ball = model.geom("ball_geom").id
    assert episode.table_half_length == pytest.approx(model.geom_size[table, 0])
    assert episode.table_half_width == pytest.approx(model.geom_size[table, 1])
    assert episode.table_z == pytest.approx(data.geom_xpos[table, 2] + model.geom_size[table, 2])
    assert episode.ball_radius == pytest.approx(model.geom_size[ball, 0])
    assert episode.net_top_z == pytest.approx(data.geom_xpos[net, 2] + model.geom_size[net, 2])
    assert episode.net_half_width == pytest.approx(model.geom_size[net, 1])
    assert episode.net_half_thickness == pytest.approx(model.geom_size[net, 0])
    shaping_cfg = cfg.rewards["outgoing_direction"].params["shaping_cfg"]
    assert shaping_cfg.gravity == pytest.approx(-model.opt.gravity[2])
    guidance_cfg = cfg.actions["return_control"].guidance
    paddle = model.geom("paddle_contact").id
    assert guidance_cfg.gravity == pytest.approx(-model.opt.gravity[2])
    assert guidance_cfg.ball_radius == pytest.approx(model.geom_size[ball, 0])
    assert guidance_cfg.paddle_radius == pytest.approx(model.geom_size[paddle, 0])
    assert guidance_cfg.paddle_half_thickness == pytest.approx(model.geom_size[paddle, 1])


def test_guidance_is_live_only_before_hit_and_expires_when_forecast_is_stale():
    env = make_manager_based_rl_env(_cfg(), num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        assert not control.guidance_state()[2].any()
        live_samples = 0
        for _ in range(150):
            env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            scores, _, valid, _ = control.guidance_state()
            if valid[0]:
                live_samples += 1
                np.testing.assert_allclose(
                    return_guidance_reward(env, "reach"),
                    scores[:, 0] * env.step_dt,
                    rtol=1e-6,
                    atol=1e-9,
                )
                saved_timestamp = control._guidance_timestamp[0]
                control._guidance_timestamp[0] = control.elapsed[0] - 0.1
                assert not control.guidance_state()[2].any()
                assert not return_guidance_reward(env, "reach").any()
                control._guidance_timestamp[0] = saved_timestamp
            if control.episodes[0].hit:
                assert not control.guidance_state()[2].any()
                assert not return_guidance_reward(env, "reach").any()
                assert not return_guidance_reward(env, "strike").any()
                break
        assert live_samples > 0
        assert control.episodes[0].hit
    finally:
        env.close()


def test_policy_is_masked_during_physical_preparation_and_partial_reset_clears_state():
    env = make_manager_based_rl_env(_cfg(), num_envs=2)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        actions = np.zeros((2, control.action_dim), dtype=np.float32)
        actions[0] = np.resize([1, -1], control.action_dim)
        for _ in range(79):
            env.step(actions)
            state = env.physics_backend.get_state()
            np.testing.assert_allclose(state["qpos"][0], state["qpos"][1], atol=1e-6)
        assert not control.ready.any()
        env.step(np.zeros((2, control.action_dim), dtype=np.float32))
        env.step(np.zeros((2, control.action_dim), dtype=np.float32))
        assert control.ready.all()
        np.testing.assert_allclose(control.prepared_time, 0.79975, atol=0.02)
        env.reset(np.array([0], dtype=np.int32))
        assert control.elapsed[0] == 0 and not control.ready[0]
        assert control.ready[1] and control.elapsed[1] > 0.8
        assert not control.episodes[0].credit
        np.testing.assert_allclose(control._reference[0], control.home)
    finally:
        env.close()


def test_passive_contact_is_rewarded_once_and_result_tracking_continues():
    env = make_manager_based_rl_env(_cfg(), num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        hit_rewards = 0
        failure_rewards = 0
        outgoing_rewards = 0
        hit_step = None
        for step in range(400):
            state = env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            hit_rewards += int(return_reward(env, "hit")[0])
            failure_rewards += int(return_reward(env, "failure")[0])
            shaping = return_shaping_reward(
                env,
                "direction",
                env.cfg.rewards["outgoing_direction"].params["shaping_cfg"],
            )[0]
            outgoing_rewards += int(shaping > 0)
            if control.episodes[0].hit and hit_step is None:
                hit_step = step
                assert not state.terminated[0]
            if state.terminated[0] or state.truncated[0]:
                break
        episode = control.episodes[0]
        assert episode.opponent_bounced and episode.receiver_bounced and episode.hit
        assert episode.done and not episode.success
        assert episode.failure_reason == "out_of_bounds"
        assert hit_rewards == failure_rewards == 1
        assert outgoing_rewards == 1
        assert episode.outgoing_time > episode.hit_time
        assert hit_step is not None and step > hit_step
        assert episode.hit_time - control.prepared_time[0] >= 0.15
        assert np.all(control._peak_speed[0] <= control._max_velocity + 0.02)
        assert control._peak_force[0] <= 100
    finally:
        env.close()


def test_post_contact_forecast_matches_actual_mujoco_free_flight():
    env = make_manager_based_rl_env(_cfg(), num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        for _ in range(150):
            env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            episode = control.episodes[0]
            if episode.outgoing_time is not None:
                break
        assert episode.outgoing_time is not None
        for _ in range(5):
            env.step(np.zeros((1, control.action_dim), dtype=np.float32))
        duration = control.elapsed[0] - episode.outgoing_time
        expected = np.array(episode.outgoing_position) + duration * np.array(
            episode.outgoing_velocity
        )
        gravity = env.cfg.rewards["outgoing_direction"].params["shaping_cfg"].gravity
        expected[2] -= 0.5 * gravity * duration * duration
        observed = env.physics_backend.get_state(("qpos",))["qpos"][0, control._ball_qpos]
        np.testing.assert_allclose(observed, expected, atol=1e-4)
    finally:
        env.close()


@pytest.mark.parametrize("command", [-1, 1])
def test_saturated_swing_references_remain_inside_native_tracking_limits(command):
    import mujoco

    cfg = _cfg()
    env = make_manager_based_rl_env(cfg, num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        for _ in range(12000):
            control._advance_swing(0, 1, command, env.physics_dt)
            control._advance_swing(0, 5, command, env.physics_dt)
        model = mujoco.MjModel.from_xml_path(cfg.scene.model_file)
        for column, joint in ((1, "joint1"), (5, "joint6")):
            lower, upper = model.joint(joint).range
            assert lower <= control._reference[0, column] <= upper
            bound = lower if command < 0 else upper
            assert control._reference[0, column] == pytest.approx(bound, abs=3e-7)
        controls = control._tracking.compute(
            control._reference, control._reference_velocity, control._reference_acceleration
        )
        assert np.isfinite(controls).all()
    finally:
        env.close()
