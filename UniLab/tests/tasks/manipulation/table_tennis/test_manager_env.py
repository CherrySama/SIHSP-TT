"""End-to-end Manager-Based reset and control checks on MuJoCo."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs import make_manager_based_rl_env
from unilab.tasks.manipulation.table_tennis import (
    BallLauncherCfg,
    ServeLauncherCfg,
    build_panthera_table_tennis_env_cfg,
)
from unilab.tasks.manipulation.table_tennis.manager_terms import (
    PantheraIncrementalPositionAction,
)

pytest.importorskip(
    "unisim.backend.mujoco.backend",
    reason="unisim-core MuJoCo adapter (mjbatch build) not available",
)
mujoco = pytest.importorskip("mujoco")

_HOME = np.asarray([0.0, 0.3, 0.3, 0.0, 0.0, 1.5])


def test_fixed_feed_reset_and_incremental_action_run_through_manager_env() -> None:
    cfg = build_panthera_table_tennis_env_cfg(ball_feed="fixed")
    assert cfg.sim_dt == pytest.approx(0.00025)
    assert cfg.ctrl_dt == pytest.approx(0.01)
    env = make_manager_based_rl_env(cfg, num_envs=1, backend_type="mujoco")
    try:
        state = env.init_state()
        action = env.action_manager.get_term("arm_target_delta")
        assert isinstance(action, PantheraIncrementalPositionAction)

        assert state.obs["obs"].shape == (1, 12)
        np.testing.assert_allclose(env.scene["robot"].data.joint_pos[0], _HOME)
        np.testing.assert_allclose(env.scene["robot"].data.joint_vel[0], 0.0)
        np.testing.assert_allclose(env.scene["ball"].data.root_link_pos_w[0], [0.75, 0.0, 0.95])
        np.testing.assert_allclose(env.scene["ball"].data.root_link_lin_vel_w[0], [-2.5, 0.0, 1.6])

        state = env.step(np.ones((1, 6), dtype=np.float32))
        np.testing.assert_allclose(action.target[0], _HOME + 0.01, atol=1e-7)
        assert not state.terminated[0]
        assert not state.truncated[0]
        assert state.reward[0] == pytest.approx(0.0)
        assert env.scene["ball"].data.root_link_pos_w[0, 0] < 0.75

        env.reset()
        np.testing.assert_allclose(action.target[0], _HOME, atol=1e-7)
        np.testing.assert_allclose(env.scene["robot"].data.joint_pos[0], _HOME)
        np.testing.assert_allclose(env.scene["ball"].data.root_link_pos_w[0], [0.75, 0.0, 0.95])
        np.testing.assert_allclose(env.scene["ball"].data.root_link_lin_vel_w[0], [-2.5, 0.0, 1.6])
    finally:
        env.close()


def test_random_ball_flight_matches_scene_through_manager_env() -> None:
    """Check the reset event and batched backend through the real table bounce."""
    # A side lane avoids passive home hits while checking both backends through
    # the table rebound and the end line. Robot collisions stay enabled.
    cfg = build_panthera_table_tennis_env_cfg(
        launcher_cfg=BallLauncherCfg(lateral_range=(-0.12, -0.12))
    )
    env = make_manager_based_rl_env(cfg, num_envs=1, backend_type="mujoco")
    model = mujoco.MjModel.from_xml_path(str(ASSETS_ROOT_PATH / "robots/panthera/scene.xml"))
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qpos = model.jnt_qposadr[joint]
    dof = model.jnt_dofadr[joint]
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    try:
        env.init_state()
        for _ in range(3):
            env.reset()
            ball = env.scene["ball"]
            mujoco.mj_resetDataKeyframe(model, data, home)
            data.qpos[qpos : qpos + 3] = ball.data.root_link_pos_w[0]
            data.qvel[dof : dof + 3] = ball.data.root_link_lin_vel_w[0]
            mujoco.mj_forward(model, data)
            for _ in range(120):
                state = env.step(np.zeros((1, 6), dtype=np.float32))
                for _ in range(cfg.sim_substeps):
                    mujoco.mj_step(model, data)
                np.testing.assert_allclose(
                    ball.data.root_link_pos_w[0], data.qpos[qpos : qpos + 3], atol=1e-5
                )
                np.testing.assert_allclose(
                    ball.data.root_link_lin_vel_w[0], data.qvel[dof : dof + 3], atol=1e-4
                )
                assert not state.terminated[0] and not state.truncated[0]
                if data.qpos[qpos] <= -1.39:
                    break
            else:
                pytest.fail("Manager-Based random feed did not pass the robot-side end line")
    finally:
        env.close()


def test_random_feed_reset_is_seeded_no_spin_and_partial_row_scoped() -> None:
    cfg = build_panthera_table_tennis_env_cfg()
    launcher = ServeLauncherCfg()
    env = make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    try:
        env.init_state()
        env.reset(seed=31)
        ball = env.scene["ball"]
        positions = ball.data.root_link_pos_w.copy()
        velocities = ball.data.root_link_lin_vel_w.copy()
        np.testing.assert_allclose(ball.data.root_link_ang_vel_w, 0.0)
        np.testing.assert_allclose(env.scene["robot"].data.joint_pos, np.tile(_HOME, (2, 1)))
        assert np.all(np.abs(positions[:, 1]) <= launcher.lateral_range[1])
        assert np.all(-velocities[:, 0] >= launcher.speed_range[0])
        assert np.all(-velocities[:, 0] <= launcher.speed_range[1])
        assert not np.array_equal(positions[0], positions[1])

        env.reset(seed=31)
        np.testing.assert_array_equal(ball.data.root_link_pos_w, positions)
        np.testing.assert_array_equal(ball.data.root_link_lin_vel_w, velocities)
        env.reset(env_ids=np.asarray([1], dtype=np.int32))
        np.testing.assert_array_equal(ball.data.root_link_pos_w[0], positions[0])
        np.testing.assert_array_equal(ball.data.root_link_lin_vel_w[0], velocities[0])
        assert not np.array_equal(ball.data.root_link_lin_vel_w[1], velocities[1])

        state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert not np.any(state.terminated)
        assert not np.any(state.truncated)
        assert state.obs["obs"].shape == (2, 12)
    finally:
        env.close()
