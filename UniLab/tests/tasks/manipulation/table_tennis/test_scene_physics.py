"""Runtime physics checks for the Panthera table-tennis scene."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.tasks.manipulation.table_tennis.ball_launcher import (
    BallLaunch,
    BallLauncherCfg,
    sample_ball_launch,
)

mujoco = pytest.importorskip("mujoco")

_SCENE = Path(ASSETS_ROOT_PATH) / "robots" / "panthera" / "scene.xml"
_HOME = np.asarray([0.0, 0.3, 0.3, 0.0, 0.0, 1.5])
_JOINT_RANGES = np.asarray(
    [
        [-2.4, 2.4],
        [-0.1, 3.2],
        [-0.1, 4.0],
        [-1.6, 1.6],
        [-1.7, 1.7],
        [-2.5, 2.5],
    ]
)
_TORQUE_LIMITS = np.asarray([21.0, 36.0, 36.0, 21.0, 10.0, 10.0])
_KP = np.asarray([30.0, 50.0, 60.0, 25.0, 15.0, 10.0])
_KD = np.asarray([3.0, 5.0, 6.0, 2.5, 1.5, 1.0])


def _id(model, object_type, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    assert result >= 0, name
    return result


def _model_data():
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    data = mujoco.MjData(model)
    home_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    return model, data


def _touching(data, geom_a: int, geom_b: int) -> bool:
    pair = {geom_a, geom_b}
    return any(
        {data.contact[index].geom1, data.contact[index].geom2} == pair for index in range(data.ncon)
    )


def test_arm_actuator_contract_and_rail_lock_compile() -> None:
    model, _ = _model_data()
    joint_ids = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}") for index in range(1, 7)]
    )
    actuator_ids = np.asarray(
        [
            _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{index}_position")
            for index in range(1, 7)
        ]
    )
    rail_lock = _id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "rail_lock")

    np.testing.assert_allclose(model.jnt_range[joint_ids], _JOINT_RANGES)
    np.testing.assert_allclose(model.actuator_ctrlrange[actuator_ids], _JOINT_RANGES)
    np.testing.assert_allclose(model.actuator_forcerange[actuator_ids, 1], _TORQUE_LIMITS)
    np.testing.assert_allclose(model.actuator_forcerange[actuator_ids, 0], -_TORQUE_LIMITS)
    np.testing.assert_allclose(model.actuator_gainprm[actuator_ids, 0], _KP)
    np.testing.assert_allclose(-model.actuator_biasprm[actuator_ids, 2], _KD)
    assert model.eq_active0[rail_lock] == 1


def test_paddle_frame_origin_and_positive_z_define_contact_normal() -> None:
    model, data = _model_data()
    frame_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_frame")
    geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_contact")
    link6_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")

    np.testing.assert_allclose(data.xpos[frame_id], data.geom_xpos[geom_id], atol=1e-12)
    frame_normal = data.xmat[frame_id].reshape(3, 3)[:, 2]
    geom_axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(frame_normal, geom_axis, atol=1e-12)
    assert np.linalg.norm(frame_normal) == pytest.approx(1.0)
    assert model.body_mass[frame_id] == pytest.approx(0.0)
    assert model.body_mass[link6_id] == pytest.approx(0.340735)


def test_table_rebound_matches_ittf_drop_window() -> None:
    model, data = _model_data()
    ball_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = model.jnt_qposadr[ball_joint]
    ball_dof = model.jnt_dofadr[ball_joint]
    ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    data.qpos[ball_qpos : ball_qpos + 3] = [0.75, 0.0, 1.08]
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[ball_dof : ball_dof + 6] = 0.0
    mujoco.mj_forward(model, data)

    rebounding = False
    peak_rebound_above_table = -np.inf
    for _ in range(round(1.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        vertical_velocity = data.qvel[ball_dof + 2]
        if vertical_velocity > 0.0:
            rebounding = True
        if rebounding:
            ball_bottom = data.xpos[ball_body, 2] - 0.02
            peak_rebound_above_table = max(peak_rebound_above_table, ball_bottom - 0.76)
        if rebounding and vertical_velocity < 0.0:
            break

    assert rebounding
    assert 0.23 <= peak_rebound_above_table <= 0.26


def test_fixed_incoming_ball_crosses_net_then_bounces_on_robot_side() -> None:
    model, data = _model_data()
    ball_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = model.jnt_qposadr[ball_joint]
    ball_dof = model.jnt_dofadr[ball_joint]
    ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    table_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    net_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "net_mesh")
    rail_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "rail_joint")
    rail_qpos = model.jnt_qposadr[rail_joint]

    np.testing.assert_allclose(data.qpos[1:7], _HOME)
    np.testing.assert_allclose(data.qpos[ball_qpos : ball_qpos + 3], [0.75, 0.0, 0.95])
    data.qvel[ball_dof : ball_dof + 6] = [-2.5, 0.0, 1.6, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    crossed_net: tuple[float, float] | None = None
    first_bounce: tuple[float, float] | None = None
    hit_region_heights: dict[float, float] = {}
    touched_net = False
    max_rail_displacement = 0.0
    previous_x = data.qpos[ball_qpos]
    for _ in range(round(1.4 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        x = float(data.qpos[ball_qpos])
        z = float(data.qpos[ball_qpos + 2])
        max_rail_displacement = max(max_rail_displacement, abs(float(data.qpos[rail_qpos])))
        touched_net |= _touching(data, ball_geom, net_geom)
        if crossed_net is None and previous_x > 0.0 >= x:
            crossed_net = (float(data.time), z)
        if first_bounce is None and _touching(data, ball_geom, table_geom):
            first_bounce = (float(data.time), x)
        for threshold in (-0.8, -1.0):
            if threshold not in hit_region_heights and previous_x > threshold >= x:
                hit_region_heights[threshold] = z
        if x <= -1.05:
            break
        previous_x = x

    assert crossed_net is not None
    assert crossed_net[0] == pytest.approx(0.30, abs=0.015)
    assert crossed_net[1] > 0.9325
    assert not touched_net
    assert first_bounce is not None
    assert first_bounce[0] == pytest.approx(0.412, abs=0.02)
    assert first_bounce[1] == pytest.approx(-0.28, abs=0.08)
    assert set(hit_region_heights) == {-0.8, -1.0}
    assert all(0.78 < height < 1.1 for height in hit_region_heights.values())
    assert max_rail_displacement < 1e-8


def _validate_random_feed(model, data, launch: BallLaunch) -> tuple[float, float, float]:
    """Return bounce X, rebound peak above the table and end-line crossing time.

    This is a ball-trajectory check with robot collisions disabled. Continue
    through the end line and first rebound apex; it does not prove arm reach.
    """
    home_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    ball_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qpos = model.jnt_qposadr[ball_joint]
    dof = model.jnt_dofadr[ball_joint]
    ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    table_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    net_geoms = [
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("net_mesh", "net_post_pos", "net_post_neg")
    ]
    data.qpos[qpos : qpos + 3] = launch.position
    data.qvel[dof : dof + 3] = launch.linear_velocity
    data.qvel[dof + 3 : dof + 6] = 0.0
    mujoco.mj_forward(model, data)

    crossed_net = False
    bounce_x = None
    bounce_count = 0
    touching_table = False
    rebounded = False
    peak_above_table = -np.inf
    exit_time = None
    for _ in range(round(1.2 / model.opt.timestep)):
        previous = data.qpos[qpos : qpos + 3].copy()
        previous_time = float(data.time)
        mujoco.mj_step(model, data)
        current = data.qpos[qpos : qpos + 3]
        assert all(not _touching(data, ball_geom, geom) for geom in net_geoms)
        if previous[0] > 0.0 >= current[0]:
            alpha = previous[0] / (previous[0] - current[0])
            net_z = previous[2] + alpha * (current[2] - previous[2])
            assert net_z > 0.9125 + 0.02
            crossed_net = True

        contact = _touching(data, ball_geom, table_geom)
        if contact and not touching_table:
            bounce_count += 1
            assert bounce_count == 1, "Second table bounce before completing the first rebound"
            assert crossed_net, "Table contact before net crossing"
            bounce_x = float(current[0])
            assert -1.1 <= bounce_x <= -0.9
            assert abs(current[1]) < 0.7425
            assert abs(bounce_x - launch.bounce_position[0]) < 0.02
        touching_table = contact
        if bounce_count and data.qvel[dof + 2] > 0:
            rebounded = True
        if rebounded:
            peak_above_table = max(peak_above_table, float(current[2] - 0.76))

        if exit_time is None and previous[0] > -1.39 >= current[0]:
            assert crossed_net and rebounded and bounce_count == 1
            alpha = (-1.39 - previous[0]) / (current[0] - previous[0])
            height = float(previous[2] + alpha * (current[2] - previous[2]))
            exit_time = previous_time + alpha * model.opt.timestep
            assert height > 0.76 + 0.02
            assert abs(current[1] - launch.position[1]) < 0.002

        if exit_time is not None and rebounded and data.qvel[dof + 2] < 0:
            assert bounce_x is not None
            assert bounce_count == 1
            # The retained legacy direct-feed window under the new contact
            # calibration; slow two-bounce service has separate acceptance.
            assert 0.33 <= peak_above_table <= 0.37
            return bounce_x, peak_above_table, float(exit_time)
    pytest.fail("Random ball did not pass the end line and reach its rebound apex in 1.2 seconds")


def test_random_feeds_clear_net_bounce_once_pass_end_line_and_reach_target_height() -> None:
    from scripts.tools.preview_panthera_ball_launcher import _load_preview_model

    model = _load_preview_model(robot_collisions=False)
    data = mujoco.MjData(model)
    cfg = BallLauncherCfg()
    np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, -cfg.gravity])
    table_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    assert model.geom_pos[table_geom, 2] + model.geom_size[table_geom, 2] == pytest.approx(
        cfg.table_z
    )
    assert model.geom_size[ball_geom, 0] == pytest.approx(cfg.ball_radius)
    np.testing.assert_allclose(
        model.geom_size[table_geom, :2], [cfg.table_half_length, cfg.table_half_width]
    )
    net_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "net_mesh")
    assert model.geom_size[net_geom, 0] == pytest.approx(cfg.net_half_thickness)
    assert model.geom_pos[net_geom, 2] + model.geom_size[net_geom, 2] == pytest.approx(
        cfg.net_top_z
    )
    rng = np.random.default_rng(20261002)
    launches = [sample_ball_launch(rng, cfg) for _ in range(128)]
    for lateral, speed, bounce_time in product(
        cfg.lateral_range, cfg.speed_range, cfg.bounce_times
    ):
        corner = replace(
            cfg,
            lateral_range=(lateral, lateral),
            speed_range=(speed, speed),
            bounce_times=(bounce_time,),
        )
        launches.append(sample_ball_launch(rng, corner))

    results = np.asarray([_validate_random_feed(model, data, launch) for launch in launches])
    assert np.ptp(results[:, 1]) > 0.015, "Post-bounce height must actually vary"


def test_full_scene_preview_preserves_passive_robot_ball_contacts() -> None:
    from scripts.tools.preview_panthera_ball_launcher import _load_preview_model

    model = _load_preview_model(robot_collisions=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    cfg = BallLauncherCfg(lateral_range=(0.0, 0.0), speed_range=(3.46, 3.46))
    launch = sample_ball_launch(np.random.default_rng(7), cfg)
    qpos = model.joint("ball_free").qposadr[0]
    dof = model.joint("ball_free").dofadr[0]
    ball_geom = model.geom("ball_geom").id
    paddle_geom = model.geom("paddle_contact").id
    data.qpos[qpos : qpos + 3] = launch.position
    data.qvel[dof : dof + 3] = launch.linear_velocity
    mujoco.mj_forward(model, data)

    touched_paddle = False
    for _ in range(round(1.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        touched_paddle |= _touching(data, ball_geom, paddle_geom)
    assert touched_paddle, "Full-scene preview must retain real passive home contacts"
