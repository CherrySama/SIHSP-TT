"""Measured preparation deadlines and safety under the actual coupled dynamics."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

pytest.importorskip("rail_alignment", reason="Optional table-tennis core not installed")
mujoco = pytest.importorskip("mujoco")

from ball_state import BallState
from scripts.tools.preview_panthera_ball_launcher import _draw_prediction, _load_preview_model

from unilab.tasks.manipulation.table_tennis.ball_launcher import (
    BallLaunch,
    BallLauncherCfg,
    sample_ball_launch,
)
from unilab.tasks.manipulation.table_tennis.prediction_adapter import PantheraBallPredictor
from unilab.visualization.panthera_rail_preview import (
    PantheraRailPreview,
    draw_rail_alignment,
)


def measure_rail_preparation(model, data, launch: BallLaunch) -> dict[str, float]:
    """Launch metadata initializes physics; only measured state reaches control.

    Prediction/target updates run at 100 Hz, the native inner force loop at
    the scene's physics rate. Safety is measured at every physics step.
    Default preview contact isolation makes this a preparation test, not a hit.
    """
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    ball = model.joint("ball_free")
    qpos, dof = int(ball.qposadr[0]), int(ball.dofadr[0])
    data.qpos[qpos : qpos + 3] = launch.position
    data.qvel[dof : dof + 3] = launch.linear_velocity
    data.qvel[dof + 3 : dof + 6] = 0.0
    rail = PantheraRailPreview(model, data)
    predictor = PantheraBallPredictor()
    cfg = rail.controller.cfg
    arm_qpos = np.asarray([int(model.joint(f"joint{i}").qposadr[0]) for i in (1, 2, 3, 4, 6)])
    arm_dof = np.asarray([int(model.joint(f"joint{i}").dofadr[0]) for i in (1, 2, 3, 4, 6)])
    arm_ids = np.asarray([model.joint(f"joint{i}").id for i in (1, 2, 3, 4, 6)])
    link4, link5 = model.body("link4").id, model.body("link5").id
    ball_geom, table_geom = model.geom("ball_geom").id, model.geom("table_top").id
    net_geoms = {model.geom(name).id for name in ("net_mesh", "net_post_pos", "net_post_neg")}
    maxima = np.zeros(6)
    crossing_time = crossing_error = crossing_speed = None
    ready_time = None
    touching_table = crossed_net = False
    bounce_count = 0
    dt = float(model.opt.timestep)
    for step in range(round(1.5 / dt)):
        mujoco.mj_forward(model, data)
        if step % round(0.01 / dt) == 0:
            predictor.cfg = replace(predictor.cfg, candidate_x=float(rail.paddle_position[0]))
            state = BallState(
                float(data.time),
                tuple(data.qpos[qpos : qpos + 3]),
                tuple(data.qvel[dof : dof + 3]),
            )
            prediction = predictor.update(state)
            rail.observe(prediction)
            if ready_time is None and rail.ready:
                ready_time = float(data.time)
        previous_ball = data.qpos[qpos : qpos + 3].copy()
        previous_paddle = rail.paddle_position
        previous_time = float(data.time)
        previous_velocity = rail.velocity
        rail.apply_force()
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        current_ball = data.qpos[qpos : qpos + 3].copy()
        acceleration = (rail.velocity - previous_velocity) / dt
        maxima = np.maximum(
            maxima,
            [
                abs(rail.velocity),
                abs(acceleration),
                abs(float(data.actuator_force[rail.actuator])),
                np.max(np.abs(data.xmat[link4] - data.xmat[link5])),
                np.max(np.abs(data.qvel[arm_dof])),
                abs(rail.position),
            ],
        )
        assert cfg.lower <= rail.position <= cfg.upper
        assert np.all(data.qpos[arm_qpos] >= model.jnt_range[arm_ids, 0])
        assert np.all(data.qpos[arm_qpos] <= model.jnt_range[arm_ids, 1])
        if previous_ball[0] > 0 >= current_ball[0]:
            crossed_net = True
        contacts = [{c.geom1, c.geom2} for c in data.contact[: data.ncon]]
        assert all(not (ball_geom in pair and pair & net_geoms) for pair in contacts)
        contact = {ball_geom, table_geom} in contacts
        if contact and not touching_table and crossing_time is None:
            bounce_count += 1
            assert crossed_net and bounce_count == 1
            assert -1.1 <= current_ball[0] <= -0.9
        touching_table = contact
        previous_distance = previous_ball[0] - previous_paddle[0]
        distance = current_ball[0] - rail.paddle_position[0]
        if crossing_time is None and previous_distance > 0 >= distance:
            assert bounce_count == 1
            fraction = previous_distance / (previous_distance - distance)
            crossing_time = previous_time + fraction * dt
            crossing_error = abs(float(rail.paddle_position[1] - current_ball[1]))
            crossing_speed = abs(rail.velocity)
    assert ready_time is not None, (launch.position, maxima)
    assert crossing_time is not None and crossing_error is not None and crossing_speed is not None
    lead_time = crossing_time - ready_time
    assert lead_time >= cfg.preparation_margin, (launch.position, lead_time, maxima)
    assert crossing_error <= cfg.position_tolerance
    assert crossing_speed <= cfg.settled_velocity
    assert maxima[0] <= cfg.max_velocity
    assert maxima[1] <= cfg.max_acceleration
    assert maxima[2] <= cfg.max_force
    assert maxima[3] <= 1e-12, "Rigid link5 orientation changed relative to link4"
    assert maxima[4] <= 1.0, "Measured arm velocity exceeded the existing limit"
    return {
        "lead_time": lead_time,
        "crossing_y_error": crossing_error,
        "crossing_rail_speed": crossing_speed,
        **dict(
            zip(
                (
                    "max_speed",
                    "max_acceleration",
                    "max_force",
                    "joint5_drift",
                    "arm_speed",
                    "rail_position",
                ),
                maxima,
                strict=True,
            )
        ),
    }


def test_random_and_boundary_feeds_prepare_in_time_without_exceeding_limits() -> None:
    model = _load_preview_model(robot_collisions=False, rail_align=True)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(20261003)
    cfg = BallLauncherCfg()
    launches = [sample_ball_launch(rng, cfg) for _ in range(64)]
    for lateral, speed, bounce_time in product(
        cfg.lateral_range, cfg.speed_range, cfg.bounce_times
    ):
        launches.append(
            sample_ball_launch(
                rng,
                replace(
                    cfg,
                    lateral_range=(lateral, lateral),
                    speed_range=(speed, speed),
                    bounce_times=(bounce_time,),
                ),
            )
        )
    for launch in launches:
        measure_rail_preparation(model, data, launch)


def test_reset_releases_only_rail_and_discards_previous_target() -> None:
    model = _load_preview_model(robot_collisions=False, rail_align=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rail = PantheraRailPreview(model, data)
    predictor = PantheraBallPredictor()
    prediction = predictor.update(BallState(0.0, (0.75, 0.4, 0.95), (-3.46, 0.0, 2.13)))
    rail.observe(prediction)
    assert rail.controller.target is not None
    rail.apply_force()
    assert rail.last_force > 0
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rail.reset()
    assert rail.controller.target is None and rail.first_ready_time is None
    assert not data.eq_active[rail.rail_lock]
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint5") == -1
    assert model.eq_active0[rail.rail_lock]
    assert data.ctrl[rail.actuator] == 0.0
    np.testing.assert_allclose(data.ctrl[:5], [0.0, 0.3, 0.3, 0.0, 1.5])


def test_rail_markers_are_visual_only_and_bounded() -> None:
    model = _load_preview_model(robot_collisions=False, rail_align=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rail = PantheraRailPreview(model, data)
    predictor = PantheraBallPredictor()
    prediction = predictor.update(BallState(0.0, (0.75, 0.4, 0.95), (-3.46, 0.0, 2.13)))
    rail.observe(prediction)
    before = model.geom_contype.copy()
    scene = mujoco.MjvScene(model, maxgeom=3)
    _draw_prediction(scene, None, [])
    draw_rail_alignment(scene, rail)
    assert scene.ngeom == 3
    np.testing.assert_array_equal(model.geom_contype, before)


def test_rail_assembly_keeps_five_centimetre_gap_across_full_travel() -> None:
    """Measure solid surfaces with full collisions, not body origins alone."""
    model = _load_preview_model(robot_collisions=True, rail_align=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rail = PantheraRailPreview(model, data)
    table = model.geom("table_top").id
    assembly = [
        model.geom(name).id
        for name in ("rail_carriage_geom", "rail_beam", "rail_support_pos", "rail_support_neg")
    ]
    robot_bodies = {model.body("rail_carriage").id}
    for body in range(model.nbody):
        if model.body_parentid[body] in robot_bodies:
            robot_bodies.add(body)
    robot_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, list(robot_bodies))))
    for lateral in np.linspace(-0.75, 0.75, 7):
        data.qpos[rail.qpos] = lateral
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.geom_xpos[assembly, 0], data.xpos[model.body("rail_carriage").id, 0]
        )
        np.testing.assert_allclose(data.geom_xmat[table].reshape(3, 3), np.eye(3))
        for geom in assembly:
            np.testing.assert_allclose(data.geom_xmat[geom].reshape(3, 3), np.eye(3))
            # These boxes are axis-aligned, so their measured positions give
            # an exact separating-axis clearance without an iterative query.
            separation = (
                np.abs(data.geom_xpos[geom] - data.geom_xpos[table])
                - model.geom_size[geom]
                - model.geom_size[table]
            )
            assert np.any(separation > 0)
        table_end = data.geom_xpos[table, 0] - model.geom_size[table, 0]
        front = np.max(data.geom_xpos[assembly, 0] + model.geom_size[assembly, 0])
        assert table_end - front == pytest.approx(0.05, abs=1e-12)
        for contact in data.contact[: data.ncon]:
            pair = {contact.geom1, contact.geom2}
            assert not (table in pair and pair & robot_geoms), "Robot/rail intersects the table"
