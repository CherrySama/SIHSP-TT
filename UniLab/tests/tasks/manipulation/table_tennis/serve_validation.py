"""Independent contact, prediction and measured rail checks for slow serves."""

from dataclasses import replace

import mujoco
import numpy as np
from ball_state import BallState

from unilab.tasks.manipulation.table_tennis.prediction_adapter import PantheraBallPredictor
from unilab.visualization.panthera_rail_preview import PantheraRailPreview


def measure_serve(model, data, launch, *, rail_align=False):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    qpos = int(model.joint("ball_free").qposadr[0])
    dof = int(model.joint("ball_free").dofadr[0])
    data.qpos[qpos : qpos + 3] = launch.position
    data.qvel[dof : dof + 3] = launch.linear_velocity
    data.qvel[dof + 3 : dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    rail = PantheraRailPreview(model, data) if rail_align else None
    predictor = PantheraBallPredictor()
    dt = float(model.opt.timestep)
    sample_steps = round(0.01 / dt)
    ball, table = model.geom("ball_geom").id, model.geom("table_top").id
    nets = {model.geom(name).id for name in ("net_mesh", "net_post_pos", "net_post_neg")}
    contacts, forecasts = [], []
    touching = False
    net_time = net_height = crossing = None
    ready_time = None
    crossing_y_error = crossing_rail_speed = None
    peak = 0.0
    maxima = np.zeros(6)
    if rail is not None:
        arms = [model.joint(f"joint{i}") for i in (1, 2, 3, 4, 6)]
        arm_qpos = np.asarray([int(j.qposadr[0]) for j in arms])
        arm_dof = np.asarray([int(j.dofadr[0]) for j in arms])
        arm_ids = np.asarray([j.id for j in arms])
        link4, link5 = model.body("link4").id, model.body("link5").id
    for step in range(round(1.5 / dt)):
        mujoco.mj_forward(model, data)
        if step % sample_steps == 0:
            if rail is not None:
                predictor.cfg = replace(predictor.cfg, candidate_x=float(rail.paddle_position[0]))
            state = BallState(
                float(data.time),
                tuple(data.qpos[qpos : qpos + 3]),
                tuple(data.qvel[dof : dof + 3]),
            )
            prediction = predictor.update(state)
            if prediction.valid:
                forecasts.append(prediction)
                if not contacts:
                    assert prediction.opponent_bounce is not None
                if len(contacts) < 2:
                    assert not predictor.has_bounced
            if rail is not None:
                rail.observe(prediction)
                if ready_time is None and rail.ready:
                    ready_time = float(data.time)
        previous = data.qpos[qpos : qpos + 3].copy()
        previous_time = float(data.time)
        previous_x = float(rail.paddle_position[0]) if rail is not None else -1.39
        if rail is not None:
            previous_velocity = rail.velocity
            rail.apply_force()
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        current = data.qpos[qpos : qpos + 3].copy()
        pairs = [{c.geom1, c.geom2} for c in data.contact[: data.ncon]]
        assert all(not (ball in pair and pair & nets) for pair in pairs)
        contact = {ball, table} in pairs
        if contact and not touching:
            contacts.append((float(data.time), current.copy()))
            assert len(contacts) <= 2
            if len(contacts) == 1:
                assert net_time is None and 0.026 < current[0] < 1.35
            else:
                assert net_time is not None and -1.1 <= current[0] <= -0.9
        touching = contact
        if previous[0] > 0 >= current[0]:
            fraction = previous[0] / (previous[0] - current[0])
            net_time = previous_time + fraction * dt
            net_height = float(previous[2] + fraction * (current[2] - previous[2]))
            assert len(contacts) == 1 and net_height > 0.9325
        if len(contacts) == 2:
            peak = max(peak, float(current[2] - 0.76))
        section = float(rail.paddle_position[0]) if rail is not None else -1.39
        before, after = previous[0] - previous_x, current[0] - section
        if crossing is None and before > 0 >= after:
            assert len(contacts) == 2
            fraction = before / (before - after)
            position = previous + fraction * (current - previous)
            crossing = (previous_time + fraction * dt, position)
            if rail is not None:
                crossing_y_error = float(abs(rail.paddle_position[1] - position[1]))
                crossing_rail_speed = abs(rail.velocity)
                assert crossing_y_error <= 0.01
                assert crossing_rail_speed <= 0.05
                assert 0.20 <= position[2] - 0.76 <= 0.30
        if rail is not None:
            maxima = np.maximum(
                maxima,
                [
                    abs(rail.velocity),
                    abs((rail.velocity - previous_velocity) / dt),
                    abs(float(data.actuator_force[rail.actuator])),
                    np.max(np.abs(data.xmat[link4] - data.xmat[link5])),
                    np.max(np.abs(data.qvel[arm_dof])),
                    abs(rail.position),
                ],
            )
            assert np.all(data.qpos[arm_qpos] >= model.jnt_range[arm_ids, 0])
            assert np.all(data.qpos[arm_qpos] <= model.jnt_range[arm_ids, 1])
        if crossing is not None and len(contacts) == 2 and peak > 0.1 and data.qvel[dof + 2] < 0:
            break
    assert crossing is not None and net_time is not None and len(contacts) == 2
    assert 0.28 <= peak <= 0.36
    assert forecasts, "No usable prediction before actual reception"
    errors = []
    for forecast in forecasts:
        assert forecast.candidate is not None
        bounce = forecast.first_bounce
        errors.append(
            [
                0
                if bounce is None
                else np.linalg.norm(np.asarray(bounce.position) - contacts[1][1]),
                np.linalg.norm(np.asarray(forecast.candidate.position) - crossing[1]),
                abs(forecast.candidate.timestamp - crossing[0]),
            ]
        )
    maximum_errors = np.max(errors, axis=0)
    assert np.all(maximum_errors <= [0.02, 0.03, 0.02]), maximum_errors
    result = {
        "opponent_bounce_x": float(contacts[0][1][0]),
        "receiver_bounce_x": float(contacts[1][1][0]),
        "rebound_peak": peak,
        "net_clearance": net_height - 0.9325,
        "candidate_height": float(crossing[1][2] - 0.76),
        "candidate_time": crossing[0],
        "prediction_errors": maximum_errors.tolist(),
    }
    if rail is not None:
        assert ready_time is not None
        lead = crossing[0] - ready_time
        assert lead >= 0.15, lead
        limits = rail.controller.cfg
        assert np.all(
            maxima
            <= [
                limits.max_velocity,
                limits.max_acceleration,
                limits.max_force,
                1e-12,
                1.0,
                limits.upper,
            ]
        ), maxima
        result.update(
            lead_time=lead,
            crossing_y_error=crossing_y_error,
            crossing_rail_speed=crossing_rail_speed,
            maxima=maxima.tolist(),
        )
    return result
