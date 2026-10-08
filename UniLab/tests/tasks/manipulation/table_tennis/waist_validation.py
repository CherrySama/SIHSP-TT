"""Measured native waist/rail motion; reported metrics are not success labels."""

from dataclasses import replace
from time import perf_counter

import mujoco
import numpy as np
from ball_state import BallState
from rail_alignment import WaistPreparationCfg

from unilab.tasks.manipulation.table_tennis.prediction_adapter import PantheraBallPredictor
from unilab.visualization.panthera_rail_preview import PantheraRailPreview


def measure_waist(model, data, launch, *, cfg=None, height_prepare=False, duration=1.35):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    ball = model.joint("ball_free")
    qpos, dof = int(ball.qposadr[0]), int(ball.dofadr[0])
    data.qpos[qpos : qpos + 3] = launch.position
    data.qvel[dof : dof + 3] = launch.linear_velocity
    rail = PantheraRailPreview(
        model,
        data,
        waist_preparation=cfg or WaistPreparationCfg(),
        height_preparation=height_prepare,
    )
    predictor = PantheraBallPredictor()
    predictor.cfg = replace(predictor.cfg, candidate_x=rail.candidate_x)
    dt = float(model.opt.timestep)
    sample_steps = round(0.01 / dt)
    peaks = np.zeros(5)
    crossing = None
    first_command = first_yaw_motion = first_rail_motion = None
    arm_settled_time = None
    contacts = {}
    trace = []
    max_rail_speed = max_rail_acceleration = max_rail_force = 0.0
    max_yaw_force = max_rail_position = max_link5_drift = 0.0
    initial_target = None
    first_arm_motion = [None] * 4
    max_observe_seconds = 0.0
    peak_arm_forces = np.zeros(5)
    first_paddle_contact = None
    link4, link5 = model.body("link4").id, model.body("link5").id
    for step in range(round(duration / dt)):
        mujoco.mj_forward(model, data)
        if step % sample_steps == 0:
            prediction = predictor.update(
                BallState(
                    float(data.time),
                    tuple(data.qpos[qpos : qpos + 3]),
                    tuple(data.qvel[dof : dof + 3]),
                )
            )
            started = perf_counter()
            rail.observe(prediction)
            max_observe_seconds = max(max_observe_seconds, perf_counter() - started)
            target = rail.controller.target
            if target is not None and initial_target is None:
                initial_target = {
                    "time": float(data.time),
                    "arrival_time": target.arrival_time,
                    "rail": target.position,
                    "yaw": rail.waist.target,
                    "arm": rail.target_arm.tolist(),
                    "height": None if rail.height is None else rail.height.target_height,
                }
            trace.append(
                {
                    "time": float(data.time),
                    "qpos": data.qpos.tolist(),
                    "arm": data.qpos[rail.arm_qpos].tolist(),
                    "arm_velocity": data.qvel[rail.arm_dof].tolist(),
                    "ball": data.qpos[qpos : qpos + 3].tolist(),
                    "rail": rail.position,
                    "rail_velocity": rail.velocity,
                    "paddle": rail.paddle_position.tolist(),
                    "normal": data.xmat[rail.paddle].reshape(3, 3)[:, 2].tolist(),
                    "ready": rail.ready,
                }
            )
        old_ball = data.qpos[qpos : qpos + 3].copy()
        previous_rail_speed = rail.velocity
        rail.apply_force()
        if first_command is None and abs(data.ctrl[rail.arm_actuators[0]]) > 1e-12:
            first_command = {
                "time": float(data.time),
                "yaw_target": float(data.ctrl[rail.arm_actuators[0]]),
                "rail_force": rail.last_force,
            }
        held = slice(4, None) if height_prepare else slice(1, None)
        np.testing.assert_array_equal(data.ctrl[rail.arm_actuators[held]], rail.home_arm[held])
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        speeds = np.abs(data.qvel[rail.arm_dof])
        peaks = np.maximum(peaks, speeds)
        peak_arm_forces = np.maximum(
            peak_arm_forces, np.abs(data.actuator_force[rail.arm_actuators])
        )
        for index, speed in enumerate(speeds[:4]):
            if speed > 1e-4 and first_arm_motion[index] is None:
                first_arm_motion[index] = float(data.time)
        max_rail_speed = max(max_rail_speed, abs(rail.velocity))
        max_rail_acceleration = max(
            max_rail_acceleration, abs((rail.velocity - previous_rail_speed) / dt)
        )
        max_rail_force = max(max_rail_force, abs(float(data.actuator_force[rail.actuator])))
        max_yaw_force = max(max_yaw_force, abs(float(data.actuator_force[rail.arm_actuators[0]])))
        max_rail_position = max(max_rail_position, abs(rail.position))
        max_link5_drift = max(
            max_link5_drift, float(np.max(np.abs(data.xmat[link4] - data.xmat[link5])))
        )
        if speeds[0] > 1e-4 and first_yaw_motion is None:
            first_yaw_motion = float(data.time)
        if abs(rail.velocity) > 1e-4 and first_rail_motion is None:
            first_rail_motion = float(data.time)
        if rail.waist.target is not None and arm_settled_time is None:
            goal = rail.target_arm
            if np.max(np.abs(data.qpos[rail.arm_qpos] - goal)) <= 0.015 and max(speeds) <= 0.05:
                arm_settled_time = float(data.time)
        for contact in data.contact[: data.ncon]:
            names = tuple(
                sorted(
                    (
                        model.geom(contact.geom1).name or f"geom_{contact.geom1}",
                        model.geom(contact.geom2).name or f"geom_{contact.geom2}",
                    )
                )
            )
            contacts.setdefault(names, float(data.time))
            if names == ("ball_geom", "paddle_contact") and first_paddle_contact is None:
                first_paddle_contact = {
                    "time": float(data.time),
                    "ball": data.qpos[qpos : qpos + 3].tolist(),
                    "paddle": rail.paddle_position.tolist(),
                    "height_error": float(abs(data.qpos[qpos + 2] - rail.paddle_position[2])),
                    "lateral_error": float(abs(data.qpos[qpos + 1] - rail.paddle_position[1])),
                }
        current_ball = data.qpos[qpos : qpos + 3]
        if crossing is None and old_ball[0] > rail.candidate_x >= current_ball[0]:
            fraction = (old_ball[0] - rail.candidate_x) / (old_ball[0] - current_ball[0])
            point = old_ball + fraction * (current_ball - old_ball)
            normal = data.xmat[rail.paddle].reshape(3, 3)[:, 2]
            crossing = {
                "time": float(data.time - dt + fraction * dt),
                "ball": point.tolist(),
                "paddle": rail.paddle_position.tolist(),
                "lateral_error": float(abs(point[1] - rail.paddle_position[1])),
                "height_error": float(abs(point[2] - rail.paddle_position[2])),
                "yaw_degrees": float(np.rad2deg(data.qpos[rail.arm_qpos[0]])),
                "facing_error_degrees": float(np.rad2deg(np.arccos(np.clip(abs(normal[0]), 0, 1)))),
            }
    return {
        "initial_target": initial_target,
        "first_command": first_command,
        "first_yaw_motion": first_yaw_motion,
        "first_rail_motion": first_rail_motion,
        "first_arm_motion": first_arm_motion,
        "max_observe_seconds": max_observe_seconds,
        "first_ready_time": rail.first_ready_time,
        "first_arm_settled_time": arm_settled_time,
        "crossing": crossing,
        "lead_time": None
        if crossing is None or rail.first_ready_time is None
        else crossing["time"] - rail.first_ready_time,
        "peak_arm_speeds": peaks.tolist(),
        "peak_arm_forces": peak_arm_forces.tolist(),
        "first_paddle_contact": first_paddle_contact,
        "max_rail_speed": max_rail_speed,
        "max_rail_acceleration": max_rail_acceleration,
        "max_rail_force": max_rail_force,
        "max_yaw_force": max_yaw_force,
        "max_rail_position": max_rail_position,
        "max_link5_drift": max_link5_drift,
        "contacts": [{"pair": list(pair), "time": time} for pair, time in contacts.items()],
        "trajectory": trace,
    }
