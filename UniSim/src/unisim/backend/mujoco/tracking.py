"""Adapter-owned scalar-joint kinematics and inverse-dynamics tracking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np

from unisim.backend.base import TrackingControlView


def _names(names: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(names)
    if not result or any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


def _scalar_joint_ids(model: Any, names: tuple[str, ...]) -> np.ndarray:
    ids = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"unknown joint {name!r}")
        if int(model.jnt_type[joint]) not in (
            int(mujoco.mjtJoint.mjJNT_SLIDE),
            int(mujoco.mjtJoint.mjJNT_HINGE),
        ):
            raise NotImplementedError(f"joint {name!r} must be scalar hinge or slide")
        ids.append(joint)
    return np.asarray(ids, dtype=np.intp)


def _require_static_model(backend: Any) -> None:
    backend._require_entity_healthy()
    if backend._fixed_variant_build is not None or backend._tracking_model_changed:
        raise NotImplementedError(
            "MuJoCo tracking/FK currently requires one unrandomized construction model"
        )


def evaluate_body_poses(
    backend: Any,
    joint_names: Sequence[str],
    configurations: np.ndarray,
    body_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    _require_static_model(backend)
    model = backend._model
    joints = _scalar_joint_ids(model, _names(joint_names, "joint_names"))
    bodies = []
    for name in _names(body_names, "body_names"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f"unknown body {name!r}")
        bodies.append(body)
    values = np.asarray(configurations)
    if values.ndim != 2 or values.shape[1] != len(joints) or values.dtype.kind not in "fi":
        raise ValueError("configurations must be a real (nconfig, njoint) array")
    if not np.isfinite(values).all():
        raise ValueError("configurations contains NaN or Inf")
    scratch = mujoco.MjData(model)
    positions = np.empty((len(values), len(bodies), 3), dtype=np.float64)
    quaternions = np.empty((len(values), len(bodies), 4), dtype=np.float64)
    qpos_ids = np.asarray(model.jnt_qposadr[joints], dtype=np.intp)
    for index, configuration in enumerate(values):
        scratch.qpos[:] = model.qpos0
        scratch.qpos[qpos_ids] = configuration
        mujoco.mj_kinematics(model, scratch)
        positions[index] = scratch.xpos[bodies]
        quaternions[index] = scratch.xquat[bodies]
    return positions, quaternions


def bind_tracking_control(
    backend: Any, joint_names: Sequence[str], actuator_names: Sequence[str]
) -> TrackingControlView:
    _require_static_model(backend)
    model = backend._model
    names = _names(joint_names, "joint_names")
    actuators = _names(actuator_names, "actuator_names")
    if len(names) != len(actuators):
        raise ValueError("tracking requires one actuator per joint")
    joints = _scalar_joint_ids(model, names)
    actuator_ids = []
    kp, kd = [], []
    for joint, name in zip(joints, actuators, strict=True):
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator < 0:
            raise ValueError(f"unknown actuator {name!r}")
        if (
            int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_JOINT)
            or int(model.actuator_trnid[actuator, 0]) != joint
            or not np.array_equal(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])
            or int(model.actuator_dyntype[actuator]) != int(mujoco.mjtDyn.mjDYN_NONE)
            or int(model.actuator_gaintype[actuator]) != int(mujoco.mjtGain.mjGAIN_FIXED)
        ):
            raise NotImplementedError(f"actuator {name!r} requires direct unit-gear transmission")
        gain = np.asarray(model.actuator_gainprm[actuator])
        bias = np.asarray(model.actuator_biasprm[actuator])
        bias_type = int(model.actuator_biastype[actuator])
        if bias_type == int(mujoco.mjtBias.mjBIAS_NONE) and gain[0] == 1:
            kp.append(0.0)
            kd.append(0.0)
        elif (
            bias_type == int(mujoco.mjtBias.mjBIAS_AFFINE)
            and gain[0] > 0
            and np.isfinite(gain[0])
            and bias[0] == 0
            and bias[1] == -gain[0]
            and bias[2] <= 0
            and np.isfinite(bias[2])
            and np.all(bias[3:] == 0)
        ):
            kp.append(float(gain[0]))
            kd.append(float(-bias[2]))
        else:
            raise NotImplementedError(f"actuator {name!r} is not a motor or linear position servo")
        if bool(model.jnt_actgravcomp[joint]):
            raise NotImplementedError("actuator gravity compensation is unsupported")
        if (
            np.count_nonzero(
                (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
                & (model.actuator_trnid[:, 0] == joint)
            )
            != 1
        ):
            raise NotImplementedError("tracking joints require a single direct actuator")
        actuator_ids.append(actuator)
    act_ids = np.asarray(actuator_ids, dtype=np.intp)
    qpos_ids = np.asarray(model.jnt_qposadr[joints], dtype=np.intp)
    dof_ids = np.asarray(model.jnt_dofadr[joints], dtype=np.intp)
    kp_array, kd_array = np.asarray(kp), np.asarray(kd)
    servo = kp_array > 0
    lower = np.where(
        model.actuator_ctrllimited[act_ids], model.actuator_ctrlrange[act_ids, 0], -np.inf
    )
    upper = np.where(
        model.actuator_ctrllimited[act_ids], model.actuator_ctrlrange[act_ids, 1], np.inf
    )
    force_lower = np.where(
        model.actuator_forcelimited[act_ids], model.actuator_forcerange[act_ids, 0], -np.inf
    )
    force_upper = np.where(
        model.actuator_forcelimited[act_ids], model.actuator_forcerange[act_ids, 1], np.inf
    )
    force_lower = np.maximum(
        force_lower,
        np.where(model.jnt_actfrclimited[joints], model.jnt_actfrcrange[joints, 0], -np.inf),
    )
    force_upper = np.minimum(
        force_upper,
        np.where(model.jnt_actfrclimited[joints], model.jnt_actfrcrange[joints, 1], np.inf),
    )
    joint_lower = np.where(model.jnt_limited[joints], model.jnt_range[joints, 0], -np.inf)
    joint_upper = np.where(model.jnt_limited[joints], model.jnt_range[joints, 1], np.inf)
    if np.any(lower >= upper) or np.any(force_lower >= force_upper):
        raise ValueError("tracking control/force limits must have lower < upper")
    scratch = mujoco.MjData(model)
    mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    full_acceleration = np.zeros(model.nv, dtype=np.float64)
    motors = np.flatnonzero(~servo)
    motor_dofs = dof_ids[motors]
    unit_motor_forces = np.eye(model.nv, dtype=np.float64)[:, motor_dofs]
    if np.any(
        np.maximum(lower[motors], force_lower[motors])
        > np.minimum(upper[motors], force_upper[motors])
    ):
        raise ValueError("motor control and physical force limits have no feasible intersection")

    def compute(
        positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray
    ) -> np.ndarray:
        _require_static_model(backend)
        if backend._pool is None:
            raise RuntimeError("tracking compute requires a materialized backend")
        if np.any(positions < joint_lower) or np.any(positions > joint_upper):
            raise ValueError("tracking positions exceed joint limits")
        output = np.empty((backend._num_envs, len(act_ids)), dtype=np.float64)
        for env in range(backend._num_envs):
            scratch.qpos[:] = backend._qpos_view[env]
            scratch.qvel[:] = backend._qvel_view[env]
            scratch.ctrl[:] = backend._ctrl_view[env]
            if model.na:
                scratch.act[:] = backend._act_view[env]
            scratch.qfrc_applied[:] = backend._entity_qfrc_view[env]
            scratch.xfrc_applied[:] = backend._xfrc_view[env].reshape(model.nbody, 6)
            mujoco.mj_forward(model, scratch)
            mujoco.mj_fullM(model, scratch, mass)
            full_acceleration.fill(0)
            full_acceleration[dof_ids] = accelerations[env]
            compensation = (mass @ full_acceleration + scratch.qfrc_bias - scratch.qfrc_passive)[
                dof_ids
            ]
            # Clamp total commanded force, including servo feedback, before
            # converting back to position-actuator control coordinates.
            force = compensation.copy()
            force[servo] += kp_array[servo] * (
                positions[env, servo] - scratch.qpos[qpos_ids[servo]]
            ) + kd_array[servo] * (velocities[env, servo] - scratch.qvel[dof_ids[servo]])
            force = np.clip(force, force_lower, force_upper)
            control = force.copy()
            control[servo] = (
                scratch.qpos[qpos_ids[servo]]
                + kd_array[servo] / kp_array[servo] * scratch.qvel[dof_ids[servo]]
                + force[servo] / kp_array[servo]
            )
            control = np.clip(control, lower, upper)
            if motors.size:
                # Preserve servo PD feedback, then cancel its coupled motor
                # acceleration residual. This is the multi-motor equivalent
                # of a 0 N / 1 N response probe, without additional forwards.
                applied_force = control.copy()
                applied_force[servo] = (
                    kp_array[servo] * (control[servo] - scratch.qpos[qpos_ids[servo]])
                    - kd_array[servo] * scratch.qvel[dof_ids[servo]]
                )
                applied_force = np.clip(applied_force, force_lower, force_upper)
                smooth_force = scratch.qfrc_smooth.copy()
                old_force = np.clip(scratch.actuator_force[act_ids], force_lower, force_upper)
                smooth_force[dof_ids] += applied_force - old_force
                response = np.linalg.solve(mass, unit_motor_forces)
                actual_acceleration = response.T @ smooth_force
                correction = np.linalg.solve(
                    response[motor_dofs], accelerations[env, motors] - actual_acceleration
                )
                control[motors] = np.clip(
                    applied_force[motors] + correction,
                    np.maximum(lower[motors], force_lower[motors]),
                    np.minimum(upper[motors], force_upper[motors]),
                )
            output[env] = control
        return output

    return TrackingControlView(
        joint_names=names,
        actuator_names=actuators,
        num_envs=backend._num_envs,
        lower=tuple(float(x) for x in lower),
        upper=tuple(float(x) for x in upper),
        kp=tuple(kp),
        kd=tuple(kd),
        _compute=compute,
    )
