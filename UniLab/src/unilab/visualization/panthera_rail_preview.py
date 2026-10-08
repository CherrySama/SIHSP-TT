"""Native MuJoCo rail-preview adapter; no manager-environment private access.

The reusable alignment logic lives in the optional SIHSP-TT core. This adapter
owns a separately loaded native viewer model, physical force application and
the simulation-only model feed-forward. Real rail drives need their own adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from rail_alignment import (
    RailAlignmentCfg,
    RailAlignmentController,
    WaistPreparationCfg,
    WaistPreparationController,
)

from unilab.visualization.panthera_height_preparation import PantheraHeightPreparation

if TYPE_CHECKING:
    from ball_prediction import BallPrediction


def load_rail_preview_model(scene_path: str | Path) -> mujoco.MjModel:
    """Compile the native preparation model with a rigid zero-angle link5.

    Cold path only. Removing the unused coordinate prevents soft-constraint
    drift without changing the existing six-axis manager action contract.
    """
    spec = mujoco.MjSpec.from_file(str(scene_path))
    original = spec.compile()
    joint = original.joint("joint5")
    actuator = original.actuator("joint5_position")
    for key in spec.keys:
        key.qpos = np.delete(np.asarray(key.qpos), int(joint.qposadr[0]))
        if len(key.qvel):
            key.qvel = np.delete(np.asarray(key.qvel), int(joint.dofadr[0]))
        key.ctrl = np.delete(np.asarray(key.ctrl), actuator.id)
    spec.delete(spec.equality("joint5_lock"))
    spec.delete(spec.actuator("joint5_position"))
    spec.delete(spec.joint("joint5"))
    return spec.compile()


class PantheraRailPreview:
    """Connect the independent controller to the preview's measured state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        waist_preparation: WaistPreparationCfg | None = None,
        height_preparation: bool = False,
    ) -> None:
        if height_preparation and waist_preparation is None:
            raise ValueError("Height preparation requires the facing waist preparation")
        self.model = model
        self.data = data
        self.controller = RailAlignmentController(
            RailAlignmentCfg(preparation_margin=0.15) if waist_preparation is not None else None
        )
        rail = model.joint("rail_joint")
        self.qpos = int(rail.qposadr[0])
        self.dof = int(rail.dofadr[0])
        self.actuator = model.actuator("rail_force").id
        self.paddle = model.body("paddle_frame").id
        self.rail_lock = model.equality("rail_lock").id
        self.waist: WaistPreparationController | None = None
        self.height: PantheraHeightPreparation | None = None
        if waist_preparation is not None:
            reference = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, reference, model.key("home").id)
            self.arm_qpos = np.asarray(
                [int(model.joint(f"joint{i}").qposadr[0]) for i in (1, 2, 3, 4, 6)]
            )
            self.arm_dof = np.asarray(
                [int(model.joint(f"joint{i}").dofadr[0]) for i in (1, 2, 3, 4, 6)]
            )
            self.arm_actuators = np.asarray(
                [model.actuator(f"joint{i}_position").id for i in (1, 2, 3, 4, 6)]
            )
            self.home_arm = reference.qpos[self.arm_qpos].copy()
            yaw_actuator = self.arm_actuators[0]
            kp = float(model.actuator_gainprm[yaw_actuator, 0])
            kd = -float(model.actuator_biasprm[yaw_actuator, 2])
            if kp <= 0 or kd < 0:
                raise ValueError("Waist preparation requires a positive position PD gain")
            self.yaw_velocity_feedforward = kd / kp
            self.yaw_kp = kp
            self.arm_kp = model.actuator_gainprm[self.arm_actuators[:4], 0].copy()
            self.arm_kd = -model.actuator_biasprm[self.arm_actuators[:4], 2].copy()
            if np.any(self.arm_kp <= 0) or np.any(self.arm_kd < 0):
                raise ValueError("Preparation requires positive position PD gains")
            self.inertia = np.empty((model.nv, model.nv))
            self.waist = WaistPreparationController(float(self.home_arm[0]), waist_preparation)
            self.final_paddles: dict[int, np.ndarray] = {}
            lower, upper = model.joint("joint1").range
            for side in (-1, 1):
                angle = self.home_arm[0] + side * waist_preparation.angle
                if not lower <= angle <= upper:
                    raise ValueError("Preparation yaw exceeds joint1 range")
                reference.qpos[self.arm_qpos[0]] = angle
                mujoco.mj_forward(model, reference)
                point = reference.xpos[self.paddle].copy()
                point[1] -= reference.qpos[self.qpos]
                self.final_paddles[side] = point
            if not np.isclose(self.final_paddles[-1][0], self.final_paddles[1][0]):
                raise ValueError("Left/right preparation require different prediction sections")
            if height_preparation:
                self.height = PantheraHeightPreparation(model, yaw_angle=waist_preparation.angle)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint5") >= 0:
            raise ValueError("Rail preparation requires link5 to be a fixed connection")
        self.last_force = 0.0
        self.first_ready_time: float | None = None
        if not np.allclose(rail.range, [self.controller.cfg.lower, self.controller.cfg.upper]):
            raise ValueError("Native rail range differs from the alignment controller limits")
        self.reset()

    def reset(self) -> None:
        self.data.eq_active[self.rail_lock] = 0
        self.data.ctrl[self.actuator] = 0.0
        self.controller.reset()
        self.last_force = 0.0
        self.first_ready_time = None
        self.settled_since: float | None = None
        self.preparation_prediction_valid = False
        self.last_prediction_time: float | None = None
        if self.waist is not None:
            self.waist.reset()
            self.data.ctrl[self.arm_actuators] = self.home_arm
        if self.height is not None:
            self.height.reset()
        mujoco.mj_forward(self.model, self.data)
        self.initial_paddle_y = float(self.paddle_position[1])

    @property
    def paddle_position(self) -> np.ndarray:
        return self.data.xpos[self.paddle].copy()

    @property
    def candidate_x(self) -> float:
        """Use the final preparation section before starting both drives."""
        if self.waist is None:
            return float(self.paddle_position[0])
        return float(self.final_paddles[-1][0])

    @property
    def goal_paddle_position(self) -> np.ndarray:
        """Final yaw geometry is for rail planning, never readiness feedback."""
        if self.waist is None or self.waist.target is None:
            return self.paddle_position
        side = 1 if self.waist.target > self.waist.home_angle else -1
        point = (
            self.height.goal_paddle.copy()
            if self.height is not None and self.height.goal_paddle is not None
            else self.final_paddles[side].copy()
        )
        point[1] += self.position
        return point

    @property
    def target_arm(self) -> np.ndarray:
        """Final arm target, used only with measured-state readiness checks."""
        target = self.home_arm.copy()
        if self.waist is not None and self.waist.target is not None:
            target[0] = self.waist.target
        if self.height is not None:
            target[1:4] = self.height.target_arm
        return target

    @property
    def position(self) -> float:
        return float(self.data.qpos[self.qpos])

    @property
    def velocity(self) -> float:
        return float(self.data.qvel[self.dof])

    @property
    def ready(self) -> bool:
        if not self._ready_conditions():
            return False
        return bool(
            self.waist is None
            or (
                self.settled_since is not None
                and self.data.time - self.settled_since >= self.waist.cfg.settle_duration - 1e-9
            )
        )

    def _ready_conditions(self) -> bool:
        if self.waist is not None:
            if (
                not self.preparation_prediction_valid
                or self.last_prediction_time is None
                or self.data.time - self.last_prediction_time > self.waist.cfg.prediction_max_age
            ):
                return False
            if not self.waist.ready(
                position=float(self.data.qpos[self.arm_qpos[0]]),
                velocity=float(self.data.qvel[self.arm_dof[0]]),
            ):
                return False
            target = self.target_arm
            if (
                np.max(np.abs(self.data.qpos[self.arm_qpos] - target))
                > self.waist.cfg.position_tolerance
                or np.max(np.abs(self.data.qvel[self.arm_dof])) > self.waist.cfg.settled_velocity
            ):
                return False
            normal = self.data.xmat[self.paddle].reshape(3, 3)[:, 2]
            if abs(normal[0]) < np.cos(np.deg2rad(8.0)):
                return False
            if self.height is not None and (
                self.height.target_height is None
                or abs(self.paddle_position[2] - self.height.target_height)
                > self.height.cfg.position_tolerance
            ):
                return False
        return bool(
            self.controller.ready(
                now=float(self.data.time),
                paddle_y=float(self.paddle_position[1]),
                velocity=self.velocity,
            )
        )

    def observe(self, prediction: BallPrediction) -> None:
        point = prediction.candidate
        usable = prediction.valid and point is not None and point.timestamp > self.data.time
        if self.waist is not None and point is not None:
            usable = usable and bool(np.isclose(point.position[0], self.candidate_x, atol=1e-6))
        if usable and point is not None:
            if self.waist is not None:
                self.waist.observe(ball_y=point.position[1], initial_paddle_y=self.initial_paddle_y)
                side = 1 if self.waist.target > self.waist.home_angle else -1
                if self.height is not None and not self.height.observe(
                    height=point.position[2], side=side
                ):
                    self.preparation_prediction_valid = False
                    self.controller.reset()
                    self.settled_since = None
                    if self.height.target is None:
                        self.waist.reset()
                    return
                self.preparation_prediction_valid = True
                self.last_prediction_time = prediction.source_timestamp
            self.controller.observe(
                rail_position=self.position,
                paddle_y=float(self.goal_paddle_position[1]),
                ball_y=point.position[1],
                arrival_time=point.timestamp,
            )
        elif self.waist is not None and prediction.reason not in {
            "contact_transition",
            "invalid_measurement",
            "duplicate_measurement",
            "out_of_order_measurement",
        }:
            # A contradicted ball path must not be declared ready from an old
            # target. Brief contact/measurement gaps retain it for at most
            # prediction_max_age; a fresh valid observation may resume it.
            self.preparation_prediction_valid = False
            self.controller.reset()
            self.settled_since = None
        # A brief table-contact transition preserves the last future target.
        # Expired targets cause physical braking in the independent controller.
        self._update_settling()

    def _update_settling(self) -> None:
        if self._ready_conditions():
            if self.settled_since is None:
                self.settled_since = float(self.data.time)
        else:
            self.settled_since = None
        if self.ready and self.first_ready_time is None:
            self.first_ready_time = float(self.data.time)

    def apply_force(self) -> None:
        dt = float(self.model.opt.timestep)
        if self.waist is not None:
            self._update_settling()
            previous_velocity = self.waist.velocity
            position = self.waist.advance(dt)
            self.yaw_reference_acceleration = (self.waist.velocity - previous_velocity) / dt
            # A native position servo uses kp*(ctrl-q)-kd*qvel. Offset ctrl
            # by kd/kp*desired_velocity to supply the PD velocity target,
            # without changing the physical state, model gains or force limit.
            target = position + self.yaw_velocity_feedforward * self.waist.velocity
            lower, upper = self.model.actuator_ctrlrange[self.arm_actuators[0]]
            self.data.ctrl[self.arm_actuators[0]] = np.clip(target, lower, upper)
            if self.height is not None:
                reference = self.home_arm[:4].copy()
                reference[0] = position
                reference[1:4] = self.height.advance(dt)
                velocity = np.r_[self.waist.velocity, self.height.velocity * self.height.direction]
                self.arm_reference_acceleration = np.r_[
                    self.yaw_reference_acceleration,
                    self.height.acceleration * self.height.direction,
                ]
                controls = reference + self.arm_kd / self.arm_kp * velocity
                bounds = self.model.actuator_ctrlrange[self.arm_actuators[:4]]
                self.data.ctrl[self.arm_actuators[:4]] = np.clip(
                    controls, bounds[:, 0], bounds[:, 1]
                )
        requested_acceleration = self.controller.acceleration(
            now=float(self.data.time), position=self.position, velocity=self.velocity, dt=dt
        )
        # Leave 1 mm/s of margin for the implicit integrator and coupled servos.
        # The checked limit remains 1.5 m/s on measured, not commanded, velocity.
        speed = self.controller.cfg.max_velocity - 0.001
        next_velocity = float(np.clip(self.velocity + requested_acceleration * dt, -speed, speed))
        requested_acceleration = float(
            np.clip(
                (next_velocity - self.velocity) / dt,
                -self.controller.cfg.max_acceleration,
                self.controller.cfg.max_acceleration,
            )
        )
        # Native model feed-forward: evaluate the instantaneous acceleration
        # response to 0 N and 1 N without integrating or changing qpos/qvel.
        # This includes home servos, rail damping and rigid link5 inertia.
        # A real drive replaces this with its calibrated inner velocity loop.
        self.data.ctrl[self.actuator] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.waist is not None:
            # The desired rail/yaw accelerations are simultaneous. Supplying
            # M*a plus bias/passive compensation avoids treating the moving
            # carriage as an independent stationary yaw base. Other joints
            # retain their home servos; these feed-forwards are native only.
            mujoco.mj_fullM(self.model, self.data, self.inertia)
            if self.height is None:
                yaw_dof = self.arm_dof[0]
                torque = (
                    self.inertia[yaw_dof, yaw_dof] * self.yaw_reference_acceleration
                    + self.inertia[yaw_dof, self.dof] * requested_acceleration
                    + self.data.qfrc_bias[yaw_dof]
                    - self.data.qfrc_passive[yaw_dof]
                )
                lower, upper = self.model.actuator_ctrlrange[self.arm_actuators[0]]
                self.data.ctrl[self.arm_actuators[0]] = np.clip(
                    self.data.ctrl[self.arm_actuators[0]] + torque / self.yaw_kp, lower, upper
                )
            else:
                # All four moving arm joints share the rail/arm acceleration
                # feed-forward. Live qpos/qvel are never assigned targets.
                dofs = self.arm_dof[:4]
                torque = (
                    self.inertia[np.ix_(dofs, dofs)] @ self.arm_reference_acceleration
                    + self.inertia[dofs, self.dof] * requested_acceleration
                    + self.data.qfrc_bias[dofs]
                    - self.data.qfrc_passive[dofs]
                )
                bounds = self.model.actuator_ctrlrange[self.arm_actuators[:4]]
                self.data.ctrl[self.arm_actuators[:4]] = np.clip(
                    self.data.ctrl[self.arm_actuators[:4]] + torque / self.arm_kp,
                    bounds[:, 0],
                    bounds[:, 1],
                )
            mujoco.mj_forward(self.model, self.data)
        base_acceleration = float(self.data.qacc[self.dof])
        self.data.ctrl[self.actuator] = 1.0
        mujoco.mj_forward(self.model, self.data)
        response = float(self.data.qacc[self.dof]) - base_acceleration
        if not np.isfinite(response) or response <= 0:
            raise RuntimeError("Rail drive has no positive finite acceleration response")
        force = (requested_acceleration - base_acceleration) / response
        limit = self.controller.cfg.max_force
        self.last_force = float(np.clip(force, -limit, limit))
        self.data.ctrl[self.actuator] = self.last_force


def draw_rail_alignment(scene: mujoco.MjvScene, rail: PantheraRailPreview) -> None:
    """Append a paddle marker and lateral-only target; neither has collisions."""
    paddle = rail.paddle_position
    points = [(paddle, [0.2, 0.8, 1.0, 1.0])]
    if rail.waist is not None and rail.waist.target is not None:
        points.append((rail.goal_paddle_position, [0.9, 0.2, 0.9, 1.0]))
    target = rail.controller.target
    if target is not None and rail.data.time < target.arrival_time:
        lateral = rail.goal_paddle_position
        lateral[1] = target.ball_y
        points.append((lateral, [1.0, 0.9, 0.1, 1.0]))
        if scene.ngeom < scene.maxgeom and np.linalg.norm(lateral - paddle) > 1e-6:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                paddle,
                np.eye(3).ravel(),
                np.asarray([1.0, 0.9, 0.1, 1.0], dtype=np.float32),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, paddle, lateral)
            scene.ngeom += 1
    for point, color in points:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.012),
            point,
            np.eye(3).ravel(),
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1
