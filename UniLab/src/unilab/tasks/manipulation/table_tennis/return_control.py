"""Manager-owned preparation, bounded residual swing and return episode state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from unilab.managers import ActionTerm, ActionTermCfg
from unilab.tasks.manipulation.table_tennis.prepared_scene import ARM_CONTACT_NAMES
from unilab.tasks.manipulation.table_tennis.return_events import ReturnEpisode
from unilab.tasks.manipulation.table_tennis.return_guidance import (
    ReturnGuidanceCfg,
    flight_window,
    guidance_scores,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.tasks.manipulation.table_tennis.return_shaping import ReturnShapingCfg

JOINT_NAMES = ("rail_joint", "joint1", "joint2", "joint3", "joint4", "joint6")
ACTUATOR_NAMES = (
    "rail_force",
    "joint1_position",
    "joint2_position",
    "joint3_position",
    "joint4_position",
    "joint6_position",
)
CONTACT_NAMES = tuple(
    f"return_contact_{name}" for name in ("paddle", "table", "net", "robot", "floor")
)


def _normal(quaternion: np.ndarray) -> np.ndarray:
    """World direction of the frame's local +Z axis, with wxyz quaternions."""
    w, x, y, z = quaternion.T
    return np.column_stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)))


@dataclass(kw_only=True)
class PantheraReturnActionCfg(ActionTermCfg):
    """Bounded yaw and paddle-roll velocities after measured preparation."""

    yaw_velocity: float = 3.0
    tilt_velocity: float = 3.0
    yaw_acceleration: float = 12.0
    tilt_acceleration: float = 20.0
    residual_control: bool = True
    rail_offset: float = 0.03
    arm_offset: float = 0.03
    rail_adjustment_velocity: float = 0.2
    rail_adjustment_acceleration: float = 2.0
    arm_adjustment_velocity: float = 0.5
    arm_adjustment_acceleration: float = 4.0
    handoff_duration: float = 0.05
    guidance: ReturnGuidanceCfg = field(default_factory=ReturnGuidanceCfg)

    def build(self, env: ManagerBasedRlEnv) -> PantheraReturnAction:
        return PantheraReturnAction(self, env)


class PantheraReturnAction(ActionTerm):
    """Prepare deterministically, learn the strike, then track the actual return."""

    requires_substep_state_feedback: ClassVar[bool] = True
    _entity: Entity

    def __init__(self, cfg: PantheraReturnActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        for value in (
            cfg.yaw_velocity,
            cfg.tilt_velocity,
            cfg.yaw_acceleration,
            cfg.tilt_acceleration,
            cfg.rail_offset,
            cfg.arm_offset,
            cfg.rail_adjustment_velocity,
            cfg.rail_adjustment_acceleration,
            cfg.arm_adjustment_velocity,
            cfg.arm_adjustment_acceleration,
            cfg.handoff_duration,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Return action velocity and acceleration limits must be positive")
        if cfg.clip is not None:
            raise ValueError("Return action clips normalized commands internally")
        if not isinstance(cfg.residual_control, bool):
            raise ValueError("residual_control must be boolean")
        self._return_cfg = cfg
        # Optional prediction/control core is imported only when this task is built.
        from rail_alignment import (
            RailAlignmentCfg,
            RailAlignmentController,
            WaistPreparationCfg,
            WaistPreparationController,
        )

        from unilab.tasks.manipulation.table_tennis.prediction_adapter import PantheraBallPredictor

        self.backend = env.physics_backend
        self._tracking = self.backend.bind_tracking_control(JOINT_NAMES, ACTUATOR_NAMES)
        self._qpos_ids = self.backend.get_joint_state_qpos_indices(JOINT_NAMES)
        self._qvel_ids = self.backend.get_joint_state_qvel_indices(JOINT_NAMES)
        ball_layout = self.backend.get_root_state_layout("ball")
        self._ball_qpos = np.asarray(ball_layout.qpos_indices[:3])
        self._ball_qvel = np.asarray(ball_layout.qvel_indices[:3])
        self._paddle_id = self.backend.get_body_ids(("paddle_frame",))
        self._arm_ids, _ = self._entity.find_joints(JOINT_NAMES[1:], preserve_order=True)
        self._rail_ids, _ = self._entity.find_joints(JOINT_NAMES[:1], preserve_order=True)
        self._contacts = self.backend.bind_sensor_data(CONTACT_NAMES)
        self._arm_contacts = self.backend.bind_sensor_data(ARM_CONTACT_NAMES)
        # Integrate reference trajectories in float64 even when public asset
        # metadata is float32; small substep increments must not round to zero.
        self.home = self.backend.get_keyframe_qpos("home")[self._qpos_ids].astype(np.float64)
        home_pose, _ = self.backend.evaluate_body_poses(
            JOINT_NAMES, self.home[None], ("paddle_frame",)
        )
        self._home_paddle = home_pose[0, 0]
        limits = self.backend.get_joint_range(names=JOINT_NAMES)
        assert limits is not None
        # Public metadata can be float32 while tracking validates native
        # float64 bounds: 2.4 rounds to 2.400000095. Keep references one ULP
        # inside the metadata bounds instead of relaxing physical validation.
        self._limits = np.column_stack(
            (np.nextafter(limits[:, 0], np.inf), np.nextafter(limits[:, 1], -np.inf))
        ).astype(np.float64)
        self._phases = np.linspace(-0.1, 0.8, 301)
        self._height_direction = np.array([0.5, 1.0, -0.5])
        self._points: dict[int, np.ndarray] = {}
        for side in (-1, 1):
            configurations = np.tile(self.home, (len(self._phases), 1))
            configurations[:, 1] += side * np.pi / 2
            configurations[:, 2:5] += self._phases[:, None] * self._height_direction
            positions, quaternions = self.backend.evaluate_body_poses(
                JOINT_NAMES, configurations, ("paddle_frame",)
            )
            points = positions[:, 0]
            normals = _normal(quaternions[:, 0])
            if (
                np.any(np.diff(points[:, 2]) <= 0)
                or np.ptp(points[:, 0]) > 1e-8
                or not np.allclose(normals, normals[0], atol=1e-8)
            ):
                raise ValueError("Preparation lookup must preserve hit section and paddle normal")
            self._points[side] = points
        self._heights = self._points[-1][:, 2]
        if not np.allclose(self._heights, self._points[1][:, 2], atol=1e-8):
            raise ValueError("Preparation left/right height branches differ")
        # Cold FK supplies an envelope; reward scoring does not claim that
        # every point of its bounding box has a realizable inverse kinematic pose.
        configurations = []
        for phase in np.linspace(self._phases[0], self._phases[-1], 10):
            for yaw in np.linspace(float(self._limits[1, 0]), float(self._limits[1, 1]), 17):
                base = self.home.copy()
                base[1] = yaw
                base[2:5] += phase * self._height_direction
                configurations.append(base)
        poses, _ = self.backend.evaluate_body_poses(
            JOINT_NAMES, np.asarray(configurations), ("paddle_frame",)
        )
        from itertools import product

        offsets = np.asarray(list(product((-cfg.arm_offset, cfg.arm_offset), repeat=3)))
        adjusted = np.repeat(np.asarray(configurations), len(offsets), axis=0)
        adjusted[:, 2:5] += np.tile(offsets, (len(configurations), 1))
        adjusted_poses, _ = self.backend.evaluate_body_poses(
            JOINT_NAMES, adjusted, ("paddle_frame",)
        )
        self._workspace_lower = adjusted_poses[:, 0].min(axis=0) - 0.01
        self._workspace_upper = adjusted_poses[:, 0].max(axis=0) + 0.01
        self._height_workspace_padding = (
            float(np.max(np.abs(adjusted_poses[:, 0, 2] - np.repeat(poses[:, 0, 2], len(offsets)))))
            + 0.01
        )
        self.candidate_x = float(self._points[-1][0, 0])
        self._predictors = [PantheraBallPredictor() for _ in range(env.num_envs)]
        for predictor in self._predictors:
            from dataclasses import replace

            predictor.cfg = replace(predictor.cfg, candidate_x=self.candidate_x)
        self._waist_cfg = WaistPreparationCfg()
        self._waists = [
            WaistPreparationController(self.home[1], self._waist_cfg) for _ in range(env.num_envs)
        ]
        self._rails = [
            RailAlignmentController(RailAlignmentCfg(preparation_margin=0.15))
            for _ in range(env.num_envs)
        ]
        self.episodes = [ReturnEpisode() for _ in range(env.num_envs)]
        self._raw_action = np.zeros((env.num_envs, 6 if cfg.residual_control else 2))
        self._previous_action = np.zeros_like(self._raw_action)
        self._command = np.zeros_like(self._raw_action)
        self._executed_command = np.zeros_like(self._raw_action)
        self._reference = np.tile(self.home, (env.num_envs, 1))
        self._preparation_base = self._reference.copy()
        self._offset_scales = np.array(
            [cfg.rail_offset, cfg.arm_offset, cfg.arm_offset, cfg.arm_offset]
        )
        self._peak_offset = np.zeros((env.num_envs, 4))
        self._peak_arm_contacts = np.zeros((env.num_envs, 3))
        self._reference_velocity = np.zeros_like(self._reference)
        self._reference_acceleration = np.zeros_like(self._reference)
        self._height_phase = np.zeros(env.num_envs)
        self._height_velocity = np.zeros(env.num_envs)
        self._height_target = np.full(env.num_envs, np.nan)
        self._height_z = np.full(env.num_envs, np.nan)
        self._last_prediction = np.full(env.num_envs, -np.inf)
        self._prediction_valid = np.zeros(env.num_envs, dtype=bool)
        self._arrival = np.zeros(env.num_envs)
        self._predicted_velocity = np.zeros((env.num_envs, 3))
        self._settled_since = np.full(env.num_envs, np.nan)
        self.ready = np.zeros(env.num_envs, dtype=bool)
        self.side = np.full(env.num_envs, -1, dtype=np.int32)
        self.elapsed = np.zeros(env.num_envs)
        self.prepared_time = np.full(env.num_envs, np.nan)
        self._peak_speed = np.zeros_like(self._reference)
        self._peak_force = np.zeros(env.num_envs)
        self._excessive_steps = np.zeros(env.num_envs, dtype=np.int32)
        self._launch_position = np.zeros((env.num_envs, 3))
        self._launch_velocity = np.zeros((env.num_envs, 3))
        self._predicted_position = np.zeros((env.num_envs, 3))
        self._guidance_windows = [np.empty((0, 7)) for _ in range(env.num_envs)]
        self._guidance_timestamp = np.full(env.num_envs, -np.inf)
        self._rebound_forecasts: list[tuple[float, np.ndarray, np.ndarray] | None] = [
            None
        ] * env.num_envs
        self._forecast_timestamp = np.full(env.num_envs, -np.inf)
        self._max_velocity = np.array([1.5, cfg.yaw_velocity, 1.5, 1.5, 1.5, cfg.tilt_velocity])
        self._max_acceleration = np.array(
            [10.0, cfg.yaw_acceleration, 8.0, 8.0, 8.0, cfg.tilt_acceleration]
        )
        self.backend.set_post_step_observer(self._observe_substep)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = np.arange(self._env.num_envs)[slice(None) if env_ids is None else env_ids]
        for i in ids:
            self._waists[i].reset()
            self._rails[i].reset()
            self._predictors[i].reset()
            self.episodes[i].reset()
            self._guidance_windows[i] = np.empty((0, 7))
            self._rebound_forecasts[i] = None
        self._reference[ids] = self.home
        self._preparation_base[ids] = self.home
        self._reference_velocity[ids] = 0
        self._reference_acceleration[ids] = 0
        self._raw_action[ids] = 0
        self._previous_action[ids] = 0
        self._command[ids] = 0
        self._executed_command[ids] = 0
        self._peak_offset[ids] = 0
        self._peak_arm_contacts[ids] = 0
        self._height_phase[ids] = 0
        self._height_velocity[ids] = 0
        self._height_target[ids] = np.nan
        self._height_z[ids] = np.nan
        self._last_prediction[ids] = -np.inf
        self._prediction_valid[ids] = False
        self._settled_since[ids] = np.nan
        self.ready[ids] = False
        self.side[ids] = -1
        self.elapsed[ids] = 0
        self.prepared_time[ids] = np.nan
        self._peak_speed[ids] = 0
        self._peak_force[ids] = 0
        self._excessive_steps[ids] = 0
        self._arrival[ids] = 0
        self._predicted_velocity[ids] = 0
        self._predicted_position[ids] = 0
        self._guidance_timestamp[ids] = -np.inf
        self._forecast_timestamp[ids] = -np.inf

    @property
    def action_dim(self) -> int:
        return int(self._raw_action.shape[1])

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    def process_actions(self, actions: np.ndarray) -> None:
        if actions.shape != self._raw_action.shape or not np.isfinite(actions).all():
            raise ValueError(f"Return actions must be finite (num_envs, {self.action_dim}) values")
        self._previous_action[:] = self._executed_command
        self._raw_action[:] = actions
        np.clip(actions, -1, 1, out=self._command)
        for episode in self.episodes:
            episode.clear_credit()
        self._update_prediction()
        self._update_guidance()

    def apply_actions(self) -> None:
        dt = self._env.physics_dt
        state = self.backend.get_state(("qpos", "qvel"))
        measured = state["qpos"][:, self._qpos_ids]
        velocity = state["qvel"][:, self._qvel_ids]
        paddle, quaternion = self.backend.get_body_pose_w(self._paddle_id)
        paddle = paddle[:, 0]
        normal = _normal(quaternion[:, 0])
        old_velocity = self._reference_velocity.copy()
        for i, episode in enumerate(self.episodes):
            if episode.done:
                self._reference_velocity[i] = 0
                self._reference_acceleration[i] = 0
                continue
            if not self.ready[i]:
                self._check_ready(i, measured[i], velocity[i], paddle[i], normal[i])
            if self.ready[i]:
                flight = episode.outgoing_time is not None
                self._executed_command[i] = 0 if flight else self._command[i]
                # Positive normalized yaw always swings towards the opponent.
                # Positive tilt raises the outgoing face on either preparation side.
                self._advance_swing(i, 1, -self.side[i] * self._executed_command[i, 0], dt)
                self._advance_swing(i, 5, -self.side[i] * self._executed_command[i, 1], dt)
                if self._return_cfg.residual_control:
                    self._advance_adjustments(i, dt, flight)
            else:
                waist = self._waists[i]
                previous = waist.velocity
                self._reference[i, 1] = waist.advance(dt)
                self._reference_velocity[i, 1] = waist.velocity
                self._reference_acceleration[i, 1] = (waist.velocity - previous) / dt
                self._advance_height(i, dt)
            if self.ready[i]:
                error = self._reference[i, 0] - measured[i, 0]
                if self._return_cfg.residual_control:
                    self._reference_acceleration[i, 0] = np.clip(
                        self._reference_acceleration[i, 0]
                        + 100 * error
                        + 20 * (self._reference_velocity[i, 0] - velocity[i, 0]),
                        -10,
                        10,
                    )
                    continue
                acceleration = np.clip(100 * error - 20 * velocity[i, 0], -10, 10)
            else:
                acceleration = self._rails[i].acceleration(
                    now=self.elapsed[i], position=measured[i, 0], velocity=velocity[i, 0], dt=dt
                )
                rail_target = self._rails[i].target
                if rail_target is not None:
                    self._reference[i, 0] = rail_target.position
            next_velocity = np.clip(velocity[i, 0] + acceleration * dt, -1.499, 1.499)
            self._reference_acceleration[i, 0] = np.clip(
                (next_velocity - velocity[i, 0]) / dt, -10, 10
            )
            self._reference_velocity[i, 0] = next_velocity
            if self.ready[i]:
                self._reference_acceleration[i, (1, 5)] = (
                    self._reference_velocity[i, (1, 5)] - old_velocity[i, (1, 5)]
                ) / dt
        controls = self._tracking.compute(
            self._reference, self._reference_velocity, self._reference_acceleration
        )
        self._peak_force[:] = np.maximum(self._peak_force, np.abs(controls[:, 0]))
        self._entity.set_joint_effort_target(controls[:, :1], joint_ids=self._rail_ids)
        self._entity.set_joint_position_target(controls[:, 1:], joint_ids=self._arm_ids)

    def _advance_adjustments(self, i: int, dt: float, flight: bool) -> None:
        cfg = self._return_cfg
        ratio = np.clip((self.elapsed[i] - self.prepared_time[i]) / cfg.handoff_duration, 0, 1)
        blend = ratio * ratio * (3 - 2 * ratio)
        for column, joint in enumerate((0, 2, 3, 4)):
            lower = max(
                self._limits[joint, 0],
                self._preparation_base[i, joint] - self._offset_scales[column],
            )
            upper = min(
                self._limits[joint, 1],
                self._preparation_base[i, joint] + self._offset_scales[column],
            )
            target = np.clip(
                self._preparation_base[i, joint]
                + blend * self._offset_scales[column] * self._command[i, column + 2],
                lower,
                upper,
            )
            speed = cfg.rail_adjustment_velocity if joint == 0 else cfg.arm_adjustment_velocity
            acceleration = (
                cfg.rail_adjustment_acceleration if joint == 0 else cfg.arm_adjustment_acceleration
            )
            goal = (
                0.0 if flight else np.clip(10 * (target - self._reference[i, joint]), -speed, speed)
            )
            self._advance_velocity(i, joint, float(goal), dt, lower, upper, acceleration)
        self._peak_offset[i] = np.maximum(
            self._peak_offset[i],
            np.abs(self._reference[i, (0, 2, 3, 4)] - self._preparation_base[i, (0, 2, 3, 4)]),
        )

    def _advance_height(self, i: int, dt: float) -> None:
        if not np.isfinite(self._height_target[i]):
            return
        error = self._height_target[i] - self._height_phase[i]
        step = 8 * dt
        stopping_speed = np.sqrt(step * step + 16 * abs(error)) - step
        goal = np.copysign(min(1.5, stopping_speed), error)
        previous = self._height_velocity[i]
        self._height_velocity[i] += np.clip(goal - previous, -step, step)
        self._height_phase[i] += self._height_velocity[i] * dt
        self._reference[i, 2:5] = self.home[2:5] + self._height_phase[i] * self._height_direction
        self._reference_velocity[i, 2:5] = self._height_velocity[i] * self._height_direction
        self._reference_acceleration[i, 2:5] = (
            (self._height_velocity[i] - previous) / dt * self._height_direction
        )

    def _advance_swing(self, i: int, joint: int, command: float, dt: float) -> None:
        assert self._limits is not None
        lower, upper = self._limits[joint]
        acceleration = self._max_acceleration[joint]
        self._advance_velocity(
            i, joint, command * self._max_velocity[joint], dt, lower, upper, acceleration
        )

    def _advance_velocity(
        self,
        i: int,
        joint: int,
        goal: float,
        dt: float,
        lower: float,
        upper: float,
        acceleration: float,
    ) -> None:
        step = acceleration * dt
        position = self._reference[i, joint]
        stopping_lower = np.sqrt(step * step + 2 * acceleration * max(0, position - lower)) - step
        stopping_upper = np.sqrt(step * step + 2 * acceleration * max(0, upper - position)) - step
        goal = np.clip(goal, -stopping_lower, stopping_upper)
        previous = self._reference_velocity[i, joint]
        self._reference_velocity[i, joint] += np.clip(
            goal - self._reference_velocity[i, joint], -step, step
        )
        self._reference[i, joint] = np.clip(
            position + self._reference_velocity[i, joint] * dt, lower, upper
        )
        self._reference_acceleration[i, joint] = (
            self._reference_velocity[i, joint] - previous
        ) / dt

    def _update_prediction(self) -> None:
        from ball_state import BallState

        state = self.backend.get_state(("qpos", "qvel"))
        positions = state["qpos"][:, self._ball_qpos]
        velocities = state["qvel"][:, self._ball_qvel]
        for i in range(self._env.num_envs):
            if self.elapsed[i] == 0:
                self._launch_position[i] = positions[i]
                self._launch_velocity[i] = velocities[i]
            if self.episodes[i].hit or self.episodes[i].done:
                continue
            prediction = self._predictors[i].update(
                BallState(float(self.elapsed[i]), tuple(positions[i]), tuple(velocities[i]))
            )
            if prediction.valid and prediction.first_bounce is not None:
                bounce = prediction.first_bounce
                cfg = self._predictors[i].cfg
                velocity = np.asarray(bounce.velocity) * np.array(
                    [cfg.tangential_retention, cfg.tangential_retention, -cfg.normal_restitution]
                )
                self._rebound_forecasts[i] = (
                    bounce.timestamp,
                    np.asarray(bounce.position),
                    velocity,
                )
                self._forecast_timestamp[i] = prediction.source_timestamp
            if self.ready[i]:
                continue
            point = prediction.candidate
            if prediction.valid and point is not None and point.timestamp > self.elapsed[i]:
                self._waists[i].observe(
                    ball_y=point.position[1], initial_paddle_y=float(self._home_paddle[1])
                )
                side = 1 if self._waists[i].target > self.home[1] else -1
                self.side[i] = side
                z = point.position[2]
                if not self._heights[0] <= z <= self._heights[-1]:
                    self._prediction_valid[i] = False
                    self._rails[i].reset()
                    self._settled_since[i] = np.nan
                    continue
                self._height_z[i] = z
                self._height_target[i] = np.interp(z, self._heights, self._phases)
                goal_y = np.interp(z, self._heights, self._points[side][:, 1])
                rail_position = state["qpos"][i, self._qpos_ids[0]]
                self._rails[i].observe(
                    rail_position=rail_position,
                    paddle_y=goal_y + rail_position,
                    ball_y=point.position[1],
                    arrival_time=point.timestamp,
                )
                self._last_prediction[i] = prediction.source_timestamp
                self._prediction_valid[i] = True
                self._arrival[i] = point.timestamp
                self._predicted_position[i] = point.position
                self._predicted_velocity[i] = point.velocity
            elif prediction.reason not in {
                "contact_transition",
                "invalid_measurement",
                "duplicate_measurement",
                "out_of_order_measurement",
            }:
                self._prediction_valid[i] = False
                self._rails[i].reset()
                self._settled_since[i] = np.nan

    def _update_guidance(self) -> None:
        state = self.backend.get_state(("qpos", "qvel"))
        contacts = self._contacts.read()
        for i, episode in enumerate(self.episodes):
            if not self.ready[i] or episode.hit or episode.done:
                continue
            # During finite table contact use the last free-flight prediction;
            # its age is still checked when observations and rewards consume it.
            if np.any(contacts[i] > 0):
                continue
            origin: tuple[float, np.ndarray, np.ndarray] | None
            if episode.receiver_bounced:
                origin = (
                    float(self.elapsed[i]),
                    state["qpos"][i, self._ball_qpos],
                    state["qvel"][i, self._ball_qvel],
                )
                source_timestamp = self.elapsed[i]
            else:
                origin = self._rebound_forecasts[i]
                source_timestamp = self._forecast_timestamp[i]
            if (
                origin is None
                or self.elapsed[i] - source_timestamp > self._return_cfg.guidance.freshness
            ):
                self._guidance_windows[i] = np.empty((0, 7))
                continue
            lower = self._workspace_lower.copy()
            upper = self._workspace_upper.copy()
            lower[1] += self._preparation_base[i, 0] - self._return_cfg.rail_offset
            upper[1] += self._preparation_base[i, 0] + self._return_cfg.rail_offset
            lower[2] = self._height_z[i] - self._height_workspace_padding
            upper[2] = self._height_z[i] + self._height_workspace_padding
            timestamp, position, velocity = origin
            self._guidance_windows[i] = flight_window(
                position,
                velocity,
                timestamp=timestamp,
                now=float(self.elapsed[i]),
                surface=episode.table_z + episode.ball_radius,
                lower=lower,
                upper=upper,
                cfg=self._return_cfg.guidance,
            )
            self._guidance_timestamp[i] = source_timestamp

    def guidance_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position, quaternion = self.backend.get_body_pose_w(self._paddle_id)
        velocity = self.backend.get_body_lin_vel_w(self._paddle_id)[:, 0]
        normal = _normal(quaternion[:, 0])
        values = np.zeros((self._env.num_envs, 2))
        targets = np.zeros((self._env.num_envs, 7))
        valid = np.zeros(self._env.num_envs, dtype=bool)
        age = np.clip(self.elapsed - self._guidance_timestamp, 0, 1)
        for i, episode in enumerate(self.episodes):
            if (
                not self.ready[i]
                or episode.hit
                or episode.done
                or age[i] > self._return_cfg.guidance.freshness
            ):
                continue
            reach, strike, target = guidance_scores(
                self._guidance_windows[i],
                now=float(self.elapsed[i]),
                paddle=position[i, 0],
                normal=normal[i],
                paddle_velocity=velocity[i],
                receiver_bounced=episode.receiver_bounced,
                cfg=self._return_cfg.guidance,
            )
            if target is not None:
                valid[i] = True
                values[i] = (reach, strike)
                targets[i] = target
        return values, targets, valid, age

    def _check_ready(
        self,
        i: int,
        position: np.ndarray,
        velocity: np.ndarray,
        paddle: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        waist = self._waists[i]
        target = self.home.copy()
        if waist.target is not None:
            target[1] = waist.target
        if np.isfinite(self._height_target[i]):
            target[2:5] += self._height_target[i] * self._height_direction
        ready = bool(
            self._prediction_valid[i]
            and self.elapsed[i] - self._last_prediction[i] <= 0.05
            and waist.ready(position=position[1], velocity=velocity[1])
            and np.max(np.abs(position[1:] - target[1:])) <= 0.015
            and np.max(np.abs(velocity[1:])) <= 0.05
            and abs(normal[0]) >= np.cos(np.deg2rad(8))
            and abs(paddle[2] - self._height_z[i]) <= 0.01
            and self._rails[i].ready(now=self.elapsed[i], paddle_y=paddle[1], velocity=velocity[0])
        )
        if not ready:
            self._settled_since[i] = np.nan
            return
        if not np.isfinite(self._settled_since[i]):
            self._settled_since[i] = self.elapsed[i]
        if self.elapsed[i] - self._settled_since[i] >= 0.03 - 1e-9:
            self.ready[i] = True
            self.prepared_time[i] = self.elapsed[i]
            # Start the policy trajectory from the settled measured state.
            # The readiness tolerances bound this handoff; physical state is untouched.
            self._reference[i] = position
            self._preparation_base[i] = position
            self._reference_velocity[i] = 0
            self._reference_acceleration[i] = 0

    def _observe_substep(self, backend: object) -> None:
        del backend
        contact = self._contacts.read()
        arm_contact = self._arm_contacts.read()
        self._peak_arm_contacts[:] = np.maximum(self._peak_arm_contacts, arm_contact)
        state = self.backend.get_state(("qpos", "qvel"))
        positions = state["qpos"][:, self._ball_qpos]
        velocities = state["qvel"][:, self._ball_qvel]
        measured_velocity = state["qvel"][:, self._qvel_ids]
        self.elapsed += self._env.physics_dt
        self._peak_speed[:] = np.maximum(self._peak_speed, np.abs(measured_velocity))
        excessive = np.any(np.abs(measured_velocity) > self._max_velocity + 0.02, axis=1)
        self._excessive_steps[:] = np.where(excessive, self._excessive_steps + 1, 0)
        for i, episode in enumerate(self.episodes):
            if not episode.done and np.any(arm_contact[i] > 0):
                reason = ("arm_self_contact", "arm_table_contact", "arm_floor_contact")[
                    int(np.flatnonzero(arm_contact[i] > 0)[0])
                ]
                episode.fail(reason)
                continue
            labels = {
                name
                for name, value in zip(
                    ("paddle", "table", "net", "robot", "floor"), contact[i], strict=True
                )
                if value > 0
            }
            # Robot subtree includes the paddle; subtract its actual contact count.
            if contact[i, 3] <= contact[i, 0]:
                labels.discard("robot")
            episode.observe(float(self.elapsed[i]), positions[i], velocities[i], labels)
            if not episode.done and self._excessive_steps[i] * self._env.physics_dt >= 0.02:
                episode.fail("joint_speed")
            if self.elapsed[i] >= self._env.max_episode_length_s - 1e-9:
                episode.finish_timeout()

    def episode_report(self, env_idx: int) -> dict[str, object]:
        episode = self.episodes[env_idx]
        return {
            "success": episode.success,
            "failure_reason": episode.failure_reason,
            "side": int(self.side[env_idx]),
            "prepared_time": float(self.prepared_time[env_idx]) if self.ready[env_idx] else None,
            "hit_time": episode.hit_time,
            "hit_position": episode.hit_position,
            "outgoing_time": episode.outgoing_time,
            "outgoing_position": episode.outgoing_position,
            "outgoing_velocity": episode.outgoing_velocity,
            "crossing_time": episode.crossing_time,
            "landing_time": episode.landing_time,
            "landing_position": episode.landing_position,
            "peak_joint_speed": self._peak_speed[env_idx].tolist(),
            "peak_rail_force": float(self._peak_force[env_idx]),
            "peak_reference_offset": self._peak_offset[env_idx].tolist(),
            "peak_arm_contacts": self._peak_arm_contacts[env_idx].tolist(),
            "launch_position": self._launch_position[env_idx].tolist(),
            "launch_velocity": self._launch_velocity[env_idx].tolist(),
            "predicted_hit_position": self._predicted_position[env_idx].tolist(),
        }


def _control(env: ManagerBasedRlEnv) -> PantheraReturnAction:
    from typing import cast

    return cast(PantheraReturnAction, env.action_manager.get_term("return_control"))


def return_observation(env: ManagerBasedRlEnv) -> np.ndarray:
    control = _control(env)
    state = control.backend.get_state(("qpos", "qvel"))
    position, quaternion = control.backend.get_body_pose_w(control._paddle_id)
    paddle = position[:, 0]
    paddle_velocity = control.backend.get_body_lin_vel_w(control._paddle_id)[:, 0]
    normal = _normal(quaternion[:, 0]) * control.side[:, None]
    _, targets, guidance_valid, guidance_age = control.guidance_state()
    flags = np.array(
        [
            (
                control.ready[i],
                episode.receiver_bounced,
                episode.hit,
                episode.crossed_net,
                episode.outgoing_time is not None,
                episode.done,
            )
            for i, episode in enumerate(control.episodes)
        ]
    )
    return np.column_stack(
        (
            (state["qpos"][:, control._qpos_ids] - control.home),
            state["qvel"][:, control._qvel_ids] / control._max_velocity,
            state["qpos"][:, control._ball_qpos] - paddle,
            state["qvel"][:, control._ball_qvel] / 3,
            paddle,
            paddle_velocity,
            normal,
            control._predicted_position - paddle,
            control._predicted_velocity / 3,
            np.clip(control._arrival - control.elapsed, -1, 2),
            control.side,
            flags,
            control._prediction_valid,
            state["qpos"][:, control._qpos_ids] - control._preparation_base,
            control._reference - control._preparation_base,
            control._reference_velocity / control._max_velocity,
            control._executed_command,
            control._previous_action,
            (targets[:, 1:4] - paddle) * guidance_valid[:, None],
            targets[:, 4:7] / 3,
            np.clip(targets[:, 0] - control.elapsed, 0, 2) * guidance_valid,
            guidance_valid,
            guidance_age,
        )
    ).astype(np.float32)


def return_reward(env: ManagerBasedRlEnv, event: str) -> np.ndarray:
    return np.array(
        [event in episode.credit for episode in _control(env).episodes], dtype=np.float32
    )


def return_action_penalty(env: ManagerBasedRlEnv) -> np.ndarray:
    control = _control(env)
    return np.sum((control._command - control._previous_action) ** 2, axis=1) * control.ready


def return_ball_history(env: ManagerBasedRlEnv) -> np.ndarray:
    control = _control(env)
    state = control.backend.get_state(("qpos", "qvel"))
    position, _ = control.backend.get_body_pose_w(control._paddle_id)
    return np.column_stack(
        (
            state["qpos"][:, control._ball_qpos] - position[:, 0],
            state["qvel"][:, control._ball_qvel] / 3,
        )
    ).astype(np.float32)


def return_guidance_reward(env: ManagerBasedRlEnv, component: str) -> np.ndarray:
    if component not in {"reach", "strike"}:
        raise ValueError(f"Unknown incoming guidance component {component!r}")
    scores, _, _, _ = _control(env).guidance_state()
    return (scores[:, 0 if component == "reach" else 1] * env.step_dt).astype(np.float32)


def return_control_cost(env: ManagerBasedRlEnv, component: str) -> np.ndarray:
    control = _control(env)
    active = np.array(
        [
            control.ready[i] and not e.done and e.outgoing_time is None
            for i, e in enumerate(control.episodes)
        ]
    )
    if component == "offset":
        values = (
            control._reference[:, (0, 2, 3, 4)] - control._preparation_base[:, (0, 2, 3, 4)]
        ) / control._offset_scales
    elif component == "acceleration":
        values = control._reference_acceleration / control._max_acceleration
    else:
        raise ValueError(f"Unknown control cost component {component!r}")
    return (np.mean(np.clip(values, -1, 1) ** 2, axis=1) * active * env.step_dt).astype(np.float32)


def return_shaping_reward(
    env: ManagerBasedRlEnv, component: str, shaping_cfg: ReturnShapingCfg
) -> np.ndarray:
    """Give bounded continuous shot feedback once, after the contact ends."""
    from unilab.tasks.manipulation.table_tennis.return_shaping import estimate_outgoing_shot

    if component not in {
        "direction",
        "net_height",
        "landing",
        "target_depth",
        "target_short_edge",
        "target_landing",
        "target_return_apex",
    }:
        raise ValueError(f"Unknown shot-quality component {component!r}")
    values = np.zeros(env.num_envs, dtype=np.float32)
    for i, episode in enumerate(_control(env).episodes):
        if "outgoing" not in episode.credit:
            continue
        assert episode.outgoing_position is not None and episode.outgoing_velocity is not None
        shot = estimate_outgoing_shot(
            episode.outgoing_position, episode.outgoing_velocity, episode, shaping_cfg
        )
        values[i] = getattr(shot, component)
    return values


def return_done(env: ManagerBasedRlEnv) -> np.ndarray:
    return np.array([episode.done for episode in _control(env).episodes])


def return_metric(env: ManagerBasedRlEnv, condition: str) -> np.ndarray:
    control = _control(env)
    if condition == "side":
        return control.side.astype(np.float32)
    if condition == "prepared":
        return control.ready.astype(np.float32)
    if condition == "preparation_failed":
        return np.array(
            [episode.done and not control.ready[i] for i, episode in enumerate(control.episodes)],
            dtype=np.float32,
        )
    if condition not in {"success", "hit", "crossed_net"}:
        raise ValueError(f"Unknown return metric {condition!r}")
    return np.array([getattr(episode, condition) for episode in control.episodes], dtype=np.float32)
