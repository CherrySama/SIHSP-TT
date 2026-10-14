"""Policy timing and bounded residuals around a measured horizontal stroke.

Uses the shared stroke owner and public SimBackend control path (ADR-0011).
Zero actions replay the reference clock; -1 on timing holds it. A policy can
delay or slightly accelerate this monotonic clock without resetting its phase.
Reference motion is guidance, never evidence of actual legal ball return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.tasks.manipulation.table_tennis.return_control import (
    PantheraReturnAction,
    _control,
    _normal,
    return_observation,
)
from unilab.tasks.manipulation.table_tennis.stroke_control import (
    JOINT_NAMES,
    PantheraStrokeAction,
    PantheraStrokeActionCfg,
    horizontal_stroke_assessment,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HorizontalReturnActionCfg(PantheraStrokeActionCfg):
    """Distinct policy contract; old direct-velocity checkpoints are incompatible."""

    right_stroke: tuple[float, ...] = ()
    left_stroke: tuple[float, ...] = ()
    clock_max_rate: float = 1.02
    clock_acceleration: float = 5.0
    yaw_residual: float = 0.015
    arm_residual: float = 0.015
    tilt_residual: float = 0.03
    yaw_residual_velocity: float = 0.15
    yaw_residual_acceleration: float = 2.0
    yaw_boost_residual: float = 0.25
    yaw_boost_velocity: float = 1.2
    yaw_boost_acceleration: float = 16.0
    yaw_profile_offset: float = 0.12
    yaw_boost_profile_offset: float = 0.18
    yaw_profile_side_offset: float = 0.0
    yaw_strength_mode: str = "binary"
    residual_velocity: float = 0.15
    residual_acceleration: float = 2.0

    def build(self, env: ManagerBasedRlEnv) -> HorizontalReturnAction:
        return HorizontalReturnAction(self, env)


class HorizontalReturnAction(PantheraStrokeAction):
    """Actions: clock rate, face tilt, side, side strength, joint3 and joint4 residuals."""

    def __init__(self, cfg: HorizontalReturnActionCfg, env: ManagerBasedRlEnv) -> None:
        if (
            cfg.stroke_mode != "hermite"
            or cfg.preparation_reach is None
            or not cfg.residual_control
        ):
            raise ValueError("Horizontal return requires reach preparation and hermite residuals")
        values = (
            cfg.clock_acceleration,
            cfg.yaw_residual,
            cfg.arm_residual,
            cfg.tilt_residual,
            cfg.yaw_residual_velocity,
            cfg.yaw_residual_acceleration,
            cfg.yaw_boost_residual,
            cfg.yaw_boost_velocity,
            cfg.yaw_boost_acceleration,
            cfg.yaw_profile_offset,
            cfg.yaw_boost_profile_offset,
            cfg.residual_velocity,
            cfg.residual_acceleration,
        )
        if any(not np.isfinite(v) or v <= 0 for v in values) or not 1 <= cfg.clock_max_rate <= 1.1:
            raise ValueError("Invalid horizontal clock or residual limits")
        if not np.isfinite(cfg.yaw_profile_side_offset) or cfg.yaw_profile_side_offset < 0:
            raise ValueError("Yaw profile side offset must be finite and non-negative")
        if cfg.yaw_strength_mode not in {
            "binary",
            "positive_continuous",
            "positive_continuous_pre_contact",
        }:
            raise ValueError("Invalid yaw strength mode")
        if (
            cfg.yaw_boost_residual < cfg.yaw_residual
            or cfg.yaw_boost_velocity < cfg.yaw_residual_velocity
            or cfg.yaw_boost_acceleration < cfg.yaw_residual_acceleration
            or cfg.yaw_boost_profile_offset < cfg.yaw_profile_offset
        ):
            raise ValueError("Yaw boost limits must not be below the nominal yaw limits")
        self._horizontal_cfg = cfg
        super().__init__(cfg, env)
        self._profiles = np.asarray([cfg.left_stroke, cfg.right_stroke], dtype=float)
        if self._profiles.shape != (2, 13):
            raise ValueError("Horizontal return requires explicit left and right 13-column strokes")
        self.set_strokes(np.tile(self._profiles[0], (env.num_envs, 1)))
        self._clock = np.zeros(env.num_envs)
        self._clock_rate = np.ones(env.num_envs)
        self._clock_acceleration = np.zeros(env.num_envs)
        self._clock_updated = np.full(env.num_envs, -np.inf)
        self._residual = np.zeros((env.num_envs, 5))
        self._residual_speed = np.zeros_like(self._residual)
        self._residual_accel = np.zeros_like(self._residual)
        self._yaw_choice = np.full(env.num_envs, np.nan)
        self._yaw_strength = np.zeros(env.num_envs)
        self._residual_scales = np.array(
            [cfg.tilt_residual, cfg.yaw_residual, *([cfg.arm_residual] * 3)]
        )
        self._residual_velocity_scales = np.array(
            [cfg.residual_velocity, cfg.yaw_residual_velocity, *([cfg.residual_velocity] * 3)]
        )
        self._residual_acceleration_scales = np.array(
            [
                cfg.residual_acceleration,
                cfg.yaw_residual_acceleration,
                *([cfg.residual_acceleration] * 3),
            ]
        )
        # Cold FK envelope for this reach branch, replacing the original arm
        # branch. A bounding box is predictive guidance, not a reachability proof.
        poses = []
        for arm in self._reach_arm_poses[::10]:
            for yaw in np.linspace(float(self._limits[1, 0]), float(self._limits[1, 1]), 17):
                pose = self.home.copy()
                pose[1], pose[2:5] = yaw, arm
                poses.append(pose)
        points, _ = self.backend.evaluate_body_poses(
            JOINT_NAMES, np.asarray(poses), ("paddle_frame",)
        )
        self._workspace_lower = points[:, 0].min(axis=0) - 0.02
        self._workspace_upper = points[:, 0].max(axis=0) + 0.02
        self._height_workspace_padding = 0.06

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        super().reset(env_ids)
        ids = np.arange(self._env.num_envs)[slice(None) if env_ids is None else env_ids]
        self._clock[ids] = 0
        self._clock_rate[ids] = 1
        self._clock_acceleration[ids] = 0
        self._clock_updated[ids] = -np.inf
        self._residual[ids] = self._residual_speed[ids] = self._residual_accel[ids] = 0
        self._yaw_choice[ids] = np.nan
        self._yaw_strength[ids] = 0

    def _apply_yaw_profile(self, i: int) -> None:
        if not np.isfinite(self._yaw_choice[i]) or not np.isfinite(self._yaw_strength[i]):
            return
        side_offset = (
            self._horizontal_cfg.yaw_profile_side_offset if self._yaw_choice[i] < 0 else 0.0
        )
        offset = self._horizontal_cfg.yaw_profile_offset + side_offset + self._yaw_strength[i] * (
            self._horizontal_cfg.yaw_boost_profile_offset
            - self._horizontal_cfg.yaw_profile_offset
        )
        self.strokes[i, 9] += -self.side[i] * self._yaw_choice[i] * offset

    def _nominal_stroke(self, i: int) -> np.ndarray:
        side = int(self.side[i] > 0)
        return self._profiles[side].copy()

    def _commit_yaw_choice(self, i: int, flight: bool) -> None:
        """Hold one side while a separate non-negative action controls its strength."""
        new_choice = False
        if not flight and not np.isfinite(self._yaw_choice[i]):
            self._yaw_choice[i] = np.sign(self._command[i, 2])
            new_choice = True
        if not flight and self._horizontal_cfg.yaw_strength_mode.endswith("pre_contact"):
            self._yaw_strength[i] = float(np.clip(self._command[i, 3], 0.0, 1.0))
            self.strokes[i] = self._nominal_stroke(i)
            self._apply_yaw_profile(i)
        elif new_choice:
            self._yaw_strength[i] = (
                float(np.clip(self._command[i, 3], 0.0, 1.0))
                if self._horizontal_cfg.yaw_strength_mode == "positive_continuous"
                else float(self._command[i, 3] > 0)
            )
            self._apply_yaw_profile(i)
        if not flight:
            self._command[i, 2] = self._yaw_choice[i]

    def apply_actions(self) -> None:
        for i, episode in enumerate(self.episodes):
            if self.ready[i]:
                self._commit_yaw_choice(i, episode.outgoing_time is not None)
        super().apply_actions()

    def process_actions(self, actions: np.ndarray) -> None:
        super().process_actions(actions)
        for i in range(self._env.num_envs):
            self.strokes[i] = self._nominal_stroke(i)
        for i in range(self._env.num_envs):
            self._apply_yaw_profile(i)

    def _update_guidance(self) -> None:
        PantheraReturnAction._update_guidance(self)

    def _stroke_time(self, i: int) -> float:
        return float(self._clock[i])

    def _stroke_clock(self, i: int, dt: float) -> tuple[float, float, float]:
        if self._clock_updated[i] != self.elapsed[i]:
            if not np.isfinite(self._clock_updated[i]):
                self._clock[i] = self.prepared_time[i]
            command = self._executed_command[i, 0]
            goal = 1 + command * (1 if command < 0 else self._horizontal_cfg.clock_max_rate - 1)
            previous = self._clock_rate[i]
            self._clock_rate[i] += np.clip(
                goal - previous,
                -self._horizontal_cfg.clock_acceleration * dt,
                self._horizontal_cfg.clock_acceleration * dt,
            )
            self._clock_acceleration[i] = (self._clock_rate[i] - previous) / dt
            self._clock[i] += self._clock_rate[i] * dt
            self._clock_updated[i] = self.elapsed[i]
        return float(self._clock[i]), float(self._clock_rate[i]), float(self._clock_acceleration[i])

    def _set_stroke_reference(
        self, i: int, joint: int, position: float, velocity: float, acceleration: float
    ) -> None:
        column = joint  # yaw/2/3/4 correspond to residual columns 1/2/3/4.
        dt = self._env.physics_dt
        target_scale = self._residual_scales[column]
        velocity_limit = self._residual_velocity_scales[column]
        acceleration_limit = self._residual_acceleration_scales[column]
        if column == 1:
            strength = self._yaw_strength[i]
            target_scale += strength * (self._horizontal_cfg.yaw_boost_residual - target_scale)
            velocity_limit += strength * (
                self._horizontal_cfg.yaw_boost_velocity - velocity_limit
            )
            acceleration_limit += strength * (
                self._horizontal_cfg.yaw_boost_acceleration - acceleration_limit
            )
        target = (
            0.0
            if column == 2
            else target_scale * self._executed_command[i, column + 1]
        )
        previous = self._residual_speed[i, column]
        goal = np.clip(
            20 * (target - self._residual[i, column]), -velocity_limit, velocity_limit
        )
        speed = previous + np.clip(
            goal - previous, -acceleration_limit * dt, acceleration_limit * dt
        )
        offset = np.clip(
            self._residual[i, column] + speed * dt,
            -self._residual_scales[column],
            self._residual_scales[column],
        )
        speed = (offset - self._residual[i, column]) / dt
        self._residual[i, column] = offset
        self._residual_speed[i, column] = speed
        self._residual_accel[i, column] = (speed - previous) / dt
        super()._set_stroke_reference(
            i,
            joint,
            position + offset,
            velocity + speed,
            acceleration + self._residual_accel[i, column],
        )

    def _advance_swing(self, i: int, joint: int, command: float, dt: float) -> None:
        if joint != 5:
            super()._advance_swing(i, joint, command, dt)
            return
        flight = self.episodes[i].outgoing_time is not None
        tilt = (
            self.strokes[i, 6] + self._horizontal_cfg.tilt_residual * self._executed_command[i, 1]
        )
        target = self._preparation_base[i, 5] - self.side[i] * tilt
        target = np.clip(target, *self._limits[5])
        goal = 0.0
        if not flight and self._stroke_time(i) >= self._arrival[i] - self.strokes[i, 7]:
            goal = float(
                np.clip(
                    10 * (target - self._reference[i, 5]),
                    -self.reference_speed_fraction * self._max_velocity[5],
                    self.reference_speed_fraction * self._max_velocity[5],
                )
            )
        self._advance_velocity(
            i,
            5,
            goal,
            dt,
            float(self._limits[5, 0]),
            float(self._limits[5, 1]),
            float(self._max_acceleration[5]),
        )

    def episode_report(self, env_idx: int) -> dict[str, object]:
        report = super().episode_report(env_idx)
        report["horizontal_assessment"] = horizontal_stroke_assessment(report)
        report["clock_time"] = float(self._clock[env_idx])
        report["clock_rate"] = float(self._clock_rate[env_idx])
        report["policy_residual"] = self._residual[env_idx].tolist()
        report["policy_yaw_choice"] = (
            float(self._yaw_choice[env_idx]) if np.isfinite(self._yaw_choice[env_idx]) else None
        )
        report["policy_yaw_strength"] = float(self._yaw_strength[env_idx])
        return report


def horizontal_observation(
    env: ManagerBasedRlEnv,
    include_hit_lateral: bool = False,
) -> np.ndarray:
    control = _horizontal_control(env)
    columns: tuple[object, ...] = (
        return_observation(
            env,
        ),
        control._clock - control.elapsed,
        control._clock_rate,
        control._residual / control._residual_scales,
        control._residual_speed / control._residual_velocity_scales,
    )
    if include_hit_lateral:
        columns += (np.clip(control._predicted_position[:, 1] / 0.55, -1.0, 1.0),)
    return np.column_stack(columns).astype(np.float32)


def _horizontal_control(env: ManagerBasedRlEnv) -> HorizontalReturnAction:
    control = _control(env)
    if not isinstance(control, HorizontalReturnAction):
        raise TypeError("Horizontal rewards require the horizontal policy action")
    return control


def horizontal_cost(env: ManagerBasedRlEnv, component: str) -> np.ndarray:
    control = _horizontal_control(env)
    active = control.ready & np.array([not e.done for e in control.episodes])
    if component == "residual":
        values = np.mean(control._executed_command[:, 1:] ** 2, axis=1)
    elif component == "smoothness":
        values = np.mean((control._executed_command - control._previous_action) ** 2, axis=1) / 4
    elif component == "form":
        _, quaternion = control.backend.get_body_pose_w(control._paddle_id)
        normal = _normal(quaternion[:, 0])
        velocity = control.backend.get_body_lin_vel_w(control._paddle_id)[:, 0]
        tilt = np.arcsin(np.clip(abs(normal[:, 2]), 0, 1))
        elevation = np.arctan2(abs(velocity[:, 2]), np.maximum(0.01, velocity[:, 0]))
        values = np.mean(
            np.column_stack(
                (
                    np.maximum(0, tilt / np.deg2rad(15) - 1),
                    np.maximum(0, elevation / np.deg2rad(15) - 1) * (velocity[:, 0] > 0.3),
                    np.maximum(
                        0, (control._swing_upper[:, 2] - control._swing_lower[:, 2]) / 0.05 - 1
                    ),
                    np.maximum(0, control._followthrough_path / 0.15 - 1),
                )
            ),
            axis=1,
        )
    else:
        raise ValueError(f"Unknown horizontal cost {component!r}")
    return (np.clip(values, 0, 1) * active * env.step_dt).astype(np.float32)


def horizontal_strength_cost(
    env: ManagerBasedRlEnv,
    boost_start_y: float = 0.25,
    full_boost_y: float = 0.50,
) -> np.ndarray:
    """Penalize strength error without prescribing the selected left/right edge."""
    if not 0 <= boost_start_y < full_boost_y:
        raise ValueError("Strength guidance requires 0 <= boost_start_y < full_boost_y")
    control = _horizontal_control(env)
    active = control.ready & np.array([e.outgoing_time is None for e in control.episodes])
    target = np.clip(
        (np.abs(control._predicted_position[:, 1]) - boost_start_y)
        / (full_boost_y - boost_start_y),
        0.0,
        1.0,
    )
    error = (control._yaw_strength - target) ** 2
    return (error * active * env.step_dt).astype(np.float32)


def horizontal_metric(env: ManagerBasedRlEnv) -> np.ndarray:
    control = _horizontal_control(env)
    return np.array(
        [
            cast(dict, control.episode_report(i)["horizontal_assessment"])["accepted"]
            for i in range(env.num_envs)
        ],
        dtype=np.float32,
    )


def horizontal_quality(env: ManagerBasedRlEnv) -> np.ndarray:
    """Once-only smooth style quality after actual legal landing.

    Uses all measured assessment terms, including a complete 200 ms followthrough.
    Missing measurements have no quality credit. This bonus cannot reward failure.
    """
    control = _horizontal_control(env)
    values = np.zeros(env.num_envs, dtype=np.float32)
    for i, episode in enumerate(control.episodes):
        if "success" not in episode.credit:
            continue
        assessment = cast(dict, control.episode_report(i)["horizontal_assessment"])
        if any(v is None for v in assessment.get("measurements", {}).values()):
            continue
        values[i] = np.exp(-20 * assessment.get("normalized_excess", np.inf))
    return values
