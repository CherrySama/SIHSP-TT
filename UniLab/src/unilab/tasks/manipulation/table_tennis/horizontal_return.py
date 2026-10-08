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
    residual_velocity: float = 0.15
    residual_acceleration: float = 2.0

    def build(self, env: ManagerBasedRlEnv) -> HorizontalReturnAction:
        return HorizontalReturnAction(self, env)


class HorizontalReturnAction(PantheraStrokeAction):
    """Actions: clock rate, face tilt, yaw, joint2, joint3 and joint4 residuals."""

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
            cfg.residual_velocity,
            cfg.residual_acceleration,
        )
        if any(not np.isfinite(v) or v <= 0 for v in values) or not 1 <= cfg.clock_max_rate <= 1.1:
            raise ValueError("Invalid horizontal clock or residual limits")
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
        self._residual_scales = np.array(
            [cfg.tilt_residual, cfg.yaw_residual, *([cfg.arm_residual] * 3)]
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

    def process_actions(self, actions: np.ndarray) -> None:
        super().process_actions(actions)
        self.strokes[:] = self._profiles[(self.side > 0).astype(int)]

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
        cfg = self._horizontal_cfg
        dt = self._env.physics_dt
        target = self._residual_scales[column] * self._executed_command[i, column + 1]
        previous = self._residual_speed[i, column]
        goal = np.clip(
            20 * (target - self._residual[i, column]), -cfg.residual_velocity, cfg.residual_velocity
        )
        speed = previous + np.clip(
            goal - previous, -cfg.residual_acceleration * dt, cfg.residual_acceleration * dt
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
        return report


def horizontal_observation(env: ManagerBasedRlEnv) -> np.ndarray:
    control = _horizontal_control(env)
    return np.column_stack(
        (
            return_observation(env),
            control._clock - control.elapsed,
            control._clock_rate,
            control._residual / control._residual_scales,
            control._residual_speed / control._horizontal_cfg.residual_velocity,
        )
    ).astype(np.float32)


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
