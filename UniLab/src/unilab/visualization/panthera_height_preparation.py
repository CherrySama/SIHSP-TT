"""Cold-path height lookup for the native, two-sided preparation preview."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class HeightPreparationCfg:
    """Simulation trajectory limits; no hardware operating claim."""

    max_velocity: float = 1.5
    max_acceleration: float = 8.0
    position_tolerance: float = 0.01

    def __post_init__(self) -> None:
        for value in (self.max_velocity, self.max_acceleration, self.position_tolerance):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Height preparation limits must be positive and finite")


class PantheraHeightPreparation:
    """Coordinate q234 by (phase/2, phase, -phase/2), preserving pitch.

    q2-q3-q4 stays at home, including during movement. This local branch
    spreads the height change across all three joints. FK is sampled once;
    online targets interpolate the monotonic height table, without IK.
    The table establishes geometric reachability, not dynamic collision safety.
    """

    direction = np.array([0.5, 1.0, -0.5])

    def __init__(
        self, model: mujoco.MjModel, *, yaw_angle: float, cfg: HeightPreparationCfg | None = None
    ) -> None:
        self.cfg = cfg or HeightPreparationCfg()
        self.data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self.data, model.key("home").id)
        home = self.data.qpos.copy()
        arm_ids = np.array([model.joint(f"joint{i}").id for i in (1, 2, 3, 4, 6)])
        qpos = model.jnt_qposadr[arm_ids]
        self.home_arm = home[qpos].copy()
        self.paddle = model.body("paddle_frame").id
        # Deliberately local, monotonic branch: unsupported heights fail closed.
        self.phases = np.linspace(-0.1, 0.8, 301)
        self.points: dict[int, np.ndarray] = {}
        limits = model.jnt_range[arm_ids[1:4]]
        for side in (-1, 1):
            points = []
            normals = []
            for phase in self.phases:
                arm = self.home_arm.copy()
                arm[0] += side * yaw_angle
                arm[1:4] += phase * self.direction
                if np.any(arm[1:4] < limits[:, 0]) or np.any(arm[1:4] > limits[:, 1]):
                    raise ValueError("Height lookup exceeds joint2--4 ranges")
                self.data.qpos[:] = home
                self.data.qpos[qpos] = arm
                mujoco.mj_forward(model, self.data)
                points.append(self.data.xpos[self.paddle].copy())
                normals.append(self.data.xmat[self.paddle].reshape(3, 3)[:, 2].copy())
            sampled = np.asarray(points)
            if np.any(np.diff(sampled[:, 2]) <= 0):
                raise ValueError("Height preparation branch is not monotonic")
            if np.ptp(sampled[:, 0]) > 1e-8 or not np.allclose(normals, normals[0], atol=1e-8):
                raise ValueError("Height curve changes the prediction section or paddle normal")
            self.points[side] = sampled
        self.heights = self.points[-1][:, 2]
        if not np.allclose(self.heights, self.points[1][:, 2], atol=1e-8):
            raise ValueError("Left and right height branches differ")
        self.reset()

    def reset(self) -> None:
        self.target: float | None = None
        self.target_height: float | None = None
        self.goal_paddle: np.ndarray | None = None
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0

    def observe(self, *, height: float, side: int) -> bool:
        if side not in (-1, 1) or not np.isfinite(height):
            raise ValueError("Height target requires a finite height and left/right side")
        if not self.heights[0] <= height <= self.heights[-1]:
            return False
        self.target = float(np.interp(height, self.heights, self.phases))
        self.target_height = float(height)
        self.goal_paddle = np.array(
            [np.interp(height, self.heights, self.points[side][:, axis]) for axis in range(3)]
        )
        return True

    @property
    def target_arm(self) -> np.ndarray:
        phase = self.target if self.target is not None else 0.0
        return self.home_arm[1:4] + phase * self.direction

    def advance(self, dt: float) -> np.ndarray:
        """Advance the common phase; q3 sets the largest joint speed/accel."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Height trajectory timestep must be positive and finite")
        previous_velocity = self.velocity
        if self.target is not None:
            error = self.target - self.position
            step = self.cfg.max_acceleration * dt
            stopping_speed = np.sqrt(step**2 + 2 * self.cfg.max_acceleration * abs(error)) - step
            goal = np.copysign(min(self.cfg.max_velocity, stopping_speed), error)
            self.velocity += float(np.clip(goal - self.velocity, -step, step))
            self.position += self.velocity * dt
        self.acceleration = (self.velocity - previous_velocity) / dt
        return self.home_arm[1:4] + self.position * self.direction
