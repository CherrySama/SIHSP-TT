"""Pure incoming free-flight forecasts and bounded pre-contact guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True, kw_only=True)
class ReturnGuidanceCfg:
    gravity: float = 9.81
    horizon: float = 0.5
    sample_dt: float = 0.01
    freshness: float = 0.05
    tolerance: float = 0.02
    distance_scale: float = 0.06
    fade_far: float = 0.16
    fade_near: float = 0.06
    strike_far: float = 0.20
    strike_near: float = 0.10
    velocity_scale: float = 0.8
    paddle_radius: float = 0.07
    paddle_half_thickness: float = 0.00725
    ball_radius: float = 0.02

    def __post_init__(self) -> None:
        if any(
            isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
            for v in vars(self).values()
        ):
            raise ValueError("Guidance scales must be finite and positive")
        if self.fade_far <= self.fade_near or self.strike_far <= self.strike_near:
            raise ValueError("Guidance time ramps must have ordered endpoints")
        if self.sample_dt > self.horizon:
            raise ValueError("Guidance sample_dt must not exceed horizon")


def flight_window(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    timestamp: float,
    now: float,
    surface: float,
    lower: np.ndarray,
    upper: np.ndarray,
    cfg: ReturnGuidanceCfg,
) -> np.ndarray:
    """Return future [absolute time, XYZ, velocity] before the first descending landing.

    The origin may be a predicted receiver rebound or a current observed
    post-bounce state. Bounds are a cold-computed workspace envelope, not an
    inverse-kinematics guarantee that every interior point is reachable.
    """
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("Flight states must be three-vectors")
    if not np.isfinite(np.r_[position, velocity, timestamp, now, surface]).all():
        raise ValueError("Flight states must be finite")
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("Workspace bounds must be three-vectors")
    if not np.isfinite(np.r_[lower, upper]).all() or np.any(lower > upper):
        raise ValueError("Workspace bounds must be finite and ordered")
    if position[2] < surface:
        return np.empty((0, 7))
    disc = velocity[2] ** 2 + 2 * cfg.gravity * (position[2] - surface)
    root = math.sqrt(disc)
    end = (
        2 * (position[2] - surface) / (root - velocity[2])
        if velocity[2] < 0
        else (velocity[2] + root) / cfg.gravity
    )
    start = max(0.0, now - timestamp)
    end = min(end, now + cfg.horizon - timestamp)
    if end <= start:
        return np.empty((0, 7))
    times = np.arange(start, end, cfg.sample_dt)
    positions = position + times[:, None] * velocity
    positions[:, 2] -= 0.5 * cfg.gravity * times**2
    velocities = np.tile(velocity, (len(times), 1))
    velocities[:, 2] -= cfg.gravity * times
    valid = np.all((positions >= lower) & (positions <= upper), axis=1)
    valid &= positions[:, 2] > surface
    valid &= timestamp + times > now
    return np.column_stack((timestamp + times, positions, velocities))[valid]


def guidance_scores(
    window: np.ndarray,
    *,
    now: float,
    paddle: np.ndarray,
    normal: np.ndarray,
    paddle_velocity: np.ndarray,
    receiver_bounced: bool,
    cfg: ReturnGuidanceCfg,
) -> tuple[float, float, np.ndarray | None]:
    """Position guidance fades near contact; stationary paddles earn no swing score."""
    if window.ndim != 2 or window.shape[1] != 7:
        raise ValueError("Guidance window must have shape (samples, 7)")
    if paddle.shape != (3,) or normal.shape != (3,) or paddle_velocity.shape != (3,):
        raise ValueError("Paddle states must be three-vectors")
    if (
        not np.isfinite(window).all()
        or not np.isfinite(np.r_[now, paddle, normal, paddle_velocity]).all()
    ):
        raise ValueError("Guidance window and paddle states must be finite")
    if not isinstance(receiver_bounced, bool):
        raise ValueError("receiver_bounced must be a boolean past-event flag")
    normal_length = float(np.linalg.norm(normal))
    if normal_length == 0:
        raise ValueError("Paddle normal must have nonzero length")
    normal = normal / normal_length
    future = window[window[:, 0] > now]
    if not len(future):
        return 0.0, 0.0, None
    distances = np.linalg.norm(future[:, 1:4] - paddle, axis=1)
    target = future[int(np.argmin(distances))]
    difference = target[1:4] - paddle
    tau = target[0] - now
    distance_score = math.exp(
        -0.5 * (max(float(np.min(distances)) - cfg.tolerance, 0) / cfg.distance_scale) ** 2
    )
    fade = float(np.clip((tau - cfg.fade_near) / (cfg.fade_far - cfg.fade_near), 0, 1))
    close = float(np.clip((cfg.strike_far - tau) / (cfg.strike_far - cfg.strike_near), 0, 1))
    axial = abs(float(difference @ normal))
    radial = math.sqrt(max(float(difference @ difference) - axial**2, 0))
    contact_score = math.exp(
        -0.5
        * (
            (max(axial - cfg.ball_radius - cfg.paddle_half_thickness, 0) / cfg.distance_scale) ** 2
            + (max(radial - cfg.paddle_radius - cfg.ball_radius, 0) / cfg.distance_scale) ** 2
        )
    )
    forward = float(np.clip(paddle_velocity[0] / cfg.velocity_scale, 0, 1))
    strike = close * contact_score * forward if receiver_bounced else 0.0
    return distance_score * fade, strike, target.copy()
