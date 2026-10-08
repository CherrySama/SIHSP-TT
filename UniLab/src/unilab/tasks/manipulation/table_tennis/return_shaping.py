"""Bounded shot-quality feedback from an observed free-flight state.

The gravity-only forecast ends at the first descending table-height crossing.
It does not predict contact impulses, table rebounds or episode success.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from unilab.tasks.manipulation.table_tennis.return_events import ReturnEpisode


@dataclass(frozen=True, kw_only=True)
class ReturnShapingCfg:
    """Owner-supplied scales in metres, seconds and metres/second."""

    gravity: float
    forward_speed: float
    net_clearance: float
    net_height_scale: float
    landing_x_scale: float
    landing_y_scale: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"ReturnShapingCfg {name} must be finite")
            if value < 0 or (name != "net_clearance" and value == 0):
                raise ValueError(f"ReturnShapingCfg {name} has an invalid scale")


@dataclass(frozen=True)
class OutgoingShot:
    """Direction is in [-1, 1], other scores in [0, 1]; no score is an event."""

    direction: float = 0.0
    net_height: float = 0.0
    landing: float = 0.0
    landing_time: float | None = None
    landing_position: tuple[float, float, float] | None = None
    net_position: tuple[float, float, float] | None = None


def estimate_outgoing_shot(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    geometry: ReturnEpisode,
    cfg: ReturnShapingCfg,
) -> OutgoingShot:
    """Score one post-contact state with a free-flight parabola.

    Signed forward speed supplies feedback for backward and weak departures.
    Landing feedback decays smoothly outside the opponent's safe rectangle.
    Net feedback penalizes horizontal distance remaining, insufficient
    clearance and lateral escape. If the ball lands before reaching the net,
    its first landing state supplies a proxy, without reporting net passage.
    Heights above the configured clearance receive no additional net reward.
    """
    if (
        len(position) != 3
        or len(velocity) != 3
        or not all(math.isfinite(value) for value in (*position, *velocity))
    ):
        raise ValueError("Outgoing shot position and velocity must be finite three-vectors")
    x, y, z = position
    vx, vy, vz = velocity
    horizontal_speed = math.hypot(vx, vy)
    direction = (
        max(-1.0, min(vx / cfg.forward_speed, 1.0)) * abs(vx) / horizontal_speed
        if horizontal_speed > 0
        else 0.0
    )
    surface = geometry.table_z + geometry.ball_radius
    if z < surface:
        return OutgoingShot(direction=direction)
    # z + vz*t - g*t²/2 = surface; use the descending root.
    discriminant = vz * vz + 2 * cfg.gravity * (z - surface)
    root = math.sqrt(discriminant)
    # Rationalize the root for descending samples near the surface, where
    # adding vz and sqrt(discriminant) would lose their small difference.
    flight_time = 2 * (z - surface) / (root - vz) if vz < 0 else (vz + root) / cfg.gravity
    landing_x = x + vx * flight_time
    landing_y = y + vy * flight_time
    low_x = geometry.ball_radius
    high_x = geometry.table_half_length - geometry.ball_radius
    half_y = geometry.table_half_width - geometry.ball_radius
    distance_x = max(low_x - landing_x, landing_x - high_x, 0.0)
    distance_y = max(abs(landing_y) - half_y, 0.0)
    landing = math.exp(
        -0.5 * ((distance_x / cfg.landing_x_scale) ** 2 + (distance_y / cfg.landing_y_scale) ** 2)
    )
    # Sample at the complete-sphere exit used by the physical event reducer.
    net_x = geometry.net_half_thickness + geometry.ball_radius
    net_position = None
    # Until the net is reachable, use the first landing as the scoring proxy.
    # Both branches coincide when the landing point approaches the net plane.
    sample_x, sample_y, sample_z = landing_x, landing_y, surface
    net_time = (net_x - x) / vx if vx > 0 else math.inf
    if 0 <= net_time <= flight_time:
        net_y = y + vy * net_time
        net_z = z + vz * net_time - 0.5 * cfg.gravity * net_time * net_time
        net_position = (net_x, net_y, net_z)
        sample_x, sample_y, sample_z = net_position
    target_z = geometry.net_top_z + geometry.ball_radius + cfg.net_clearance
    horizontal_error = max(net_x - sample_x, 0.0)
    height_error = max(target_z - sample_z, 0.0)
    lateral_error = max(abs(sample_y) - (geometry.net_half_width - geometry.ball_radius), 0.0)
    net_height = math.exp(
        -0.5
        * (
            (horizontal_error / cfg.landing_x_scale) ** 2
            + (height_error / cfg.net_height_scale) ** 2
            + (lateral_error / cfg.landing_y_scale) ** 2
        )
    )
    return OutgoingShot(
        direction=direction,
        net_height=net_height,
        landing=landing,
        landing_time=flight_time,
        landing_position=(landing_x, landing_y, surface),
        net_position=net_position,
    )
