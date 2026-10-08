"""Gravity and calibrated, no-spin table-bounce prediction.

This module imports neither a simulator nor a robot framework. The current
model is a baseline for no-spin feeds: air drag and spin are not modelled.
Its physical parameters must be calibrated before use on real hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from ball_state import BallState


@dataclass(frozen=True)
class BallPredictionCfg:
    """Geometry and effective bounce parameters supplied by the application.

    Restitution is an effective forecasting coefficient, not an assertion
    about a rubber/table material. ``candidate_x`` is a diagnostic section;
    crossing it does not establish robot reachability or a hitting pose.
    """

    normal_restitution: float
    tangential_retention: float
    table_z: float = 0.76
    ball_radius: float = 0.02
    table_half_length: float = 1.37
    table_half_width: float = 0.7625
    net_top_z: float = 0.9125
    net_half_thickness: float = 0.006
    gravity: float = 9.81
    candidate_x: float = -1.39
    horizon: float = 1.0
    sample_dt: float = 0.01
    contact_band: float = 0.01
    allow_opponent_bounce: bool = False
    opponent_tangential_retention: float = 0.94
    opponent_contact_duration: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "allow_opponent_bounce":
                if not isinstance(value, bool):
                    raise ValueError("allow_opponent_bounce must be a boolean")
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"BallPredictionCfg {name} must be finite")
        for name in (
            "normal_restitution",
            "tangential_retention",
            "opponent_tangential_retention",
        ):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"BallPredictionCfg {name} must be in (0, 1]")
        for name in (
            "ball_radius",
            "table_half_length",
            "table_half_width",
            "gravity",
            "horizon",
            "sample_dt",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"BallPredictionCfg {name} must be positive")
        if min(self.net_half_thickness, self.contact_band, self.opponent_contact_duration) < 0:
            raise ValueError(
                "BallPredictionCfg thickness and contact band must be non-negative"
            )
        if self.sample_dt > self.horizon:
            raise ValueError("BallPredictionCfg sample_dt must not exceed horizon")
        if self.ball_radius >= min(self.table_half_width, self.table_half_length):
            raise ValueError("BallPredictionCfg ball must fit inside the table")
        if self.net_top_z <= self.table_z:
            raise ValueError("BallPredictionCfg net top must be above the table")
        if self.candidate_x >= -self.net_half_thickness - self.ball_radius:
            raise ValueError("BallPredictionCfg candidate must be on the robot side")


@dataclass(frozen=True)
class BallPrediction:
    """Samples from the current state onward, on the capture timestamp clock."""

    source_timestamp: float
    trajectory: tuple[BallState, ...]
    net_crossing: BallState | None = None
    first_bounce: BallState | None = None
    candidate: BallState | None = None
    valid: bool = False
    reason: str = ""
    opponent_bounce: BallState | None = None


def _flight(state: BallState, time: float, gravity: float) -> BallState:
    x, y, z = state.position
    vx, vy, vz = state.velocity
    return BallState(
        timestamp=state.timestamp + time,
        position=(x + vx * time, y + vy * time, z + vz * time - 0.5 * gravity * time**2),
        velocity=(vx, vy, vz - gravity * time),
    )


def _samples(state: BallState, duration: float, cfg: BallPredictionCfg) -> list[BallState]:
    result = [state]
    for index in range(1, math.ceil(duration / cfg.sample_dt)):
        result.append(_flight(state, index * cfg.sample_dt, cfg.gravity))
    if duration > 0:
        result.append(_flight(state, duration, cfg.gravity))
    return result


def predict_ball(
    state: BallState,
    cfg: BallPredictionCfg,
    *,
    has_bounced: bool = False,
    now: float | None = None,
) -> BallPrediction:
    """Forecast a receiver bounce and, optionally, a preceding opponent bounce.

    ``has_bounced`` is past-event information from the state estimator, not
    future information from a launcher. It is necessary because a single
    upward-moving snapshot cannot distinguish a launch from a rebound.
    Return times remain absolute capture-clock times: a planner must compare
    them with its current time, including any measurement/processing delay.
    """

    def invalid(reason: str) -> BallPrediction:
        return BallPrediction(source_timestamp=state.timestamp, trajectory=(), reason=reason)

    if not isinstance(has_bounced, bool):
        raise ValueError("has_bounced must be a boolean past-event flag")
    if now is None:
        now = state.timestamp
    if isinstance(now, bool) or not isinstance(now, Real) or not math.isfinite(now):
        raise ValueError("now must be finite seconds on the capture timestamp clock")
    if now < state.timestamp:
        return invalid("future_timestamp")
    if not state.valid:
        return invalid("invalid_measurement")
    if state.velocity[0] >= -1e-8:
        return invalid("not_incoming")
    if state.position[0] <= cfg.candidate_x:
        return invalid("candidate_already_passed")
    plane_z = cfg.table_z + cfg.ball_radius
    if state.position[2] <= plane_z + cfg.contact_band:
        # A soft table contact spans multiple samples in simulation and in
        # reality. Its state is not a free-flight state; do not bounce it twice.
        return invalid("contact_transition")

    net_edge = cfg.net_half_thickness + cfg.ball_radius
    if has_bounced and state.position[0] >= -net_edge:
        return invalid("inconsistent_bounce_history")
    if not has_bounced and not cfg.allow_opponent_bounce and state.position[0] >= -net_edge:
        for x in (min(state.position[0], net_edge), -net_edge):
            crossing = _flight(state, (x - state.position[0]) / state.velocity[0], cfg.gravity)
            if crossing.position[2] <= cfg.net_top_z + cfg.ball_radius:
                return invalid("net_collision")
            if abs(crossing.position[1]) >= cfg.table_half_width - cfg.ball_radius:
                return invalid("outside_net_playing_width")
    net_crossing = None
    opponent_bounce = None
    first_bounce = None
    segments: list[tuple[BallState, float]] = []
    after = state
    elapsed = 0.0

    def bounce_time(point: BallState) -> float:
        vz = point.velocity[2]
        return (
            vz + math.sqrt(vz**2 + 2 * cfg.gravity * (point.position[2] - plane_z))
        ) / cfg.gravity

    def rebound(point: BallState, retention: float) -> BallState:
        vx, vy, vz = point.velocity
        return BallState(
            timestamp=point.timestamp,
            position=point.position,
            velocity=(retention * vx, retention * vy, -cfg.normal_restitution * vz),
        )

    if not has_bounced:
        # Derive every future impact from the measured state. Launcher labels
        # are never inputs; 'has_bounced' refers only to a past receiver bounce.
        time = bounce_time(after)
        impact = _flight(after, time, cfg.gravity)
        if impact.position[0] >= net_edge and cfg.allow_opponent_bounce:
            bx, by, _ = impact.position
            if bx >= cfg.table_half_length - cfg.ball_radius:
                return invalid("opponent_bounce_outside_table")
            if abs(by) >= cfg.table_half_width - cfg.ball_radius:
                return invalid("opponent_bounce_outside_table_width")
            opponent_bounce = impact
            segments.append((after, time))
            after = rebound(impact, cfg.opponent_tangential_retention)
            contact_time = cfg.opponent_contact_duration
            # Resolve a calibrated finite contact interval before the next
            # flight. Its planar displacement uses mean incident/exit speed.
            after = BallState(
                timestamp=after.timestamp + contact_time,
                position=(
                    impact.position[0]
                    + 0.5 * (impact.velocity[0] + after.velocity[0]) * contact_time,
                    impact.position[1]
                    + 0.5 * (impact.velocity[1] + after.velocity[1]) * contact_time,
                    plane_z,
                ),
                velocity=after.velocity,
            )
            # Contact stays at table height, below the net. It cannot carry
            # the ball into the net slab between the two free-flight segments.
            if after.position[0] <= net_edge:
                return invalid("net_collision")
            elapsed += time + contact_time
            time = bounce_time(after)
            impact = _flight(after, time, cfg.gravity)
        first_bounce = impact
        bx, by, _ = first_bounce.position
        if not -cfg.table_half_length + cfg.ball_radius < bx < -net_edge:
            return invalid("first_bounce_outside_robot_half")
        if abs(by) >= cfg.table_half_width - cfg.ball_radius:
            return invalid("first_bounce_outside_table_width")
        if bx <= cfg.candidate_x:
            return invalid("candidate_before_first_bounce")
        if elapsed + time >= cfg.horizon:
            return invalid("bounce_outside_horizon")
        segments.append((after, time))
        after = rebound(first_bounce, cfg.tangential_retention)
        elapsed += time

    duration = (cfg.candidate_x - after.position[0]) / after.velocity[0]
    if duration <= 0 or elapsed + duration > cfg.horizon:
        return invalid("candidate_outside_horizon")
    candidate = _flight(after, duration, cfg.gravity)
    if candidate.timestamp <= now:
        return invalid("candidate_already_passed")
    if candidate.position[2] <= plane_z + cfg.contact_band:
        return invalid("candidate_too_low_or_after_second_bounce")
    if abs(candidate.position[1]) >= cfg.table_half_width - cfg.ball_radius:
        return invalid("candidate_outside_receiving_width")
    if first_bounce is not None:
        next_bounce_time = 2 * after.velocity[2] / cfg.gravity
        if next_bounce_time < duration:
            return invalid("second_bounce_before_candidate")
    segments.append((after, duration))
    # Check the net on the actual segment traversing it, after any opponent
    # bounce. A continuation of the initial parabola would be incorrect.
    trajectory: list[BallState] = []
    for segment, segment_duration in segments:
        end_x = segment.position[0] + segment.velocity[0] * segment_duration
        if segment.position[0] >= -net_edge and end_x <= net_edge:
            for x in (min(segment.position[0], net_edge), max(end_x, -net_edge)):
                time = (x - segment.position[0]) / segment.velocity[0]
                crossing = _flight(segment, time, cfg.gravity)
                if crossing.position[2] <= cfg.net_top_z + cfg.ball_radius:
                    return invalid("net_collision")
                if abs(crossing.position[1]) >= cfg.table_half_width - cfg.ball_radius:
                    return invalid("outside_net_playing_width")
            if segment.position[0] >= 0 >= end_x:
                time = -segment.position[0] / segment.velocity[0]
                net_crossing = _flight(segment, time, cfg.gravity)
        points = _samples(segment, segment_duration, cfg)
        trajectory.extend(
            points[1:]
            if trajectory and points[0].timestamp == trajectory[-1].timestamp
            else points
        )
    if now > state.timestamp:
        # Propagate latency using the predicted segment containing 'now'. Do
        # not relabel an old measured position as the current ball position.
        if opponent_bounce is not None and (
            opponent_bounce.timestamp
            <= now
            <= opponent_bounce.timestamp + cfg.opponent_contact_duration
        ):
            return invalid("contact_transition")
        segment = next(start for start, length in segments if now <= start.timestamp + length)
        present = _flight(segment, now - segment.timestamp, cfg.gravity)
        if present.position[2] <= plane_z + cfg.contact_band:
            return invalid("contact_transition")
        trajectory = [present] + [point for point in trajectory if point.timestamp > now]
    return BallPrediction(
        source_timestamp=state.timestamp,
        trajectory=tuple(trajectory),
        net_crossing=net_crossing,
        first_bounce=first_bounce,
        candidate=candidate,
        valid=True,
        reason="ok",
        opponent_bounce=opponent_bounce,
    )
