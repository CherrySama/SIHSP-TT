"""Simulator-independent lateral alignment and bounded waist preparation.

Positions use the prediction core's table frame. Rail alignment controls only
Y; this controller does not establish height alignment or a successful return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class RailAlignmentCfg:
    """Provisional simulation limits, not Panthera hardware specifications."""

    lower: float = -0.75
    upper: float = 0.75
    max_velocity: float = 1.5
    max_acceleration: float = 10.0
    max_force: float = 100.0
    position_tolerance: float = 0.01
    settled_velocity: float = 0.05
    preparation_margin: float = 0.05

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _finite(name, getattr(self, name))
        if self.lower >= self.upper:
            raise ValueError("RailAlignmentCfg lower must be below upper")
        for name in (
            "max_velocity",
            "max_acceleration",
            "max_force",
            "position_tolerance",
            "settled_velocity",
            "preparation_margin",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"RailAlignmentCfg {name} must be positive")


@dataclass(frozen=True)
class RailTarget:
    position: float
    ball_y: float
    arrival_time: float
    within_travel: bool


@dataclass(frozen=True)
class WaistPreparationCfg:
    """Trial preparation parameters, not certified hardware operating limits."""

    angle: float = math.pi / 2
    max_velocity: float = 3.0
    max_acceleration: float = 12.0
    side_deadband: float = 0.03
    position_tolerance: float = 0.015
    settled_velocity: float = 0.05
    settle_duration: float = 0.03
    prediction_max_age: float = 0.05

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if _finite(name, getattr(self, name)) <= 0:
                raise ValueError(f"WaistPreparationCfg {name} must be positive")


class WaistPreparationController:
    """Latch an incoming side, then accelerate and brake a joint1 trajectory."""

    def __init__(self, home_angle: float, cfg: WaistPreparationCfg) -> None:
        self.home_angle = _finite("home_angle", home_angle)
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.target: float | None = None
        self.position = self.home_angle
        self.velocity = 0.0

    def observe(self, *, ball_y: float, initial_paddle_y: float) -> None:
        offset = _finite("ball_y", ball_y) - _finite("initial_paddle_y", initial_paddle_y)
        if self.target is None:
            # A central ball still needs a facing paddle; use the right-side
            # preparation within the deadband rather than keeping home yaw.
            side = 1.0 if offset > self.cfg.side_deadband else -1.0
            self.target = self.home_angle + side * self.cfg.angle

    def advance(self, dt: float) -> float:
        dt = _finite("dt", dt)
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self.target is None:
            return self.position
        error = self.target - self.position
        step = self.cfg.max_acceleration * dt
        stopping_speed = math.sqrt(step**2 + 2 * self.cfg.max_acceleration * abs(error)) - step
        goal = math.copysign(min(self.cfg.max_velocity, stopping_speed), error)
        self.velocity += _clip(goal - self.velocity, -step, step)
        self.position += self.velocity * dt
        return self.position

    def ready(self, *, position: float, velocity: float) -> bool:
        return bool(
            self.target is not None
            and abs(_finite("position", position) - self.target) <= self.cfg.position_tolerance
            and abs(_finite("velocity", velocity)) <= self.cfg.settled_velocity
        )


class RailAlignmentController:
    """Keep only the latest usable future target; reset between incoming balls."""

    def __init__(self, cfg: RailAlignmentCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else RailAlignmentCfg()
        self.target: RailTarget | None = None
        self._holding = False

    def reset(self) -> None:
        self.target = None
        self._holding = False

    def observe(
        self, *, rail_position: float, paddle_y: float, ball_y: float, arrival_time: float
    ) -> RailTarget:
        """Resolve the rail position from measured paddle and predicted ball Y."""
        rail_position = _finite("rail_position", rail_position)
        paddle_y = _finite("paddle_y", paddle_y)
        ball_y = _finite("ball_y", ball_y)
        arrival_time = _finite("arrival_time", arrival_time)
        position = rail_position + ball_y - paddle_y
        self.target = RailTarget(
            position=_clip(position, self.cfg.lower, self.cfg.upper),
            ball_y=ball_y,
            arrival_time=arrival_time,
            within_travel=self.cfg.lower <= position <= self.cfg.upper,
        )
        return self.target

    def acceleration(self, *, now: float, position: float, velocity: float, dt: float) -> float:
        """Return a bounded acceleration, including braking at the target and limits."""
        now = _finite("now", now)
        position = _finite("position", position)
        velocity = _finite("velocity", velocity)
        dt = _finite("dt", dt)
        if dt <= 0:
            raise ValueError("dt must be positive")
        acceleration = self.cfg.max_acceleration
        # The physical travel boundary is not a target to cross by a rounding
        # error. Reserve a few integration intervals of acceleration distance.
        guard = min(4 * acceleration * dt**2, (self.cfg.upper - self.cfg.lower) / 4)
        lower, upper = self.cfg.lower + guard, self.cfg.upper - guard

        def stopping_speed(distance: float) -> float:
            # Reserve one integration interval before v^2/(2a) braking.
            # This is a physical velocity goal, never a rewritten measured state.
            step = acceleration * dt
            return math.sqrt(step**2 + 2 * acceleration * max(0.0, distance)) - step

        goal = 0.0
        if self.target is not None and now < self.target.arrival_time:
            error = _clip(self.target.position, lower, upper) - position
            # Once settled, tolerate sub-centimetre posture/forecast changes
            # instead of repeatedly accelerating to chase a millimetre offset.
            if abs(error) > self.cfg.position_tolerance:
                self._holding = False
            elif (
                abs(error) <= self.cfg.position_tolerance / 2
                and abs(velocity) <= self.cfg.settled_velocity
            ):
                self._holding = True
            if not self._holding:
                goal = math.copysign(
                    min(self.cfg.max_velocity, stopping_speed(abs(error))), error
                )
        goal = _clip(
            goal,
            -stopping_speed(position - lower),
            stopping_speed(upper - position),
        )
        return _clip((goal - velocity) / dt, -acceleration, acceleration)

    def ready(self, *, now: float, paddle_y: float, velocity: float) -> bool:
        """Report measured lateral readiness before the preparation deadline."""
        now = _finite("now", now)
        paddle_y = _finite("paddle_y", paddle_y)
        velocity = _finite("velocity", velocity)
        return bool(
            self.target is not None
            and self.target.within_travel
            and now <= self.target.arrival_time - self.cfg.preparation_margin
            and abs(paddle_y - self.target.ball_y) <= self.cfg.position_tolerance
            and abs(velocity) <= self.cfg.settled_velocity
        )
