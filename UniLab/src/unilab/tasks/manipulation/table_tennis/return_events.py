"""Single-return episode events from timestamped physical contact samples.

The robot occupies negative X. Contact labels must describe contacts involving
the ball, sampled on every physical substep, rather than proximity estimates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

_CONTACT_LABELS = frozenset({"paddle", "table", "net", "robot", "floor"})


def _vector(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must have three components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must be finite")
    return result[0], result[1], result[2]


@dataclass
class ReturnEpisode:
    """Track one incoming serve and the first return without assigning rewards.

    ``credit`` accumulates newly earned events until ``clear_credit`` is called
    at the policy-step boundary, so events from early physics substeps survive.
    Geometry defaults match the current Panthera scene, in metres.
    """

    table_half_length: float = 1.37
    table_half_width: float = 0.7625
    table_z: float = 0.76
    ball_radius: float = 0.02
    net_top_z: float = 0.9125
    net_half_width: float = 0.79
    net_half_thickness: float = 0.006
    net_contact_tolerance: float = 0.006
    require_opponent_bounce: bool = True
    opponent_bounced: bool = field(default=False, init=False)
    receiver_bounced: bool = field(default=False, init=False)
    hit: bool = field(default=False, init=False)
    crossed_net: bool = field(default=False, init=False)
    net_touched: bool = field(default=False, init=False)
    done: bool = field(default=False, init=False)
    success: bool = field(default=False, init=False)
    failure_reason: str | None = field(default=None, init=False)
    credit: set[str] = field(default_factory=set, init=False)
    hit_time: float | None = field(default=None, init=False)
    hit_position: tuple[float, float, float] | None = field(default=None, init=False)
    outgoing_time: float | None = field(default=None, init=False)
    outgoing_position: tuple[float, float, float] | None = field(default=None, init=False)
    outgoing_velocity: tuple[float, float, float] | None = field(default=None, init=False)
    crossing_time: float | None = field(default=None, init=False)
    landing_time: float | None = field(default=None, init=False)
    landing_position: tuple[float, float, float] | None = field(default=None, init=False)
    _crossing_valid: bool = field(default=False, init=False, repr=False)
    _previous_time: float | None = field(default=None, init=False, repr=False)
    _previous_position: tuple[float, float, float] | None = field(
        default=None, init=False, repr=False
    )
    _contacts: set[str] = field(default_factory=set, init=False, repr=False)

    def clear_credit(self) -> None:
        """Clear delivered event credits once per policy step."""
        self.credit.clear()

    def reset(self) -> None:
        """Reset episode history while retaining scene geometry."""
        self.opponent_bounced = False
        self.receiver_bounced = False
        self.hit = False
        self.crossed_net = False
        self.net_touched = False
        self.done = False
        self.success = False
        self.failure_reason = None
        self.credit.clear()
        self.hit_time = None
        self.hit_position = None
        self.outgoing_time = None
        self.outgoing_position = None
        self.outgoing_velocity = None
        self.crossing_time = None
        self.landing_time = None
        self.landing_position = None
        self._crossing_valid = False
        self._previous_time = None
        self._previous_position = None
        self._contacts.clear()

    def observe(
        self,
        timestamp: float,
        ball_position: Sequence[float],
        ball_velocity: Sequence[float],
        contacts: set[str],
    ) -> None:
        """Consume one physical substep; terminal outcomes remain latched.

        Table contacts are classified by the ball centre projected onto the
        playing surface. Continuous contact earns one event, even when several
        samples have the same label. Net contact alone does not end a return.
        """
        if self.done:
            return
        now = float(timestamp)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        if self._previous_time is not None and now <= self._previous_time:
            raise ValueError("timestamp must increase on every physical substep")
        position = _vector(ball_position, "ball_position")
        velocity = _vector(ball_velocity, "ball_velocity")
        if not isinstance(contacts, set) or not contacts <= _CONTACT_LABELS:
            raise ValueError(f"contacts must be a set of labels from {sorted(_CONTACT_LABELS)}")
        started = contacts - self._contacts
        previous = self._previous_position
        previous_time = self._previous_time
        self._previous_time = now
        self._previous_position = position
        self._contacts = contacts.copy()

        # Invalid robot/floor contact takes precedence over a simultaneous hit
        # or landing, preventing an invalid collision from earning success.
        if "robot" in contacts:
            self._fail("robot_contact")
            return
        if "floor" in contacts:
            self._fail("out_of_bounds" if self.hit else "missed_ball")
            return
        if "net" in contacts and self.hit:
            self.net_touched = True

        was_hit = self.hit
        if "paddle" in started:
            if self.hit:
                self._fail("double_hit")
                return
            if self.require_opponent_bounce and not self.opponent_bounced:
                self._fail("hit_before_opponent_bounce")
                return
            if not self.receiver_bounced:
                self._fail("hit_before_receiver_bounce")
                return
            self.hit = True
            self.hit_time = now
            self.hit_position = position
            self.credit.add("hit")

        # A segment ending at the first hit still belongs to the incoming
        # flight. Crossing detection begins with the next physical sample.
        if was_hit and previous is not None and previous[0] < 0.0 <= position[0]:
            if not self._valid_crossing_at(previous, position, 0.0):
                return
            self._crossing_valid = True

        # Merely reaching the mid-plane can still be a failed net collision.
        # Award passage only once the complete sphere has left the net slab.
        edge = self.net_half_thickness + self.ball_radius
        if was_hit and previous is not None and not self.crossed_net:
            if position[0] < 0.0:
                self._crossing_valid = False
            if self._crossing_valid and previous[0] < edge <= position[0]:
                if not self._valid_crossing_at(previous, position, edge):
                    return
                self.crossed_net = True
                fraction = (edge - previous[0]) / (position[0] - previous[0])
                assert previous_time is not None
                # Both observations are physical samples; interpolate the
                # boundary time rather than using the policy-step timestamp.
                self.crossing_time = previous_time + fraction * (now - previous_time)
                self.credit.add("crossed_net")

        if "table" in started:
            if not self._on_playing_surface(position):
                self._fail("out_of_bounds")
                return
            if self.hit:
                self.landing_time = now
                self.landing_position = position
                if position[0] <= 0.0:
                    self._fail("returned_to_receiver")
                elif not self.crossed_net:
                    self._fail("not_over_net")
                else:
                    self.done = True
                    self.success = True
                    self.credit.add("success")
                return
            if position[0] > 0.0:
                if self.opponent_bounced or self.receiver_bounced:
                    self._fail("invalid_serve_bounce")
                    return
                self.opponent_bounced = True
            elif position[0] < 0.0:
                if self.require_opponent_bounce and not self.opponent_bounced:
                    self._fail("receiver_bounce_before_opponent")
                    return
                if self.receiver_bounced:
                    self._fail("receiver_second_bounce")
                    return
                self.receiver_bounced = True
            else:
                self._fail("invalid_serve_bounce")
                return

        # Crossing a boundary above the table is not yet a failed return: the
        # incoming serve and valid strikes can both originate beyond an end
        # line. Resolve out-of-bounds once the ball falls below the surface.
        if position[2] < self.table_z and velocity[2] <= 0.0:
            outside = (
                abs(position[0]) > self.table_half_length + self.ball_radius
                or abs(position[1]) > self.table_half_width + self.ball_radius
            )
            if outside:
                self._fail("out_of_bounds" if self.hit else "missed_ball")

        # The impulse is still evolving during paddle contact. Sample the
        # first free-flight state, once, rather than the initial touch velocity.
        if not self.done and was_hit and not contacts and self.outgoing_time is None:
            self.outgoing_time = now
            self.outgoing_position = position
            self.outgoing_velocity = velocity
            self.credit.add("outgoing")

    def _valid_crossing_at(
        self,
        previous: tuple[float, float, float],
        position: tuple[float, float, float],
        plane_x: float,
    ) -> bool:
        fraction = (plane_x - previous[0]) / (position[0] - previous[0])
        crossing_y = previous[1] + fraction * (position[1] - previous[1])
        crossing_z = previous[2] + fraction * (position[2] - previous[2])
        if abs(crossing_y) > self.net_half_width - self.ball_radius:
            self._fail("around_net")
            return False
        tolerance = self.net_contact_tolerance if self.net_touched else 0.0
        if crossing_z < self.net_top_z + self.ball_radius - tolerance:
            self._fail("below_net")
            return False
        return True

    def _on_playing_surface(self, position: tuple[float, float, float]) -> bool:
        return (
            abs(position[0]) <= self.table_half_length
            and abs(position[1]) <= self.table_half_width
            # Exclude contacts on the table's vertical sides. The calibrated
            # sphere/table contact permits at most 6 mm of penetration.
            and position[2] >= self.table_z + self.ball_radius - 0.006
        )

    def _fail(self, reason: str) -> None:
        self.done = True
        self.failure_reason = reason
        self.credit.add("failure")

    def fail(self, reason: str) -> None:
        """Latch an externally detected failure once without replacing a result."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be a non-empty string")
        if not self.done:
            self._fail(reason)

    def finish_timeout(self) -> None:
        """Resolve an unfinished episode without replacing a decided outcome."""
        if not self.done:
            self._fail("timeout")


__all__ = ["ReturnEpisode"]
