"""Ball states shared by simulation and real-world state estimation.

Coordinates are in the table frame: X points towards the opponent, Y points
left when looking towards the opponent, and Z points up. Positions and
velocities describe the ball centre in metres and metres/second. Timestamps
are capture times in seconds on the same clock used by the consuming planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class BallState:
    """A snapshot, or a predicted state explicitly contained in a forecast.

    A real-world producer must first transform its measured position/velocity
    into the table frame. Capture time must not be replaced by arrival time.
    Invalid measurements carry ``valid=False``; they never produce a target.
    """

    timestamp: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    valid: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, Real)
            or not math.isfinite(self.timestamp)
        ):
            raise ValueError("BallState timestamp must be finite seconds")
        if not isinstance(self.valid, bool):
            raise ValueError("BallState valid must be a boolean")
        object.__setattr__(self, "timestamp", float(self.timestamp))
        for name in ("position", "velocity"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"BallState {name} must contain three finite values")
            # Copy into an immutable tuple so mutable sensor buffers cannot
            # alter a state after it has been given to the predictor.
            object.__setattr__(self, name, tuple(float(component) for component in value))
