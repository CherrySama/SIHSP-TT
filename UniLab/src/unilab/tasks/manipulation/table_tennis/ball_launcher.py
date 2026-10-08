"""Correlated incoming-ball generation in the Panthera table frame.

The landing window is a ball-trajectory constraint, not an arm-reachability
claim. Table restitution and the post-bounce flight are checked in MuJoCo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np

from unilab.envs.mdp.events import resolve_env_ids
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


@dataclass(frozen=True)
class BallLauncherCfg:
    """Retained direct-feed window landing 0.9--1.1 m behind the net.

    Positions are metres, speeds metres/second, gravity metres/second squared.
    Negative X is the robot half. Bounce time and horizontal speed jointly
    determine the bounce position and post-bounce height. The three default
    times retain the earlier feed interface. The default two-bounce service
    uses ServeLauncherCfg; changing either window needs physics validation.
    """

    lateral_range: tuple[float, float] = (-0.5, 0.5)
    speed_range: tuple[float, float] = (3.33, 3.59)
    bounce_times: tuple[float, ...] = (0.496, 0.504, 0.512)
    launch_x: float = 0.75
    launch_z: float = 0.95
    net_clearance: float = 0.01

    # Constants belong to the fixed Panthera scene, not sampling overrides.
    table_z: ClassVar[float] = 0.76
    ball_radius: ClassVar[float] = 0.02
    gravity: ClassVar[float] = 9.81
    net_top_z: ClassVar[float] = 0.9125
    net_half_thickness: ClassVar[float] = 0.006
    table_half_length: ClassVar[float] = 1.37
    table_half_width: ClassVar[float] = 0.7625

    def __post_init__(self) -> None:
        for name in (
            "launch_x",
            "launch_z",
            "net_clearance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"BallLauncherCfg {name} must be a finite real number")
        for name in ("lateral_range", "speed_range"):
            bounds = getattr(self, name)
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"BallLauncherCfg {name} must be a (min, max) tuple")
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    for value in bounds
                )
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"BallLauncherCfg {name} must contain finite ordered bounds")
        if self.net_clearance < 0:
            raise ValueError("BallLauncherCfg net clearance must be non-negative")
        if not isinstance(self.bounce_times, tuple) or not self.bounce_times:
            raise ValueError("BallLauncherCfg bounce_times must be a non-empty tuple")
        if any(
            isinstance(time, bool)
            or not isinstance(time, Real)
            or not math.isfinite(time)
            or time <= 0.0
            for time in self.bounce_times
        ):
            raise ValueError("BallLauncherCfg bounce_times must be finite and positive")
        net_edge = self.net_half_thickness + self.ball_radius
        if not net_edge < self.launch_x < self.table_half_length - self.ball_radius:
            raise ValueError("BallLauncherCfg launch_x must be inside the opponent half")
        if self.launch_z <= self.table_z + self.ball_radius:
            raise ValueError("BallLauncherCfg launch_z must be above the playing surface")
        if self.speed_range[0] <= 0:
            raise ValueError("BallLauncherCfg speed_range must be positive")
        if (
            max(abs(value) for value in self.lateral_range)
            >= self.table_half_width - self.ball_radius
        ):
            raise ValueError("BallLauncherCfg lateral_range must keep the ball inside the table")

        # Height is concave in time. For each flight-time level its minimum
        # over the speed range and expanded net slab is at the endpoints.
        # Check the complete window so reset needs no rejection sampling.
        for speed in self.speed_range:
            for bounce_time in self.bounce_times:
                bounce_x = self.launch_x - speed * bounce_time
                if not -self.table_half_length + self.ball_radius < bounce_x < -net_edge:
                    raise ValueError(
                        "BallLauncherCfg bounce position must be inside the robot half"
                    )
                vertical_speed = _launch_vertical_speed(self, bounce_time)
                for x in (net_edge, -net_edge):
                    time = (self.launch_x - x) / speed
                    z = self.launch_z + vertical_speed * time - 0.5 * self.gravity * time**2
                    if z < self.net_top_z + self.ball_radius + self.net_clearance:
                        raise ValueError("BallLauncherCfg feed window does not clear the net")


@dataclass(frozen=True)
class BallLaunch:
    """One no-spin initial state and its pre-contact ballistic landing."""

    position: tuple[float, float, float]
    linear_velocity: tuple[float, float, float]
    bounce_position: tuple[float, float, float]
    bounce_time: float


@dataclass(frozen=True)
class ServeLauncherCfg:
    """Slow no-initial-spin service: opponent bounce, net, receiver bounce.

    Receiver windows are jointly sampled design targets. The MuJoCo tests
    check actual contacts, heights and rail preparation independently.
    """

    lateral_range: tuple[float, float] = (-0.5, 0.5)
    speed_range: tuple[float, float] = (2.8, 3.0)
    receiver_bounce_range: tuple[float, float] = (-0.96, -0.91)
    receiver_peak_range: tuple[float, float] = (0.30, 0.34)
    launch_x: float = 1.42
    normal_restitution: float = 0.90
    opponent_tangential_retention: float = 0.94
    net_clearance: float = 0.03

    def __post_init__(self) -> None:
        for name in (
            "lateral_range",
            "speed_range",
            "receiver_bounce_range",
            "receiver_peak_range",
        ):
            bounds = getattr(self, name)
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(
                    isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                    for v in bounds
                )
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"ServeLauncherCfg {name} must contain finite ordered bounds")
        for name in (
            "launch_x",
            "normal_restitution",
            "opponent_tangential_retention",
            "net_clearance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"ServeLauncherCfg {name} must be finite")
        if self.launch_x < BallLauncherCfg.table_half_length + BallLauncherCfg.ball_radius:
            raise ValueError("Serve launch must be behind the opponent end line")
        if self.speed_range[0] <= 0 or self.net_clearance < 0:
            raise ValueError("Serve speed must be positive and net clearance non-negative")
        for coefficient in (self.normal_restitution, self.opponent_tangential_retention):
            if not 0 < coefficient <= 1:
                raise ValueError("Serve bounce coefficients must be in (0, 1]")
        if max(abs(v) for v in self.lateral_range) >= 0.7425:
            raise ValueError("Serve lateral window must fit inside the table")
        if not -1.35 < self.receiver_bounce_range[0] <= self.receiver_bounce_range[1] < -0.026:
            raise ValueError("Serve receiver bounce window must fit inside the robot half")
        if self.receiver_peak_range[0] <= BallLauncherCfg.ball_radius:
            raise ValueError("Serve receiver peak must be above the ball contact plane")
        # Verify the complete small Cartesian target window before sampling.
        # All values are coupled by the two-flight solution below.
        for speed in self.speed_range:
            for bounce_x in self.receiver_bounce_range:
                for peak in self.receiver_peak_range:
                    launch = _serve_from_targets(self, 0.0, speed, bounce_x, peak)
                    first_x = launch.bounce_position[0]
                    if not 0.026 < first_x < 1.35:
                        raise ValueError("Serve opponent bounce must fit inside the opponent half")
                    if launch.position[2] <= 0.78 or launch.bounce_time <= 0:
                        raise ValueError(
                            "Serve initial state must be above the table before impact"
                        )
                    up = -self.normal_restitution * (
                        launch.linear_velocity[2] - 9.81 * launch.bounce_time
                    )
                    horizontal = self.opponent_tangential_retention * speed
                    for x in (-0.026, 0.026):
                        t = (first_x - x) / horizontal
                        z = 0.78 + up * t - 0.5 * 9.81 * t**2
                        if z < 0.9325 + self.net_clearance:
                            raise ValueError("Serve target window does not clear the net")


def _serve_from_targets(
    cfg: ServeLauncherCfg, lateral: float, speed: float, receiver_x: float, peak: float
) -> BallLaunch:
    """Backsolve the two free flights, with no imposed initial rotation.

    Peak describes ball-centre height above the table after the receiver
    bounce. Coefficients are effective fits for the current simulation only.
    """
    gravity = BallLauncherCfg.gravity
    plane = BallLauncherCfg.table_z + BallLauncherCfg.ball_radius
    receiver_incident = math.sqrt(2 * gravity * (peak - BallLauncherCfg.ball_radius)) / (
        cfg.normal_restitution
    )
    between_time = 2 * receiver_incident / gravity
    first_x = receiver_x + cfg.opponent_tangential_retention * speed * between_time
    time = (cfg.launch_x - first_x) / speed
    first_incident = receiver_incident / cfg.normal_restitution
    launch_z = plane + first_incident * time - 0.5 * gravity * time**2
    return BallLaunch(
        position=(cfg.launch_x, lateral, launch_z),
        linear_velocity=(-speed, 0.0, gravity * time - first_incident),
        bounce_position=(first_x, lateral, plane),
        bounce_time=time,
    )


def _launch_vertical_speed(cfg: BallLauncherCfg, time: float) -> float:
    # Solve z(t) = table_z + radius with vx=-speed, rather than drawing
    # an unrelated vz that could bounce before the net or miss the table.
    vertical_speed = (
        cfg.table_z + cfg.ball_radius - cfg.launch_z + 0.5 * cfg.gravity * time**2
    ) / time
    return vertical_speed


def sample_ball_launch(
    rng: np.random.Generator, cfg: BallLauncherCfg | ServeLauncherCfg
) -> BallLaunch:
    """Draw a legal ballistic approach without imposed spin or lateral drift."""
    lateral = float(rng.uniform(*cfg.lateral_range))
    speed = float(rng.uniform(*cfg.speed_range))
    if isinstance(cfg, ServeLauncherCfg):
        receiver_x = float(rng.uniform(*cfg.receiver_bounce_range))
        peak = float(rng.uniform(*cfg.receiver_peak_range))
        return _serve_from_targets(cfg, lateral, speed, receiver_x, peak)
    time = float(rng.choice(cfg.bounce_times))
    bounce_x = cfg.launch_x - speed * time
    vertical_speed = _launch_vertical_speed(cfg, time)
    return BallLaunch(
        position=(cfg.launch_x, lateral, cfg.launch_z),
        linear_velocity=(-speed, 0.0, vertical_speed),
        bounce_position=(bounce_x, lateral, cfg.table_z + cfg.ball_radius),
        bounce_time=time,
    )


def launch_random_ball(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    launcher_cfg: BallLauncherCfg | ServeLauncherCfg,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Stage random ball states through the existing reset transaction."""
    ids = resolve_env_ids(env, env_ids)
    asset = cast("Entity", env.scene[asset_cfg.name])
    root_states = np.array(asset.data.default_root_state[ids], copy=True)
    for row, env_id in enumerate(ids):
        launch = sample_ball_launch(env.rng, launcher_cfg)
        root_states[row, :3] = launch.position
        root_states[row, :3] += env.scene.env_origins[env_id]
        root_states[row, 7:10] = launch.linear_velocity
        root_states[row, 10:13] = 0.0
    asset.write_root_state_to_sim(root_states, env_ids=ids)
