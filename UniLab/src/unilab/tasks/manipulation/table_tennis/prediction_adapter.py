"""Thin table-frame state adapter for the independent prediction core.

Requires the optional ``table-tennis`` dependency group. Simulation getters
are public Entity APIs; this module does not inspect a backend model/data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ball_prediction import BallPrediction, BallPredictionCfg, predict_ball
from ball_state import BallState

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


def panthera_prediction_cfg() -> BallPredictionCfg:
    """Offline effective fit for the current scene, not real-table parameters.

    Effective coefficients for the 0.25-ms table-contact model and slow
    no-initial-spin serves. Table friction can generate rotation after impact.
    A changed contact model, arbitrary timing, drag or imposed spin needs new
    calibration. These coefficients cannot be taken straight to hardware.
    """
    return BallPredictionCfg(
        normal_restitution=0.90,
        tangential_retention=0.947,
        allow_opponent_bounce=True,
        opponent_tangential_retention=0.94,
        opponent_contact_duration=0.005,
        horizon=1.5,
    )


def read_sim_ball_state(env: ManagerBasedRlEnv, env_id: int = 0) -> BallState:
    """Read the present ball state relative to this environment's table origin.

    Episode time resets with the scene, so the consumer must reset its event
    history too. Real adapters instead use their capture-clock timestamps.
    """
    if not 0 <= env_id < env.num_envs:
        raise IndexError("env_id is outside the simulation batch")
    ball = cast("Entity", env.scene["ball"])
    position = ball.data.root_link_pos_w[env_id] - env.scene.env_origins[env_id]
    velocity = ball.data.root_link_lin_vel_w[env_id]
    return BallState(
        timestamp=float(env.episode_length_buf[env_id] * env.step_dt),
        position=(float(position[0]), float(position[1]), float(position[2])),
        velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
    )


class PantheraBallPredictor:
    """Observe present/past snapshots and remember the first table rebound.

    The velocity sign transition near the table is observable by a real state
    estimator too. This is a truth-state adapter, not a noisy-vision filter.
    """

    def __init__(self, cfg: BallPredictionCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else panthera_prediction_cfg()
        self.reset()

    def reset(self) -> None:
        self._previous: BallState | None = None
        self._latest_timestamp: float | None = None
        self._bounce_count = 0
        self._opponent_bounce_count = 0

    @property
    def has_bounced(self) -> bool:
        return self._bounce_count > 0

    def update(self, state: BallState, *, now: float | None = None) -> BallPrediction:
        result = predict_ball(state, self.cfg, has_bounced=self.has_bounced, now=now)
        if result.reason == "future_timestamp":
            return result
        if self._latest_timestamp is not None and state.timestamp < self._latest_timestamp:
            return BallPrediction(
                source_timestamp=state.timestamp,
                trajectory=(),
                reason="out_of_order_measurement",
            )
        if state.timestamp == self._latest_timestamp:
            return BallPrediction(
                source_timestamp=state.timestamp, trajectory=(), reason="duplicate_measurement"
            )
        self._latest_timestamp = state.timestamp
        if not state.valid:
            # A rejected measurement must not erase the last usable velocity
            # sign, especially while the ball is occluded around a bounce.
            return result
        previous = self._previous
        observed_bounce = False
        if previous is not None:
            if (
                previous.velocity[2] < 0 < state.velocity[2]
                and state.position[2] < self.cfg.table_z + self.cfg.ball_radius + 0.08
            ):
                if state.position[0] < 0:
                    self._bounce_count += 1
                    observed_bounce = True
                elif self.cfg.allow_opponent_bounce:
                    self._opponent_bounce_count += 1
        self._previous = state
        if self._bounce_count > 1:
            return BallPrediction(
                source_timestamp=state.timestamp, trajectory=(), reason="observed_second_bounce"
            )
        if self._opponent_bounce_count > 1 or (
            self._opponent_bounce_count and result.opponent_bounce is not None
        ):
            return BallPrediction(
                source_timestamp=state.timestamp,
                trajectory=(),
                reason="repeated_opponent_bounce",
            )
        if observed_bounce:
            return predict_ball(state, self.cfg, has_bounced=self.has_bounced, now=now)
        return result
