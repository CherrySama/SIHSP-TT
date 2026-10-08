"""Independent physics, time and validity contracts of the shared predictor."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ball_prediction import BallPredictionCfg, predict_ball
from ball_state import BallState


_CFG = BallPredictionCfg(normal_restitution=0.935, tangential_retention=0.86625)
_INCOMING = BallState(10.0, (0.75, 0.2, 0.95), (-3.46, 0.0, 2.134))


def test_state_copies_mutable_sensor_buffers() -> None:
    position = np.array([0.75, 0.2, 0.95])
    velocity = np.array([-3.46, 0.0, 2.134])
    state = BallState(10.0, position, velocity)  # type: ignore[arg-type]
    position[:] = 0
    velocity[:] = 0
    assert state.position == _INCOMING.position
    assert state.velocity == _INCOMING.velocity


def test_free_flight_after_observed_bounce_keeps_capture_clock() -> None:
    state = BallState(100.0, (-1.1, 0.2, 1.0), (-2.5, -0.1, 1.0))
    result = predict_ball(state, _CFG, has_bounced=True)
    assert result.valid and result.candidate is not None
    duration = 0.29 / 2.5
    assert result.candidate.timestamp == pytest.approx(100.0 + duration)
    assert result.candidate.position == pytest.approx(
        (-1.39, 0.2 - 0.1 * duration, 1.0 + duration - 4.905 * duration**2)
    )
    assert result.first_bounce is None
    assert result.net_crossing is None


def test_forecast_contains_one_legal_bounce_and_a_future_candidate() -> None:
    result = predict_ball(_INCOMING, _CFG)
    assert result.valid
    assert result.net_crossing is not None
    assert result.first_bounce is not None
    assert result.candidate is not None
    assert 10.0 < result.net_crossing.timestamp < result.first_bounce.timestamp
    assert result.first_bounce.timestamp < result.candidate.timestamp
    assert result.first_bounce.position[2] == pytest.approx(0.78)
    assert -1.1 < result.first_bounce.position[0] < -0.9
    assert result.first_bounce.velocity[2] < 0
    assert result.candidate.velocity[2] > 0
    assert result.candidate.position[0] == pytest.approx(-1.39)
    times = np.array([point.timestamp for point in result.trajectory])
    assert np.all(np.diff(times) > 0)
    assert result.trajectory[-1] == result.candidate


def test_latency_advances_forecast_without_relabelling_the_measurement() -> None:
    initial = predict_ball(_INCOMING, _CFG)
    delayed = predict_ball(_INCOMING, _CFG, now=10.1)
    assert delayed.valid and delayed.candidate == initial.candidate
    assert delayed.source_timestamp == 10.0
    assert delayed.trajectory[0].timestamp == pytest.approx(10.1)
    assert delayed.trajectory[0].position != _INCOMING.position
    assert all(point.timestamp >= 10.1 for point in delayed.trajectory)
    assert predict_ball(_INCOMING, _CFG, now=11.0).reason == "candidate_already_passed"
    assert predict_ball(_INCOMING, _CFG, now=9.9).reason == "future_timestamp"


def test_latency_crossing_a_bounce_uses_rebound_motion() -> None:
    result = predict_ball(_INCOMING, _CFG, now=10.55)
    assert result.valid
    assert result.trajectory[0].timestamp == pytest.approx(10.55)
    assert result.trajectory[0].velocity[2] > 0
    assert result.trajectory[0].velocity[0] == pytest.approx(-3.46 * 0.86625)


@pytest.mark.parametrize(
    "state, reason",
    [
        (replace(_INCOMING, valid=False), "invalid_measurement"),
        (replace(_INCOMING, velocity=(3.0, 0.0, 1.0)), "not_incoming"),
        (replace(_INCOMING, position=(-1.4, 0.0, 1.0)), "candidate_already_passed"),
        (replace(_INCOMING, velocity=(-3.46, 0.0, -1.0)), "net_collision"),
        (replace(_INCOMING, position=(0.75, 0.8, 0.95)), "outside_net_playing_width"),
        (BallState(10.0, (-0.95, 0.0, 0.783), (-3.0, 0.0, 2.4)), "contact_transition"),
    ],
)
def test_invalid_or_unreliable_states_never_return_a_target(state, reason) -> None:
    result = predict_ball(state, _CFG)
    assert not result.valid and result.reason == reason
    assert result.candidate is None and result.trajectory == ()


def test_outside_table_landing_and_early_candidate_are_rejected() -> None:
    assert predict_ball(_INCOMING, replace(_CFG, candidate_x=-0.2)).reason == (
        "candidate_before_first_bounce"
    )
    beyond_end = BallState(10.0, (-1.2, 0.0, 1.0), (-3.0, 0.0, -1.0))
    assert predict_ball(beyond_end, _CFG).reason == "first_bounce_outside_robot_half"


@pytest.mark.parametrize(
    "overrides",
    [
        {"normal_restitution": 1.1},
        {"tangential_retention": 0.0},
        {"gravity": 0.0},
        {"candidate_x": 0.1},
        {"horizon": float("nan")},
        {"sample_dt": 2.0},
    ],
)
def test_invalid_physical_parameters_fail_at_configuration(overrides) -> None:
    with pytest.raises(ValueError):
        replace(_CFG, **overrides)


def test_core_import_does_not_load_simulation_or_learning_frameworks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ball_prediction, sys; "
            "assert not {'unilab', 'mujoco', 'torch', 'unisim'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _service():
    cfg = BallPredictionCfg(
        normal_restitution=0.90,
        tangential_retention=0.947,
        allow_opponent_bounce=True,
        opponent_tangential_retention=0.94,
        opponent_contact_duration=0.005,
        candidate_x=-1.24,
        horizon=1.5,
    )
    time = (1.42 - 0.6) / 2.9
    state = BallState(
        10.0,
        (1.42, 0.2, 0.78 + 3.0 * time - 4.905 * time * time),
        (-2.9, 0.0, 9.81 * time - 3.0),
    )
    return state, cfg


def test_service_events_are_ordered_and_history_is_receiver_specific():
    state, cfg = _service()
    result = predict_ball(state, cfg)
    assert result.valid
    assert result.opponent_bounce is not None and result.opponent_bounce.position[0] > 0
    assert result.net_crossing is not None and result.first_bounce is not None
    assert result.candidate is not None and result.first_bounce.position[0] < 0
    assert (
        result.opponent_bounce.timestamp
        < result.net_crossing.timestamp
        < result.first_bounce.timestamp
        < result.candidate.timestamp
    )
    assert all(
        a.timestamp < b.timestamp for a, b in zip(result.trajectory, result.trajectory[1:])
    )
    disabled = predict_ball(state, replace(cfg, allow_opponent_bounce=False))
    assert not disabled.valid and disabled.candidate is None
    assert predict_ball(state, cfg, has_bounced=True).reason == "inconsistent_bounce_history"


def test_service_latency_uses_each_flight_and_rejects_contact_interval():
    state, cfg = _service()
    result = predict_ball(state, cfg)
    assert result.opponent_bounce is not None and result.first_bounce is not None
    assert result.candidate is not None
    first = result.opponent_bounce
    assert predict_ball(state, cfg, now=first.timestamp + 0.002).reason == "contact_transition"
    for now, retention in (
        (first.timestamp + 0.1, 0.94),
        (result.first_bounce.timestamp + 0.05, 0.94 * 0.947),
    ):
        delayed = predict_ball(state, cfg, now=now)
        assert delayed.valid and delayed.candidate == result.candidate
        assert delayed.trajectory[0].timestamp == pytest.approx(now)
        assert delayed.trajectory[0].velocity[0] == pytest.approx(-2.9 * retention)
    assert (
        predict_ball(state, cfg, now=result.first_bounce.timestamp).reason
        == "contact_transition"
    )
    assert (
        predict_ball(state, cfg, now=result.candidate.timestamp).reason
        == "candidate_already_passed"
    )


def test_service_contact_displacement_cannot_skip_the_net():
    _, cfg = _service()
    state = BallState(0.0, (0.34, 0.0, 0.8976166667), (-3.0, 0.0, -0.6856666667))
    result = predict_ball(state, replace(cfg, opponent_contact_duration=0.05))
    assert not result.valid
    assert result.reason == "net_collision"
    assert result.candidate is None
