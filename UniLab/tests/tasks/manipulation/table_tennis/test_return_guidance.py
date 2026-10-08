"""Incoming-window forecasts and pre-contact guidance are bounded pure computations."""

from dataclasses import replace

import numpy as np
import pytest

from unilab.tasks.manipulation.table_tennis.return_guidance import (
    ReturnGuidanceCfg,
    flight_window,
    guidance_scores,
)

CFG = ReturnGuidanceCfg()
SURFACE = 0.78
PADDLE = np.array([-1.55, 0.0, 1.05])
LOWER = np.array([-3.0, -0.5, SURFACE])
UPPER = np.array([-0.5, 0.5, 1.5])


def _window(position=PADDLE, *, time=0.25, velocity=(-2.9, 0.0, 1.0)):
    return np.array([[time, *position, *velocity]], dtype=np.float64)


def _scores(window, **kwargs):
    defaults = {
        "now": 0.0,
        "paddle": PADDLE,
        "normal": np.array([1.0, 0.0, 0.0]),
        "paddle_velocity": np.array([CFG.velocity_scale, 0.0, 0.0]),
        "receiver_bounced": True,
        "cfg": CFG,
    }
    return guidance_scores(window, **(defaults | kwargs))


def test_forecast_keeps_future_window_after_old_candidate_section():
    position = np.array([-1.60, 0.0, 1.05])
    window = flight_window(
        position,
        np.array([-2.5, 0.0, -0.3]),
        timestamp=7.0,
        now=7.0,
        surface=SURFACE,
        lower=LOWER,
        upper=UPPER,
        cfg=CFG,
    )
    assert window.shape[1] == 7 and len(window) > 0
    assert np.all(window[:, 1] < -1.53)
    assert np.all(window[:, 0] > 7.0)


def test_window_samples_only_future_free_flight_before_first_descending_landing():
    position = np.array([-0.94, 0.0, SURFACE])
    velocity = np.array([-2.75, 0.0, 2.3])
    timestamp, now = 1.0, 1.1
    window = flight_window(
        position,
        velocity,
        timestamp=timestamp,
        now=now,
        surface=SURFACE,
        lower=LOWER,
        upper=UPPER,
        cfg=CFG,
    )
    first_landing = timestamp + 2 * velocity[2] / CFG.gravity
    assert len(window) > 0
    assert np.all((window[:, 0] > now) & (window[:, 0] < first_landing))
    assert np.all(window[:, 3] > SURFACE)
    times = window[:, 0] - timestamp
    expected_positions = position + times[:, None] * velocity
    expected_positions[:, 2] -= 0.5 * CFG.gravity * times**2
    np.testing.assert_allclose(window[:, 1:4], expected_positions, atol=1e-12)
    np.testing.assert_allclose(window[:, 6], velocity[2] - CFG.gravity * times, atol=1e-12)


def test_predicted_rebound_origin_may_be_future_and_respects_now_based_horizon():
    cfg = replace(CFG, horizon=0.15)
    window = flight_window(
        np.array([-0.94, 0.0, SURFACE]),
        np.array([-2.75, 0.0, 2.3]),
        timestamp=1.1,
        now=1.0,
        surface=SURFACE,
        lower=LOWER,
        upper=UPPER,
        cfg=cfg,
    )
    assert len(window) > 0
    assert np.all((window[:, 0] > 1.0) & (window[:, 0] < 1.15))


def test_workspace_envelope_filters_samples_without_claiming_ik_reachability():
    lower, upper = np.array([-1.85, -0.05, 0.9]), np.array([-1.7, 0.05, 1.2])
    window = flight_window(
        PADDLE,
        np.array([-2.5, 0.0, 0.2]),
        timestamp=0.0,
        now=0.0,
        surface=SURFACE,
        lower=lower,
        upper=upper,
        cfg=CFG,
    )
    assert len(window) > 0
    assert np.all((window[:, 1:4] >= lower) & (window[:, 1:4] <= upper))


@pytest.mark.parametrize("now", [0.5, 1.0])
def test_window_expires_at_first_landing_without_predicting_another_bounce(now):
    window = flight_window(
        PADDLE,
        np.array([-2.75, 0.0, 1.0]),
        timestamp=0.0,
        now=now,
        surface=SURFACE,
        lower=LOWER,
        upper=UPPER,
        cfg=CFG,
    )
    assert window.shape == (0, 7)


@pytest.mark.parametrize("time", [-0.1, 0.0])
def test_past_or_current_target_is_not_a_future_guidance_sample(time):
    reach, strike, target = _scores(_window(time=time))
    assert reach == strike == 0.0
    assert target is None


def test_empty_window_returns_no_guidance():
    assert _scores(np.empty((0, 7))) == (0.0, 0.0, None)


@pytest.mark.parametrize("incoming_speed", [-0.5, -3.0, -12.0])
def test_stationary_paddle_never_earns_strike_from_ball_approach_speed(incoming_speed):
    _, strike, _ = _scores(
        _window(time=0.08, velocity=(incoming_speed, 0.0, 1.0)),
        paddle_velocity=np.zeros(3),
    )
    assert strike == 0.0


@pytest.mark.parametrize("velocity", [(-1.0, 0.0, 0.0), (0.0, 3.0, 2.0)])
def test_backward_or_sideways_paddle_velocity_earns_no_forward_strike(velocity):
    assert _scores(_window(time=0.08), paddle_velocity=np.array(velocity))[1] == 0.0


def test_swing_before_legal_receiver_bounce_has_no_strike_score():
    assert _scores(_window(time=0.08), receiver_bounced=False)[1] == 0.0


@pytest.mark.parametrize("offset", [(0.0, 0.4, 0.0), (0.4, 0.0, 0.0)])
def test_finite_face_radial_or_axial_misalignment_suppresses_fast_air_swing(offset):
    aligned = _scores(_window(time=0.08))[1]
    misplaced = _scores(_window(PADDLE + np.array(offset), time=0.08))[1]
    assert aligned == 1.0
    assert 0 <= misplaced < 1e-4


def test_near_plane_outside_finite_paddle_disc_is_not_treated_as_contact():
    close = _scores(_window(PADDLE + [0.0, 0.05, 0.0], time=0.08))[1]
    far = _scores(_window(PADDLE + [0.0, 0.3, 0.0], time=0.08))[1]
    assert close == 1.0
    assert far < 0.01


def test_normal_magnitude_and_sign_do_not_change_finite_face_geometry():
    window = _window(PADDLE + [0.04, 0.06, 0.0], time=0.08)
    baseline = _scores(window)[1]
    assert _scores(window, normal=np.array([7.0, 0.0, 0.0]))[1] == pytest.approx(baseline)
    assert _scores(window, normal=np.array([-1.0, 0.0, 0.0]))[1] == pytest.approx(baseline)


@pytest.mark.parametrize("offset", [0.0, 0.01, 0.02])
def test_reach_saturates_inside_position_tolerance(offset):
    assert _scores(_window(PADDLE + [0.0, offset, 0.0]))[0] == pytest.approx(1.0)


def test_reach_uses_nearest_future_point_once_and_returns_an_independent_target():
    closest = _window(PADDLE + [0.0, 0.01, 0.0])
    other = _window(PADDLE + [0.0, 0.3, 0.0], time=0.3)
    window = np.vstack((other, closest, closest, closest))
    reach, _, target = _scores(window)
    assert reach == pytest.approx(1.0)
    np.testing.assert_array_equal(target, closest[0])
    target[1] += 1.0
    np.testing.assert_array_equal(window[1], closest[0])


@pytest.mark.parametrize(
    ("time", "expected"), [(0.25, 1.0), (0.16, 1.0), (0.11, 0.5), (0.06, 0.0), (0.01, 0.0)]
)
def test_reach_fades_out_near_contact(time, expected):
    assert _scores(_window(time=time))[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("time", "expected"), [(0.3, 0.0), (0.2, 0.0), (0.15, 0.5), (0.1, 1.0), (0.01, 1.0)]
)
def test_strike_time_gate_only_rewards_near_encounters(time, expected):
    assert _scores(_window(time=time))[1] == pytest.approx(expected)


@pytest.mark.parametrize("speed", [CFG.velocity_scale, 2 * CFG.velocity_scale, 100.0])
def test_strike_speed_score_saturates(speed):
    assert _scores(_window(time=0.08), paddle_velocity=np.array([speed, 0.0, 0.0]))[1] == 1.0


def test_scores_are_unitless_and_bounded():
    rng = np.random.default_rng(23)
    for _ in range(100):
        window = _window(PADDLE + rng.uniform(-0.3, 0.3, 3), time=rng.uniform(0.001, 0.5))
        reach, strike, _ = _scores(window, paddle_velocity=rng.uniform(-3.0, 3.0, 3))
        assert 0.0 <= reach <= 1.0
        assert 0.0 <= strike <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gravity": 0.0},
        {"distance_scale": float("nan")},
        {"freshness": True},
        {"fade_near": CFG.fade_far},
        {"strike_far": CFG.strike_near},
        {"sample_dt": 2 * CFG.horizon},
    ],
)
def test_invalid_guidance_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        replace(CFG, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"normal": np.zeros(3)},
        {"paddle": np.zeros(2)},
        {"paddle_velocity": np.array([float("nan"), 0.0, 0.0])},
        {"now": float("inf")},
        {"receiver_bounced": "yes"},
    ],
)
def test_invalid_paddle_states_and_event_flags_fail_closed(kwargs):
    with pytest.raises(ValueError):
        _scores(_window(), **kwargs)


def test_invalid_forecast_workspace_bounds_fail_closed():
    for lower, upper in (
        (np.zeros(2), UPPER),
        (np.array([float("nan"), 0, 0]), UPPER),
        (UPPER, LOWER),
    ):
        with pytest.raises(ValueError, match="bounds"):
            flight_window(
                PADDLE,
                np.array([-2.75, 0.0, 1.0]),
                timestamp=0.0,
                now=0.0,
                surface=SURFACE,
                lower=lower,
                upper=upper,
                cfg=CFG,
            )
