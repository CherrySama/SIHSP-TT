"""Outgoing-shot feedback must guide flight without declaring success."""

from dataclasses import replace

import numpy as np
import pytest

from unilab.tasks.manipulation.table_tennis.return_events import ReturnEpisode
from unilab.tasks.manipulation.table_tennis.return_shaping import (
    ReturnShapingCfg,
    estimate_outgoing_shot,
)

CFG = ReturnShapingCfg(
    gravity=9.81,
    forward_speed=3.0,
    net_clearance=0.08,
    net_height_scale=0.18,
    landing_x_scale=0.75,
    landing_y_scale=0.25,
)
GEOMETRY = ReturnEpisode()
POSITION = (-1.45, 0.0, 1.05)


@pytest.mark.parametrize("vx", [-4.0, 0.0, 4.0])
def test_free_flight_prediction_matches_independent_time_integration(vx):
    velocity = np.array([vx, 0.1, 1.8])
    shot = estimate_outgoing_shot(POSITION, tuple(velocity), GEOMETRY, CFG)
    p = np.array(POSITION)
    dt = 1e-5
    while p[2] > GEOMETRY.table_z + GEOMETRY.ball_radius:
        p += velocity * dt + np.array([0.0, 0.0, -0.5 * CFG.gravity * dt * dt])
        velocity[2] -= CFG.gravity * dt
    np.testing.assert_allclose(shot.landing_position, p, atol=5e-5)
    if vx > 0:
        assert shot.landing == 1.0
    else:
        assert 0 < shot.landing < 1
    assert not GEOMETRY.hit and not GEOMETRY.success and not GEOMETRY.credit


def test_forward_speed_feedback_distinguishes_weak_and_strong_returns():
    weak = estimate_outgoing_shot(POSITION, (0.3, 0.0, 1.8), GEOMETRY, CFG)
    strong = estimate_outgoing_shot(POSITION, (4.0, 0.0, 1.8), GEOMETRY, CFG)
    assert 0 < weak.direction < strong.direction <= 1
    assert 0 < weak.landing < strong.landing == 1
    assert 0 < weak.net_height < strong.net_height <= 1
    assert weak.net_position is None


def test_net_quality_saturates_once_clearance_is_met():
    useful = estimate_outgoing_shot(POSITION, (4.0, 0.0, 1.8), GEOMETRY, CFG)
    high = estimate_outgoing_shot(POSITION, (4.0, 0.0, 7.0), GEOMETRY, CFG)
    assert useful.net_height == high.net_height == 1.0
    assert useful.landing > high.landing


def test_lateral_escape_and_end_line_overshoot_reduce_feedback():
    useful = estimate_outgoing_shot(POSITION, (4.0, 0.0, 1.8), GEOMETRY, CFG)
    lateral = estimate_outgoing_shot(POSITION, (4.0, 4.0, 1.8), GEOMETRY, CFG)
    overshoot = estimate_outgoing_shot(POSITION, (12.0, 0.0, 1.8), GEOMETRY, CFG)
    assert useful.direction > lateral.direction
    assert useful.landing > lateral.landing
    assert useful.net_height > lateral.net_height
    assert useful.landing > overshoot.landing


def test_backward_departure_retains_prediction_and_receives_negative_direction():
    backward = estimate_outgoing_shot(POSITION, (-3.0, 0.0, 1.8), GEOMETRY, CFG)
    stopped = estimate_outgoing_shot(POSITION, (0.0, 0.0, 1.8), GEOMETRY, CFG)
    assert backward.direction == -1.0
    assert stopped.direction == 0.0
    assert backward.landing_time == stopped.landing_time
    assert backward.landing_position[0] < stopped.landing_position[0] == POSITION[0]
    assert 0 < backward.landing < stopped.landing < 1
    assert 0 < backward.net_height < stopped.net_height < 1
    assert backward.net_position is stopped.net_position is None


@pytest.mark.parametrize("vy", [0.0, 0.2])
def test_direction_and_flight_scores_are_continuous_through_zero_forward_speed(vy):
    left = estimate_outgoing_shot(POSITION, (-1e-7, vy, 1.8), GEOMETRY, CFG)
    center = estimate_outgoing_shot(POSITION, (0.0, vy, 1.8), GEOMETRY, CFG)
    right = estimate_outgoing_shot(POSITION, (1e-7, vy, 1.8), GEOMETRY, CFG)
    assert left.direction < center.direction == 0 < right.direction
    for name in ("direction", "landing", "net_height"):
        assert abs(getattr(left, name) - getattr(right, name)) < 1e-6


@pytest.mark.parametrize("vx", [-12.0, -3.0, -0.3, 0.0, 0.3, 3.0, 12.0])
def test_forward_direction_saturates_at_signed_speed_scale(vx):
    shot = estimate_outgoing_shot(POSITION, (vx, 0.0, 1.8), GEOMETRY, CFG)
    assert shot.direction == pytest.approx(np.clip(vx / CFG.forward_speed, -1.0, 1.0))


def test_landing_and_unreached_net_quality_improve_continuously_with_forward_speed():
    shots = [
        estimate_outgoing_shot(POSITION, (vx, 0.0, 1.8), GEOMETRY, CFG)
        for vx in (-3.0, -1.0, 0.0, 0.3)
    ]
    assert all(shot.net_position is None for shot in shots)
    assert np.all(np.diff([shot.landing for shot in shots]) > 0)
    assert np.all(np.diff([shot.net_height for shot in shots]) > 0)


def test_net_proxy_meets_reachable_flight_score_at_first_landing_boundary():
    surface = GEOMETRY.table_z + GEOMETRY.ball_radius
    flight_time = (1.8 + np.sqrt(1.8**2 + 2 * CFG.gravity * (POSITION[2] - surface))) / CFG.gravity
    net_x = GEOMETRY.net_half_thickness + GEOMETRY.ball_radius
    threshold = (net_x - POSITION[0]) / flight_time
    before = estimate_outgoing_shot(POSITION, (threshold - 1e-7, 0.1, 1.8), GEOMETRY, CFG)
    after = estimate_outgoing_shot(POSITION, (threshold + 1e-7, 0.1, 1.8), GEOMETRY, CFG)
    assert before.net_position is None
    assert after.net_position is not None
    assert abs(before.net_height - after.net_height) < 1e-6
    assert before.net_height > 0


def test_reachable_net_clearance_deficit_improves_until_saturation():
    target_z = GEOMETRY.net_top_z + GEOMETRY.ball_radius + CFG.net_clearance
    net_x = GEOMETRY.net_half_thickness + GEOMETRY.ball_radius
    net_time = (net_x - POSITION[0]) / 4.0
    values = []
    for height in (target_z - 0.12, target_z - 0.03, target_z, target_z + 0.4):
        vz = (height - POSITION[2] + 0.5 * CFG.gravity * net_time**2) / net_time
        shot = estimate_outgoing_shot(POSITION, (4.0, 0.0, vz), GEOMETRY, CFG)
        assert shot.net_position is not None
        assert shot.net_position[2] == pytest.approx(height)
        values.append(shot.net_height)
    assert values[0] < values[1] < values[2] == values[3] == 1.0


def test_net_proxy_keeps_lateral_escape_penalty_before_reaching_net():
    centered = estimate_outgoing_shot(POSITION, (0.3, 0.0, 1.8), GEOMETRY, CFG)
    escaping = estimate_outgoing_shot(POSITION, (0.3, 4.0, 1.8), GEOMETRY, CFG)
    assert centered.net_position is escaping.net_position is None
    assert centered.net_height > escaping.net_height > 0


def test_descending_root_remains_positive_close_to_the_surface():
    surface = GEOMETRY.table_z + GEOMETRY.ball_radius
    shot = estimate_outgoing_shot((-1.45, 0.0, surface + 1e-12), (-3.0, 0.0, -100.0), GEOMETRY, CFG)
    assert shot.landing_time > 0
    assert shot.landing_time == pytest.approx(1e-14, rel=1e-4, abs=0)


@pytest.mark.parametrize("vz", [-2.0, 0.0, 2.0])
def test_initial_surface_height_selects_first_descending_crossing(vz):
    surface = GEOMETRY.table_z + GEOMETRY.ball_radius
    shot = estimate_outgoing_shot((-1.45, 0.0, surface), (-3.0, 0.0, vz), GEOMETRY, CFG)
    assert shot.landing_time == pytest.approx(max(2 * vz / CFG.gravity, 0.0))
    assert shot.landing_position[2] == surface


def test_no_future_table_crossing_does_not_remove_signed_direction():
    shot = estimate_outgoing_shot((-1.45, 0.0, 0.5), (-3.0, 0.0, -2.0), GEOMETRY, CFG)
    assert shot.direction == -1
    assert shot.landing_time is shot.landing_position is shot.net_position is None
    assert shot.landing == shot.net_height == 0


def test_scores_remain_bounded_across_forward_backward_and_lateral_departures():
    rng = np.random.default_rng(17)
    for velocity in rng.uniform((-12, -8, -4), (12, 8, 8), size=(250, 3)):
        shot = estimate_outgoing_shot(POSITION, tuple(velocity), GEOMETRY, CFG)
        assert -1 <= shot.direction <= 1
        assert 0 <= shot.net_height <= 1
        assert 0 <= shot.landing <= 1


def test_bad_samples_and_scales_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        estimate_outgoing_shot(POSITION, (float("nan"), 0.0, 1.0), GEOMETRY, CFG)
    for kwargs in ({"gravity": 0}, {"landing_x_scale": -1}, {"net_clearance": -0.1}):
        with pytest.raises(ValueError):
            replace(CFG, **kwargs)
