"""Safety and lateral-target contracts without a simulator."""

from __future__ import annotations

import math

import pytest

from rail_alignment import RailAlignmentCfg, RailAlignmentController


def test_target_compensates_measured_paddle_offset_and_is_lateral_only() -> None:
    controller = RailAlignmentController()
    target = controller.observe(rail_position=0.1, paddle_y=0.16, ball_y=-0.3, arrival_time=0.7)
    assert target.position == pytest.approx(-0.36)
    assert target.within_travel
    assert not controller.ready(now=0.1, paddle_y=0.16, velocity=0.0)
    assert controller.ready(now=0.6, paddle_y=-0.3, velocity=0.04)
    assert not controller.ready(now=0.66, paddle_y=-0.3, velocity=0.0)
    assert not controller.ready(now=0.6, paddle_y=-0.3, velocity=0.06)


def test_outside_travel_target_is_reported_and_never_ready() -> None:
    controller = RailAlignmentController()
    target = controller.observe(rail_position=0.0, paddle_y=0.0, ball_y=0.9, arrival_time=1.0)
    assert target.position == 0.75
    assert not target.within_travel
    assert not controller.ready(now=0.1, paddle_y=0.9, velocity=0.0)


@pytest.mark.parametrize("target", [-0.75, -0.5, 0.5, 0.75])
def test_physical_integration_brakes_within_velocity_acceleration_and_travel(target) -> None:
    controller = RailAlignmentController()
    controller.observe(rail_position=0.0, paddle_y=0.0, ball_y=target, arrival_time=2.0)
    dt = 0.002
    position = velocity = 0.0
    for step in range(1000):
        acceleration = controller.acceleration(
            now=step * dt, position=position, velocity=velocity, dt=dt
        )
        assert abs(acceleration) <= controller.cfg.max_acceleration
        position += velocity * dt + 0.5 * acceleration * dt**2
        velocity += acceleration * dt
        assert abs(velocity) <= 1.5 + 1e-12
        assert -0.75 <= position <= 0.75
    assert position == pytest.approx(target, abs=0.001)
    assert abs(velocity) < 0.05


def test_absent_expired_and_reset_targets_stop_instead_of_chasing_old_positions() -> None:
    controller = RailAlignmentController()
    assert (
        controller.acceleration(now=0.0, position=0.2, velocity=0.5, dt=0.01)
        == -controller.cfg.max_acceleration
    )
    controller.observe(rail_position=0.0, paddle_y=0.0, ball_y=0.5, arrival_time=0.7)
    assert (
        controller.acceleration(now=0.71, position=0.2, velocity=0.5, dt=0.01)
        == -controller.cfg.max_acceleration
    )
    controller.reset()
    assert controller.target is None
    assert controller.acceleration(now=0.0, position=0.2, velocity=0.0, dt=0.01) == 0.0


def test_invalid_parameters_and_measurements_fail_before_control() -> None:
    with pytest.raises(ValueError):
        RailAlignmentCfg(max_acceleration=math.nan)
    with pytest.raises(ValueError):
        RailAlignmentCfg(lower=1.0, upper=0.0)
    with pytest.raises(ValueError):
        RailAlignmentCfg(max_force=0.0)
    controller = RailAlignmentController()
    with pytest.raises(ValueError):
        controller.observe(rail_position=0.0, paddle_y=math.inf, ball_y=0.1, arrival_time=0.7)
    assert controller.target is None
    with pytest.raises(ValueError):
        controller.acceleration(now=0.0, position=0.0, velocity=0.0, dt=0.0)
