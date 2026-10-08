"""Geometry and reproducibility of the correlated Panthera incoming feed."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.tasks.manipulation.table_tennis.ball_launcher import (
    BallLauncherCfg,
    ServeLauncherCfg,
    sample_ball_launch,
)


def test_launches_are_reproducible_and_satisfy_correlated_bounce_constraint() -> None:
    cfg = BallLauncherCfg()
    rng = np.random.default_rng(19)
    replay = np.random.default_rng(19)
    launches = [sample_ball_launch(rng, cfg) for _ in range(128)]
    assert launches == [sample_ball_launch(replay, cfg) for _ in range(128)]
    other_rng = np.random.default_rng(20)
    assert launches != [sample_ball_launch(other_rng, cfg) for _ in launches]

    for launch in launches:
        position = np.asarray(launch.position)
        velocity = np.asarray(launch.linear_velocity)
        landing = position + velocity * launch.bounce_time
        landing[2] -= 0.5 * cfg.gravity * launch.bounce_time**2
        np.testing.assert_allclose(landing, launch.bounce_position, atol=1e-12)
        assert cfg.lateral_range[0] <= position[1] <= cfg.lateral_range[1]
        assert cfg.speed_range[0] <= -velocity[0] <= cfg.speed_range[1]
        assert -cfg.table_half_length + cfg.ball_radius < landing[0] < -cfg.ball_radius
        assert launch.bounce_time in cfg.bounce_times
        assert velocity[2] - cfg.gravity * launch.bounce_time < 0.0
        for x in (
            -cfg.net_half_thickness - cfg.ball_radius,
            cfg.net_half_thickness + cfg.ball_radius,
        ):
            time = (cfg.launch_x - x) / -velocity[0]
            z = position[2] + velocity[2] * time - 0.5 * cfg.gravity * time**2
            assert z >= cfg.net_top_z + cfg.ball_radius + cfg.net_clearance

    values = np.asarray([(*launch.position, *launch.linear_velocity) for launch in launches])
    assert np.ptp(values[:, 1]) > 0.2
    assert np.ptp(values[:, 3]) > 0.25
    assert np.ptp(values[:, 5]) > 0.08


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"speed_range": (0.0, 1.0)}, "positive"),
        ({"speed_range": (3.0, 2.0)}, "ordered"),
        ({"bounce_times": (0.1,)}, "robot half"),
        ({"bounce_times": ()}, "non-empty"),
        ({"bounce_times": (0.0,)}, "positive"),
        ({"lateral_range": (-0.8, 0.8)}, "inside the table"),
        ({"launch_x": np.nan}, "finite"),
        ({"launch_z": 0.77}, "above the playing surface"),
        ({"bounce_times": (0.35,), "speed_range": (2.7, 2.7)}, "clear the net"),
    ],
)
def test_invalid_feed_windows_fail_before_reset(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        BallLauncherCfg(**overrides)


def test_service_sampling_is_seeded_correlated_and_starts_behind_end_line():
    cfg = ServeLauncherCfg()
    rng = np.random.default_rng(31)
    replay = np.random.default_rng(31)
    for _ in range(128):
        launch = sample_ball_launch(rng, cfg)
        assert launch == sample_ball_launch(replay, cfg)
        assert launch.position[0] == pytest.approx(1.42)
        assert -3.0 <= launch.linear_velocity[0] <= -2.8
        assert launch.linear_velocity[1] == 0.0
        impact = (
            np.asarray(launch.position) + np.asarray(launch.linear_velocity) * launch.bounce_time
        )
        impact[2] -= 4.905 * launch.bounce_time**2
        np.testing.assert_allclose(impact, launch.bounce_position, atol=1e-12)
        up = -cfg.normal_restitution * (launch.linear_velocity[2] - 9.81 * launch.bounce_time)
        receiver_x = (
            impact[0]
            - cfg.opponent_tangential_retention * (-launch.linear_velocity[0]) * 2 * up / 9.81
        )
        assert cfg.receiver_bounce_range[0] <= receiver_x <= cfg.receiver_bounce_range[1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"launch_x": 1.3},
        {"speed_range": (0.0, 3.0)},
        {"normal_restitution": 0.0},
        {"receiver_peak_range": (0.01, 0.02)},
        {"receiver_bounce_range": (-1.5, -1.4)},
        {"lateral_range": (-0.8, 0.8)},
        {"speed_range": (9.0, 10.0)},
        {"launch_x": float("nan")},
    ],
)
def test_unusable_service_windows_are_rejected(overrides):
    with pytest.raises(ValueError):
        ServeLauncherCfg(**overrides)
