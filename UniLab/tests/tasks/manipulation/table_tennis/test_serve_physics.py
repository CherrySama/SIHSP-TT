"""Slow service acceptance against actual contacts and measured preparation."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

pytest.importorskip("ball_prediction", reason="Optional table-tennis core not installed")
mujoco = pytest.importorskip("mujoco")

from scripts.tools.preview_panthera_ball_launcher import _load_preview_model
from tests.tasks.manipulation.table_tennis.serve_validation import measure_serve

from unilab.tasks.manipulation.table_tennis.ball_launcher import (
    ServeLauncherCfg,
    sample_ball_launch,
)


def serves(seed=20261004, count=64):
    cfg = ServeLauncherCfg()
    rng = np.random.default_rng(seed)
    launches = [sample_ball_launch(rng, cfg) for _ in range(count)]
    for y, speed, bounce, peak in product(
        cfg.lateral_range, cfg.speed_range, cfg.receiver_bounce_range, cfg.receiver_peak_range
    ):
        launches.append(
            sample_ball_launch(
                rng,
                replace(
                    cfg,
                    lateral_range=(y, y),
                    speed_range=(speed, speed),
                    receiver_bounce_range=(bounce, bounce),
                    receiver_peak_range=(peak, peak),
                ),
            )
        )
    return launches


def test_fixed_baseline_then_random_and_boundary_serves_prepare_in_time():
    model = _load_preview_model(robot_collisions=False, rail_align=True)
    data = mujoco.MjData(model)
    cfg = replace(
        ServeLauncherCfg(),
        lateral_range=(0.0, 0.0),
        speed_range=(2.9, 2.9),
        receiver_bounce_range=(-0.94, -0.94),
        receiver_peak_range=(0.32, 0.32),
    )
    measure_serve(model, data, sample_ball_launch(np.random.default_rng(7), cfg), rail_align=True)
    for launch in serves():
        assert launch.position[0] == pytest.approx(1.42)
        assert -3.0 <= launch.linear_velocity[0] <= -2.8
        measure_serve(model, data, launch, rail_align=True)


def test_finer_timestep_preserves_service_geometry_and_height():
    model = _load_preview_model(robot_collisions=False)
    data = mujoco.MjData(model)
    for launch in serves(count=4):
        model.opt.timestep = 0.00025
        base = measure_serve(model, data, launch)
        model.opt.timestep = 0.000125
        finer = measure_serve(model, data, launch)
        assert abs(base["receiver_bounce_x"] - finer["receiver_bounce_x"]) <= 0.025
        assert abs(base["rebound_peak"] - finer["rebound_peak"]) <= 0.015
