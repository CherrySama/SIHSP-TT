"""Physical preparation timing, facing direction, coupled limits and reset."""

# Optional core imports must follow dependency checks.
# ruff: noqa: E402

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("rail_alignment", reason="Optional table-tennis core not installed")
mujoco = pytest.importorskip("mujoco")

from ball_prediction import BallPrediction
from ball_state import BallState
from rail_alignment import WaistPreparationCfg
from scripts.tools.preview_panthera_ball_launcher import _load_preview_model
from tests.tasks.manipulation.table_tennis.waist_validation import measure_waist

from unilab.tasks.manipulation.table_tennis.ball_launcher import (
    ServeLauncherCfg,
    sample_ball_launch,
)
from unilab.visualization.panthera_rail_preview import PantheraRailPreview


def _launch(lateral):
    return sample_ball_launch(
        np.random.default_rng(7),
        replace(
            ServeLauncherCfg(),
            lateral_range=(lateral, lateral),
            speed_range=(2.9, 2.9),
            receiver_bounce_range=(-0.94, -0.94),
            receiver_peak_range=(0.32, 0.32),
        ),
    )


@pytest.mark.parametrize("lateral", [-0.35, 0.35, 0.0])
def test_both_drives_start_together_and_prepare_actual_facing_paddle(lateral):
    model = _load_preview_model(robot_collisions=False, rail_align=True)
    result = measure_waist(model, mujoco.MjData(model), _launch(lateral))
    assert result["first_command"]["time"] == 0.0
    assert abs(result["first_command"]["rail_force"]) > 0
    assert result["first_rail_motion"] == result["first_yaw_motion"] == model.opt.timestep
    assert result["first_ready_time"] is not None and result["lead_time"] >= 0.15
    crossing = result["crossing"]
    side = 1 if lateral > 0.03 else -1
    assert crossing["yaw_degrees"] == pytest.approx(side * 90, abs=0.5)
    assert crossing["facing_error_degrees"] < 8.0
    assert crossing["lateral_error"] <= 0.01
    # 3 rad/s is a trajectory limit. Permit 0.01 rad/s measured tracking
    # tolerance and report the actual peak rather than clipping the state.
    assert result["peak_arm_speeds"][0] <= 3.01
    assert max(result["peak_arm_speeds"][1:]) <= 1.0
    assert result["max_rail_speed"] <= 1.5
    assert result["max_rail_acceleration"] <= 10.0
    assert result["max_rail_force"] <= 100.0
    assert result["max_yaw_force"] <= 21.0
    assert result["max_rail_position"] <= 0.75
    assert result["max_link5_drift"] < 1e-12


@pytest.mark.parametrize("lateral", [-0.35, 0.35])
def test_full_collisions_separate_preparation_from_passive_ball_contact(lateral):
    model = _load_preview_model(robot_collisions=True, rail_align=True)
    result = measure_waist(model, mujoco.MjData(model), _launch(lateral))
    assert result["first_ready_time"] is not None
    ball_contacts = [
        contact
        for contact in result["contacts"]
        if "ball_geom" in contact["pair"] and "table_top" not in contact["pair"]
    ]
    assert ball_contacts
    assert all(contact["pair"] == ["ball_geom", "paddle_contact"] for contact in ball_contacts)
    assert min(c["time"] for c in ball_contacts) - result["first_ready_time"] >= 0.15


def test_prediction_rejection_freshness_and_reset():
    model = _load_preview_model(robot_collisions=False, rail_align=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rail = PantheraRailPreview(model, data, waist_preparation=WaistPreparationCfg())
    invalid = BallPrediction(0.0, (), valid=False, reason="invalid_measurement")
    rail.observe(invalid)
    rail.apply_force()
    assert rail.waist.target is None and rail.controller.target is None
    np.testing.assert_array_equal(data.ctrl[rail.arm_actuators], rail.home_arm)
    assert not rail.ready
    point = BallState(1.2, (rail.candidate_x, -0.35, 1.0), (-2.0, 0.0, 0.0))
    rail.observe(BallPrediction(0.0, (), candidate=point, valid=True))
    assert rail.waist.target == -np.pi / 2
    # A synthetic pose fixture isolates the status logic, not motion timing.
    data.qpos[rail.arm_qpos[0]] = rail.waist.target
    data.qpos[rail.qpos] = rail.controller.target.position
    mujoco.mj_forward(model, data)
    rail.observe(BallPrediction(0.0, (), candidate=point, valid=True))
    data.time = 0.031
    rail.observe(BallPrediction(0.031, (), valid=False, reason="contact_transition"))
    assert rail.ready
    data.time = 0.051
    rail.observe(BallPrediction(0.051, (), valid=False, reason="invalid_measurement"))
    assert not rail.ready
    data.time = 0.06
    rail.observe(BallPrediction(0.06, (), candidate=point, valid=True))
    data.time = 0.091
    rail.observe(BallPrediction(0.091, (), candidate=point, valid=True))
    assert rail.ready
    rail.observe(BallPrediction(0.091, (), valid=False, reason="not_incoming"))
    assert not rail.ready and rail.controller.target is None
    assert rail.settled_since is None
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rail.reset()
    assert rail.waist.target is None and not rail.preparation_prediction_valid
    assert rail.first_ready_time is None and rail.last_prediction_time is None
    np.testing.assert_array_equal(data.ctrl[rail.arm_actuators], rail.home_arm)
