"""Actual coordinated height preparation, not interpolated success labels."""

# Optional core imports must follow dependency checks.
# ruff: noqa: E402
import numpy as np
import pytest

pytest.importorskip("rail_alignment", reason="Optional table-tennis core not installed")
mujoco = pytest.importorskip("mujoco")

from ball_prediction import BallPrediction
from ball_state import BallState
from rail_alignment import WaistPreparationCfg
from scripts.tools.preview_panthera_ball_launcher import _load_preview_model
from tests.tasks.manipulation.table_tennis.test_waist_preparation import _launch
from tests.tasks.manipulation.table_tennis.waist_validation import measure_waist

from unilab.visualization.panthera_height_preparation import PantheraHeightPreparation
from unilab.visualization.panthera_rail_preview import PantheraRailPreview


@pytest.mark.parametrize("lateral", [-0.35, 0.35, 0.0])
def test_height_rail_and_waist_start_together_and_align_measured_paddle(lateral):
    model = _load_preview_model(robot_collisions=False, rail_align=True)
    result = measure_waist(model, mujoco.MjData(model), _launch(lateral), height_prepare=True)
    assert result["first_arm_motion"] == [model.opt.timestep] * 4
    assert result["first_rail_motion"] == model.opt.timestep
    assert result["first_ready_time"] is not None and result["lead_time"] >= 0.15
    crossing = result["crossing"]
    assert crossing["height_error"] <= 0.01 and crossing["lateral_error"] <= 0.01
    assert crossing["facing_error_degrees"] < 8.0
    assert result["peak_arm_speeds"][0] <= 3.01
    assert max(result["peak_arm_speeds"][1:4]) <= 1.51
    assert result["peak_arm_speeds"][4] <= 1.0
    assert np.all(np.array(result["peak_arm_forces"]) <= [21, 36, 36, 21, 10])
    assert result["max_rail_speed"] <= 1.5
    assert result["max_rail_acceleration"] <= 10.0
    assert result["max_rail_force"] <= 100.0
    assert result["max_rail_position"] <= 0.75
    assert result["max_link5_drift"] < 1e-12


@pytest.mark.parametrize("lateral", [-0.35, 0.35])
def test_full_height_preparation_has_only_table_and_passive_paddle_contacts(lateral):
    model = _load_preview_model(robot_collisions=True, rail_align=True)
    result = measure_waist(model, mujoco.MjData(model), _launch(lateral), height_prepare=True)
    contacts = result["contacts"]
    assert all(
        c["pair"] in (["ball_geom", "table_top"], ["ball_geom", "paddle_contact"]) for c in contacts
    )
    hit = next(c["time"] for c in contacts if c["pair"] == ["ball_geom", "paddle_contact"])
    assert hit - result["first_ready_time"] >= 0.15


def test_lookup_preserves_section_normal_pitch_and_rejects_unsupported_height():
    model = _load_preview_model(robot_collisions=True, rail_align=True)
    height = PantheraHeightPreparation(model, yaw_angle=np.pi / 2)
    data = mujoco.MjData(model)
    qpos = np.array([int(model.joint(f"joint{i}").qposadr[0]) for i in (1, 2, 3, 4, 6)])
    for side in (-1, 1):
        for phase in height.phases:
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            data.qpos[qpos[0]] = side * np.pi / 2
            data.qpos[qpos[1:4]] += phase * height.direction
            mujoco.mj_forward(model, data)
            assert data.ncon == 0
            assert data.xpos[height.paddle, 0] == pytest.approx(-1.53, abs=1e-9)
            assert abs(data.xmat[height.paddle].reshape(3, 3)[0, 2]) == pytest.approx(
                np.sin(1.5), abs=1e-9
            )
        for z in (1.02, 1.08, 1.16):
            assert height.observe(height=z, side=side)
            data.qpos[qpos[1:4]] = height.target_arm
            mujoco.mj_forward(model, data)
            assert abs(data.xpos[height.paddle, 2] - z) < 1e-6
    old = height.target
    assert not height.observe(height=height.heights[-1] + 0.01, side=1)
    assert height.target == old
    assert not height.observe(height=height.heights[0] - 0.01, side=-1)
    height.reset()
    assert height.target is None and height.target_height is None


def test_unsupported_height_cannot_start_or_reuse_readiness_and_reset_clears_target():
    model = _load_preview_model(robot_collisions=False, rail_align=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rail = PantheraRailPreview(
        model, data, waist_preparation=WaistPreparationCfg(), height_preparation=True
    )

    def prediction(z):
        return BallPrediction(
            float(data.time),
            (),
            candidate=BallState(1.3, (rail.candidate_x, -0.35, z), (-2, 0, 0)),
            valid=True,
        )

    rail.observe(prediction(2.0))
    assert rail.waist.target is None and rail.height.target is None
    assert rail.controller.target is None and not rail.ready
    rail.observe(prediction(1.08))
    assert rail.waist.target is not None and rail.height.target is not None
    old = rail.height.target
    rail.observe(prediction(2.0))
    assert rail.height.target == old
    assert not rail.preparation_prediction_valid and rail.controller.target is None
    assert not rail.ready
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rail.reset()
    assert rail.height.target is None and rail.height.target_height is None
    assert rail.height.position == rail.height.velocity == 0.0
    np.testing.assert_array_equal(data.ctrl[rail.arm_actuators], rail.home_arm)
