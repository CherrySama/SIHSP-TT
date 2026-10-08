"""Physical acceptance and immutable contact parameters in scripted search."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("rail_alignment")
mujoco = pytest.importorskip("mujoco")

from unilab.envs.manager_based_rl_env import make_manager_based_rl_env
from unilab.tasks.manipulation.table_tennis.prepared_scene import prepared_scene_path
from unilab.tasks.manipulation.table_tennis.return_feasibility import (
    DEFAULT_URDF,
    JOINT_NAMES,
    HorizontalStrokeCriteria,
    build_feasibility_env_cfg,
    feasibility_scene_path,
    horizontal_stroke_assessment,
    urdf_boundaries,
)


def test_profile_preserves_contacts_and_mass_with_urdf_force_and_position_limits():
    original = mujoco.MjModel.from_xml_path(prepared_scene_path())
    model = mujoco.MjModel.from_xml_path(feasibility_scene_path())
    for attribute in (
        "body_mass",
        "body_inertia",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_size",
        "pair_solref",
        "pair_solimp",
    ):
        np.testing.assert_array_equal(getattr(model, attribute), getattr(original, attribute))
    np.testing.assert_array_equal(model.opt.gravity, original.opt.gravity)
    assert not model.body_gravcomp.any()
    for name, boundary in urdf_boundaries().items():
        if name == "joint5":
            continue
        np.testing.assert_allclose(model.joint(name).range, [boundary.lower, boundary.upper])
        np.testing.assert_allclose(
            model.actuator(f"{name}_position").forcerange, [-boundary.effort, boundary.effort]
        )


def test_zero_stroke_keeps_real_preparation_and_contact_then_tracks_failure():
    cfg = build_feasibility_env_cfg(stroke_mode="hermite")
    cfg.actions["return_control"].record_trace = True
    env = make_manager_based_rl_env(cfg, num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        for _ in range(400):
            state = env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            if state.terminated[0] or state.truncated[0]:
                break
        result = control.episode_report(0)
        assert result["prepared_time"] == pytest.approx(0.79975, abs=0.02)
        assert result["hit_time"] is not None
        assert result["outgoing_time"] > result["hit_time"]
        assert result["urdf_boundary_passed"] and not result["success"]
        assert result["failure_reason"] == "out_of_bounds"
        motion = result["motion"]
        assert np.isfinite(motion["hit_paddle_velocity"]).all()
        assert np.linalg.norm(motion["hit_paddle_normal"]) == pytest.approx(1, abs=1e-5)
        # Preparation moved the paddle by tens of centimetres; it must not
        # contaminate the measured stroke after the preparation handoff.
        assert motion["swing_path_m"] < 0.02
        assert max(motion["swing_span_m"]) < 0.02
        assert motion["return_apex_above_table_m"] is not None
        assert motion["net_bottom_clearance_m"] is None
        assert len(control.trace) > 4000
        effort = np.array([row["effort"][0] for row in control.trace])
        assert (np.abs(effort) <= control._effort_limit + 1e-4).all()
        # The shoulder/elbow actuator readings now include gravity compensation.
        assert np.max(np.abs(effort[:, JOINT_NAMES.index("joint3")])) > 4.0
    finally:
        env.close()


def test_measured_speed_rejects_preparation_exceeding_supplied_urdf(tmp_path):
    source = ET.parse(DEFAULT_URDF)
    source.getroot().find("joint[@name='joint1']/limit").set("velocity", "2.0")
    urdf = tmp_path / "bounded.urdf"
    source.write(urdf)
    cfg = build_feasibility_env_cfg(str(urdf))
    env = make_manager_based_rl_env(cfg, num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        for _ in range(120):
            state = env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            if state.terminated[0] or state.truncated[0]:
                break
        result = control.episode_report(0)
        assert result["failure_reason"] == "urdf_velocity"
        assert not result["urdf_boundary_passed"]
        assert result["hit_time"] is None and not result["success"]
    finally:
        env.close()


def test_legacy_legal_stroke_reports_motion_from_actual_substeps():
    cfg = build_feasibility_env_cfg(stroke_mode="hermite")
    cfg.actions["return_control"].record_trace = True
    env = make_manager_based_rl_env(cfg, num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        control.set_strokes(np.array([[0.26, 0.15, 3.591, -4.275, 4.275, 3.591, 0.6, 0.35]]))
        for _ in range(400):
            state = env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            if state.terminated[0] or state.truncated[0]:
                break
        record = control.episode_report(0)
        assert record["success"] and record["urdf_boundary_passed"]
        motion = record["motion"]
        time = np.array([row["time"] for row in control.trace])
        hit = int(np.argmin(abs(time - record["hit_time"])))
        paddle = np.array([row["paddle"][0] for row in control.trace]).astype(float)
        np.testing.assert_allclose(
            motion["hit_paddle_velocity"], (paddle[hit] - paddle[hit - 1]) / env.physics_dt
        )
        airborne = time >= record["outgoing_time"] - 1e-9
        height = np.array([row["qpos"][0, control._ball_qpos[2]] for row in control.trace])
        assert motion["return_apex_above_table_m"] == pytest.approx(
            max(height[airborne]) - control.episodes[0].table_z
        )
        assert motion["net_bottom_clearance_m"] > 0
        assert motion["hit_paddle_velocity"][2] > motion["hit_paddle_velocity"][0]
        departure = int(np.argmin(abs(time - record["outgoing_time"])))
        end = int(np.searchsorted(time, record["outgoing_time"] + 0.2))
        follow = paddle[departure : end + 1]
        assert motion["followthrough_path_m"] == pytest.approx(
            np.linalg.norm(np.diff(follow, axis=0), axis=1).sum(), abs=1e-7
        )
        np.testing.assert_allclose(
            motion["followthrough_span_m"], np.ptp(follow, axis=0), atol=1e-7
        )
        assert motion["followthrough_end_speed_m_s"] == pytest.approx(
            np.linalg.norm(paddle[end] - paddle[end - 1]) / env.physics_dt
        )
        assert not horizontal_stroke_assessment(record)["accepted"]
    finally:
        env.close()


@pytest.mark.parametrize("reach", [0.7, 0.75])
def test_reach_preparation_executes_before_handoff_and_resets(reach):
    cfg = build_feasibility_env_cfg(
        stroke_mode="hermite", preparation_reach=reach, reference_speed_fraction=0.98
    )
    env = make_manager_based_rl_env(cfg, num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        for _ in range(120):
            state = env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            assert not state.terminated[0] and not state.truncated[0]
            if control.ready[0]:
                break
        assert control.ready[0]
        assert not control.episodes[0].hit
        assert not control._boundary_failed[0]
        assert control.prepared_time[0] < control._arrival[0] - 0.15
        base = control._preparation_base[0]
        paddle, quaternion = control.backend.get_body_pose_w(control._paddle_id)
        assert abs(paddle[0, 0, 1] - base[0]) == pytest.approx(reach, abs=0.005)
        assert paddle[0, 0, 2] == pytest.approx(control._height_z[0], abs=0.01)
        assert abs(base[2] - base[3] - base[4]) < 0.002
        assert control._peak_effort[0, 2] > 20
        assert control.guidance_state()[2].tolist() == [False]
        assert quaternion.shape == (1, 1, 4)
        env.reset()
        assert not control.ready[0] and np.isnan(control.prepared_time[0])
        np.testing.assert_array_equal(control._reference[0], control.home)
        assert not control.episodes[0].hit
        for _ in range(5):
            env.step(np.zeros((1, control.action_dim), dtype=np.float32))
        assert not control.ready[0]
        assert not control._boundary_failed[0]
    finally:
        env.close()


def test_reach_preparation_has_a_forward_dominant_legal_witness():
    cfg = build_feasibility_env_cfg(
        stroke_mode="hermite", preparation_reach=0.7, reference_speed_fraction=0.98
    )
    env = make_manager_based_rl_env(cfg, num_envs=1)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        control.set_strokes(
            np.array(
                [
                    [
                        0.35,
                        0.1,
                        4.116,
                        -0.7510612612,
                        2.9701729494,
                        -3.7212342106,
                        0.3782695236,
                        0.25,
                        0.0,
                        -0.0362006272,
                        -0.0004635475,
                        -0.0568233858,
                        0.0562268603,
                    ]
                ]
            )
        )
        for _ in range(400):
            state = env.step(np.zeros((1, control.action_dim), dtype=np.float32))
            if state.terminated[0] or state.truncated[0]:
                break
        result = control.episode_report(0)
        assert result["success"] and result["urdf_boundary_passed"]
        assert result["hit_time"] > result["prepared_time"]
        assert result["crossing_time"] < result["landing_time"]
        motion = result["motion"]
        # This is a physical regression witness, not a calibrated video match.
        assert motion["hit_paddle_velocity"][0] > 2 * motion["hit_paddle_velocity"][2] > 0
        assert motion["swing_span_m"][2] < 0.1
        assert motion["return_apex_above_table_m"] < 0.65
        assert motion["net_bottom_clearance_m"] > 0.03
    finally:
        env.close()


def test_horizontal_assessment_rejects_legal_lob_and_incomplete_followthrough():
    record = {
        "success": True,
        "urdf_boundary_passed": True,
        "motion": {
            "hit_paddle_velocity": [3, 0, 0.3],
            "hit_paddle_normal": [1, 0, 0],
            "contact_face_tilt_max_rad": 0.1,
            "contact_velocity_elevation_max_rad": 0.1,
            "swing_span_m": [0.2, 0.01, 0.02],
            "return_apex_above_table_m": 0.4,
            "swing_path_m": 0.2,
            "backswing_backward_m": 0.05,
            "followthrough_forward_m": 0.04,
            "followthrough_path_m": 0.05,
            "followthrough_span_m": [0.04, 0.01, 0.01],
            "followthrough_end_speed_m_s": 0.05,
        },
    }
    assert horizontal_stroke_assessment(record)["accepted"]
    record["motion"]["return_apex_above_table_m"] = 0.7
    assert "return_apex_above_table_m" in horizontal_stroke_assessment(record)["violations"]
    record["motion"]["return_apex_above_table_m"] = 0.4
    record["motion"]["contact_face_tilt_max_rad"] = 0.5
    assert not horizontal_stroke_assessment(record)["accepted"]
    record["motion"]["contact_face_tilt_max_rad"] = 0.1
    record["motion"]["followthrough_end_speed_m_s"] = None
    assert "followthrough_end_speed_m_s" in horizontal_stroke_assessment(record)["violations"]
    record["motion"]["followthrough_end_speed_m_s"] = 0.05
    record["success"] = False
    assert horizontal_stroke_assessment(record)["motion_passed"]
    assert not horizontal_stroke_assessment(record)["accepted"]
    with pytest.raises(ValueError, match="finite and positive"):
        HorizontalStrokeCriteria(face_tilt_rad=0)
