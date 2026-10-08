"""Prediction adapter against actual MuJoCo flight, rather than launch metadata."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

pytest.importorskip("ball_prediction", reason="Optional table-tennis core not installed")
mujoco = pytest.importorskip("mujoco")

from ball_state import BallState
from scripts.tools.preview_panthera_ball_launcher import _draw_prediction, _load_preview_model

from unilab.tasks.manipulation.table_tennis.ball_launcher import (
    BallLaunch,
    BallLauncherCfg,
    sample_ball_launch,
)
from unilab.tasks.manipulation.table_tennis.prediction_adapter import (
    PantheraBallPredictor,
    read_sim_ball_state,
)


def measure_prediction_errors(model, data, launch: BallLaunch) -> np.ndarray:
    """Return maximum bounce/candidate position and candidate time errors.

    Labels come from contact and an interpolated crossing in the actual run.
    Only current position, velocity and time enter the 100-Hz predictor.
    Launch metadata is used solely to initialize the independent simulator.
    """
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    qpos = model.joint("ball_free").qposadr[0]
    dof = model.joint("ball_free").dofadr[0]
    ball_geom = model.geom("ball_geom").id
    table_geom = model.geom("table_top").id
    net_geoms = {model.geom(name).id for name in ("net_mesh", "net_post_pos", "net_post_neg")}
    data.qpos[qpos : qpos + 3] = launch.position
    data.qvel[dof : dof + 3] = launch.linear_velocity
    data.qvel[dof + 3 : dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    predictor = PantheraBallPredictor()
    predictions = []
    first_contact = None
    contacts = 0
    was_touching = False
    saw_post_bounce = False
    saw_contact_transition = False
    crossed_net = False
    for step in range(round(1.0 / model.opt.timestep)):
        if step % round(0.01 / model.opt.timestep) == 0:
            state = BallState(
                float(data.time),
                tuple(data.qpos[qpos : qpos + 3]),
                tuple(data.qvel[dof : dof + 3]),
            )
            prediction = predictor.update(state)
            if state.position[2] <= 0.79:
                assert not prediction.valid and prediction.reason == "contact_transition"
                saw_contact_transition = True
            else:
                assert prediction.valid, (state, prediction.reason)
                assert prediction.candidate is not None
                predictions.append(prediction)
                if predictor.has_bounced:
                    assert prediction.first_bounce is None
                    saw_post_bounce = True
        previous = data.qpos[qpos : qpos + 3].copy()
        previous_time = float(data.time)
        mujoco.mj_step(model, data)
        current = data.qpos[qpos : qpos + 3].copy()
        touching = False
        for contact in data.contact[: data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if ball_geom in pair:
                assert not pair & net_geoms, "Prediction validation feed touched the net"
            touching |= pair == {ball_geom, table_geom}
        if previous[0] > 0 >= current[0]:
            crossed_net = True
        if touching and not was_touching:
            contacts += 1
            assert crossed_net and contacts == 1
            first_contact = current.copy()
        was_touching = touching
        if previous[0] > predictor.cfg.candidate_x >= current[0]:
            fraction = (predictor.cfg.candidate_x - previous[0]) / (current[0] - previous[0])
            candidate = previous + fraction * (current - previous)
            candidate_time = previous_time + fraction * model.opt.timestep
            assert first_contact is not None and saw_post_bounce and saw_contact_transition
            errors = []
            for prediction in predictions:
                bounce_error = (
                    np.linalg.norm(np.asarray(prediction.first_bounce.position) - first_contact)
                    if prediction.first_bounce is not None
                    else 0.0
                )
                errors.append(
                    (
                        bounce_error,
                        np.linalg.norm(np.asarray(prediction.candidate.position) - candidate),
                        abs(prediction.candidate.timestamp - candidate_time),
                    )
                )
            maximum = np.max(errors, axis=0)
            assert np.all(maximum <= [0.02, 0.03, 0.02]), maximum
            return maximum
    pytest.fail("Actual ball did not reach the diagnostic section")


def test_random_and_boundary_feeds_meet_prediction_error_limits() -> None:
    model = _load_preview_model(robot_collisions=False)
    data = mujoco.MjData(model)
    cfg = BallLauncherCfg()
    rng = np.random.default_rng(20261003)
    launches = [sample_ball_launch(rng, cfg) for _ in range(128)]
    for lateral, speed, bounce_time in product(
        cfg.lateral_range, cfg.speed_range, cfg.bounce_times
    ):
        launches.append(
            sample_ball_launch(
                rng,
                replace(
                    cfg,
                    lateral_range=(lateral, lateral),
                    speed_range=(speed, speed),
                    bounce_times=(bounce_time,),
                ),
            )
        )
    for launch in launches:
        measure_prediction_errors(model, data, launch)


def test_history_reset_and_out_of_order_observations() -> None:
    predictor = PantheraBallPredictor()
    predictor.update(BallState(0.49, (-0.95, 0.0, 0.80), (-3.0, 0.0, -2.8)))
    predictor.update(BallState(0.50, (-0.98, 0.0, 0.785), (-2.6, 0.0, 2.6)))
    assert predictor.has_bounced
    stale = predictor.update(BallState(0.48, (-0.9, 0.0, 0.82), (-3.0, 0.0, -2.7)))
    assert not stale.valid and stale.reason == "out_of_order_measurement"
    assert predictor.has_bounced
    predictor.reset()
    assert not predictor.has_bounced
    result = predictor.update(BallState(0.0, (0.75, 0.0, 0.95), (-3.46, 0.0, 2.134)))
    assert result.valid and result.first_bounce is not None


def test_opponent_bounce_does_not_mark_receiver_bounced_and_repeats_are_rejected():
    predictor = PantheraBallPredictor()
    predictor.update(BallState(0.30, (0.6, 0.0, 0.81), (-2.9, 0.0, -3.0)))
    result = predictor.update(BallState(0.31, (0.57, 0.0, 0.82), (-2.7, 0.0, 2.7)))
    assert result.valid and not predictor.has_bounced
    predictor.update(BallState(0.80, (0.4, 0.0, 0.81), (-2.7, 0.0, -2.0)))
    result = predictor.update(BallState(0.81, (0.37, 0.0, 0.82), (-2.6, 0.0, 2.0)))
    assert result.reason == "repeated_opponent_bounce" and not predictor.has_bounced
    predictor.reset()
    state = BallState(0.0, (0.75, 0.0, 0.95), (-3.46, 0.0, 2.134))
    assert predictor.update(state).valid


def test_invalid_measurement_does_not_erase_bounce_history_and_second_bounce_is_rejected() -> None:
    predictor = PantheraBallPredictor()
    predictor.update(BallState(0.49, (-0.95, 0.0, 0.80), (-3.0, 0.0, -2.8)))
    result = predictor.update(BallState(0.495, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), valid=False))
    assert not result.valid
    stale = predictor.update(BallState(0.492, (-0.96, 0.0, 0.795), (-3.0, 0.0, -2.8)))
    assert stale.reason == "out_of_order_measurement"
    predictor.update(BallState(0.50, (-0.98, 0.0, 0.785), (-2.6, 0.0, 2.6)))
    assert predictor.has_bounced
    predictor.update(BallState(1.0, (-1.1, 0.0, 0.80), (-0.2, 0.0, -2.8)))
    result = predictor.update(BallState(1.01, (-1.11, 0.0, 0.81), (-0.17, 0.0, 2.6)))
    assert not result.valid and result.reason == "observed_second_bounce"
    assert result.candidate is None
    assert not predictor.update(BallState(1.02, (-1.12, 0.0, 0.85), (-0.17, 0.0, 2.5))).valid


def test_future_timestamp_does_not_poison_observation_history() -> None:
    predictor = PantheraBallPredictor()
    state = BallState(0.0, (0.75, 0.0, 0.95), (-3.46, 0.0, 2.134))
    assert predictor.update(replace(state, timestamp=10.0), now=0.0).reason == "future_timestamp"
    assert predictor.update(state).valid


def test_duplicate_capture_timestamp_cannot_create_a_bounce() -> None:
    predictor = PantheraBallPredictor()
    predictor.update(BallState(0.49, (-0.95, 0.0, 0.80), (-3.0, 0.0, -2.8)))
    duplicate = predictor.update(BallState(0.49, (-0.98, 0.0, 0.815), (-2.6, 0.0, 2.6)))
    assert not duplicate.valid and duplicate.reason == "duplicate_measurement"
    assert not predictor.has_bounced
    predictor.update(BallState(0.50, (-0.98, 0.0, 0.785), (-2.6, 0.0, 2.6)))
    assert predictor.has_bounced


def test_public_environment_adapter_uses_table_origin_and_episode_clock() -> None:
    pytest.importorskip("unisim.backend.mujoco.backend")
    from unilab.envs import make_manager_based_rl_env
    from unilab.tasks.manipulation.table_tennis import build_panthera_table_tennis_env_cfg

    cfg = build_panthera_table_tennis_env_cfg()
    env = make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    try:
        env.init_state()
        env.reset(seed=31)
        for env_id in range(2):
            state = read_sim_ball_state(env, env_id)
            assert state.timestamp == 0.0
            np.testing.assert_allclose(
                state.position,
                env.scene["ball"].data.root_link_pos_w[env_id] - env.scene.env_origins[env_id],
            )
        env.step(np.zeros((2, 6), dtype=np.float32))
        assert read_sim_ball_state(env, 1).timestamp == pytest.approx(0.01)
        env.reset(env_ids=np.asarray([1], dtype=np.int32))
        assert read_sim_ball_state(env, 0).timestamp == pytest.approx(0.01)
        assert read_sim_ball_state(env, 1).timestamp == 0.0
        with pytest.raises(IndexError):
            read_sim_ball_state(env, 2)
    finally:
        env.close()


def test_prediction_overlay_is_bounded_and_removes_expired_forecast() -> None:
    model = _load_preview_model(robot_collisions=False)
    scene = mujoco.MjvScene(model, maxgeom=100)
    before = model.geom_contype.copy()
    predictor = PantheraBallPredictor()
    state = BallState(0.0, (0.75, 0.0, 0.95), (-3.46, 0.0, 2.134))
    prediction = predictor.update(state)
    _draw_prediction(scene, prediction, [])
    assert 0 < scene.ngeom <= scene.maxgeom
    _draw_prediction(scene, replace(prediction, valid=False), [])
    assert scene.ngeom == 0
    history = [state.position] * 200
    _draw_prediction(scene, prediction, history)
    assert scene.ngeom == scene.maxgeom
    np.testing.assert_array_equal(model.geom_contype, before)
