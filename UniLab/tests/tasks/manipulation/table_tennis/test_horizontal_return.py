"""Actual preparation, policy timing, partial reset and reward contracts."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

pytest.importorskip("rail_alignment")
pytest.importorskip("mujoco")

from unilab.envs import make_manager_based_rl_env
from unilab.tasks.manipulation.table_tennis.ball_launcher import ServeLauncherCfg
from unilab.tasks.manipulation.table_tennis.training import build_panthera_horizontal_env_cfg


def test_reference_initialization_is_trainable_and_contract_is_saved():
    import torch
    from tensordict import TensorDict

    from unilab.tasks.manipulation.table_tennis.horizontal_policy import ReferenceResidualModel

    observations = TensorDict({"policy": torch.randn(4, 12)}, batch_size=[4])
    model = ReferenceResidualModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        6,
        hidden_dims=[16, 16],
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.05,
        },
    )
    np.testing.assert_array_equal(model(observations).detach().numpy(), np.zeros((4, 6)))
    loss = (model(observations) - 0.1).square().mean()
    loss.backward()
    assert model.mlp[-1].bias.grad.abs().sum() > 0
    assert all(p.requires_grad for p in model.parameters())


def test_policy_contract_rejects_old_direct_action_semantics(tmp_path):
    import json

    from unilab.utils.sim2sim import (
        CrossBackendIncompatibleError,
        extract_contract_snapshot,
        resolve_sim2sim_config,
    )

    directory = Path(__file__).parents[4] / "src/unilab/conf/ppo"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        old = compose("config", overrides=["task=panthera_return/mujoco"])
        new = compose("config", overrides=["task=panthera_horizontal/mujoco"])
    # Both have six actions; the semantics and observations still must differ.
    assert old.env.actions.return_control._target_ != new.env.actions.return_control._target_
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": extract_contract_snapshot(old)})
    )
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, new, algo_name="ppo", strict=True)


def test_horizontal_owner_is_distinct_and_positive_failure_reward_is_bounded():
    directory = Path(__file__).parents[4] / "src/unilab/conf/ppo"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        cfg = compose("config", overrides=["task=panthera_horizontal/mujoco"])
    assert cfg.training.task_name == "PantheraHorizontalReturn"
    assert cfg.env.actions.return_control._target_.endswith("HorizontalReturnActionCfg")
    assert cfg.env.observations.policy.terms.return_state.func.endswith("horizontal_observation")
    assert cfg.reward.adjustment_cost.params.component == "residual"
    assert cfg.reward.horizontal_form.weight < 0
    maximum = (
        cfg.reward.first_hit.weight
        + cfg.reward.crossed_net.weight
        + sum(
            cfg.reward[k].weight
            for k in ("outgoing_direction", "predicted_net_height", "predicted_landing")
        )
        + (cfg.reward.reaching.weight + cfg.reward.striking.weight) * cfg.env.max_episode_seconds
    )
    assert maximum + cfg.reward.failure.weight < 0


def test_random_preparation_handles_forecast_refresh_without_relaxing_ready_gates():
    cfg = build_panthera_horizontal_env_cfg()
    cfg.auto_reset = False
    env = make_manager_based_rl_env(cfg, num_envs=40)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        for _ in range(95):
            state = env.step(np.zeros((40, 6), np.float32))
            assert not state.terminated.any() and not state.truncated.any()
            if control.ready.all():
                break
        assert control.ready.all()
        assert all(not e.hit for e in control.episodes)
        assert not control._boundary_failed.any()
        assert control._waist_cfg.settle_duration == 0.03
        assert np.all(control.prepared_time < control._arrival)
    finally:
        env.close()


def test_clock_action_changes_real_stroke_timing_and_partial_reset_is_local():
    cfg = build_panthera_horizontal_env_cfg()
    cfg.auto_reset = False
    cfg.events["launch_ball"].params["launcher_cfg"] = ServeLauncherCfg(
        lateral_range=(0.35, 0.35),
        speed_range=(2.9, 2.9),
        receiver_bounce_range=(-0.94, -0.94),
        receiver_peak_range=(0.32, 0.32),
    )
    env = make_manager_based_rl_env(cfg, num_envs=2)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        actions = np.zeros((2, 6), np.float32)
        actions[1, 0] = -1  # Hold with a finite deceleration, not a fixed launch timer.
        previous_clock = np.zeros(2)
        for _ in range(98):
            env.step(actions)
            assert np.all(control._clock >= previous_clock)
            assert (control._clock_rate >= 0).all()
            previous_clock = control._clock.copy()
        assert control.ready.all()
        assert control._clock[0] - control._clock[1] > 0.07
        assert abs(control._reference[0, 1] - control._reference[1, 1]) > 0.02
        assert control._clock_rate[1] < 0.15
        preserved = (control._clock[1], control._reference[1].copy())
        env.reset(np.array([0], np.int32))
        assert control._clock[0] == 0 and not control.ready[0]
        assert control._clock[1] == preserved[0] and control.ready[1]
        np.testing.assert_array_equal(control._reference[1], preserved[1])
    finally:
        env.close()


def test_zero_policy_is_an_actual_legal_reference_and_event_rewards_are_once_only():
    cfg = build_panthera_horizontal_env_cfg()
    cfg.auto_reset = False
    env = make_manager_based_rl_env(cfg, num_envs=2)
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        credit = [dict() for _ in range(2)]
        records = {}
        for _ in range(200):
            state = env.step(np.zeros((2, 6), np.float32))
            assert np.isfinite(state.reward).all()
            for i in range(2):
                if i in records:
                    continue
                for key in control.episodes[i].credit:
                    credit[i][key] = credit[i].get(key, 0) + 1
                if state.terminated[i] or state.truncated[i]:
                    records[i] = control.episode_report(i)
            if len(records) == 2:
                break
            ended = np.flatnonzero(state.terminated | state.truncated)
            if len(ended):
                env.reset(ended.astype(np.int32))
        assert len(records) == 2
        for i, record in records.items():
            assert record["success"] and record["urdf_boundary_passed"]
            assert (
                record["hit_time"]
                < record["outgoing_time"]
                < record["crossing_time"]
                < record["landing_time"]
            )
            assert credit[i] == {"hit": 1, "outgoing": 1, "crossed_net": 1, "success": 1}
            assert "horizontal_assessment" in record
    finally:
        env.close()
