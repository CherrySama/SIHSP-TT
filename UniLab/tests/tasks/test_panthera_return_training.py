"""Owner config and independent evaluation contracts for PantheraReturn."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).parents[2]


def test_registry_bootstrap_does_not_import_optional_table_tennis_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from unilab.base import registry; "
            "registry.ensure_registries(); "
            "assert registry.contains('PantheraReturn'); "
            "import sys; "
            "assert 'table_tennis_core' not in sys.modules; "
            "assert 'unilab.tasks.manipulation.table_tennis.return_control' not in sys.modules; "
            "assert 'unilab.tasks.manipulation.table_tennis.ball_launcher' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_hydra_owner_exposes_random_feed_and_event_rewards() -> None:
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=panthera_return/mujoco"])
    assert cfg.training.task_name == "PantheraReturn"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.training.init_at_random_ep_len is False
    assert cfg.env.sim_dt == 0.00025
    assert cfg.env.ctrl_dt == 0.01
    assert cfg.env.max_episode_seconds == 4.0
    assert cfg.env.scale_rewards_by_dt is False
    assert cfg.env.actions.return_control.entity_name == "robot"
    assert cfg.env.events.launch_ball.params.launcher_cfg._target_.endswith("ServeLauncherCfg")
    assert cfg.reward.success.weight > cfg.reward.crossed_net.weight > cfg.reward.first_hit.weight
    assert cfg.algo.actor.obs_normalization is True
    assert cfg.algo.seed == cfg.env.seed == 7
    shaping_terms = ("outgoing_direction", "predicted_net_height", "predicted_landing")
    shaping_max = sum(cfg.reward[name].weight for name in shaping_terms)
    assert shaping_max == pytest.approx(2.0)
    assert cfg.env.actions.return_control.residual_control is True
    assert cfg.env.observations.policy.terms.ball_history.history_length == 5
    guidance_max = (
        cfg.reward.reaching.weight + cfg.reward.striking.weight
    ) * cfg.env.max_episode_seconds
    assert (
        cfg.reward.first_hit.weight
        + cfg.reward.crossed_net.weight
        + shaping_max
        + guidance_max
        + cfg.reward.failure.weight
        < 0
    )


def test_owner_materialization_allows_formal_hydra_declarations(monkeypatch) -> None:
    from unilab.base.config_adapter import BackendAdapter
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.tasks.manipulation.table_tennis import prepared_scene
    from unilab.tasks.manipulation.table_tennis.training import build_panthera_return_env_cfg

    monkeypatch.setattr(prepared_scene, "prepared_scene_path", lambda: "/tmp/prepared.xml")
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=panthera_return/mujoco"])
    environment_cfg = build_panthera_return_env_cfg()
    overrides = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    apply_cfg_overrides(environment_cfg, overrides)
    assert environment_cfg.scene is not None
    assert environment_cfg.scene.model_file == "/tmp/prepared.xml"
    assert environment_cfg.rewards["success"].weight == 10.0
    assert environment_cfg.actions["return_control"].entity_name == "robot"
    assert environment_cfg.rewards["predicted_landing"].params["shaping_cfg"].gravity == 9.81


def _evaluation_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_panthera_return", ROOT / "scripts/tools/evaluate_panthera_return.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_denominator_includes_preparation_failure_and_timeout() -> None:
    evaluator = _evaluation_module()
    summary = evaluator.summarize_episodes(
        [
            {"success": True},
            {"success": False, "failure_reason": "preparation_failed"},
            {"success": False, "failure_reason": "timeout"},
        ]
    )
    assert summary["episodes"] == 3
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["failure_reasons"] == {"preparation_failed": 1, "timeout": 1}


def test_perfect_sample_does_not_report_certainty() -> None:
    evaluator = _evaluation_module()
    low, high = evaluator.wilson_interval(1000, 1000)
    assert 0.996 < low < 1.0
    assert high == pytest.approx(1.0)


def test_feed_coverage_includes_both_exact_boundaries() -> None:
    evaluator = _evaluation_module()
    records = [
        {"success": True, "launch_position": [0.8, -0.5, 1.1]},
        {"success": False, "launch_position": [0.8, 0.5, 1.1]},
    ]
    coverage = evaluator.summarize_feed_bins(
        records, field="launch_position", coordinate=1, edges=[-0.5, 0.0, 0.5]
    )
    assert [group["episodes"] for group in coverage["bins"]] == [1, 1]
    assert coverage["outside_configured_range"] == 0


def test_evaluation_resets_idle_terminal_slots_without_counting_dummy_serves(
    monkeypatch, tmp_path
) -> None:
    import argparse
    from types import SimpleNamespace

    import numpy as np
    from omegaconf import OmegaConf

    from unilab import rl
    from unilab.base import config_adapter

    evaluator = _evaluation_module()

    class FakeEnv:
        num_envs = 4
        cfg = SimpleNamespace(
            events={
                "launch_ball": SimpleNamespace(
                    params={
                        "launcher_cfg": SimpleNamespace(
                            lateral_range=(-0.5, 0.5), speed_range=(2.8, 3.0)
                        )
                    }
                )
            }
        )

        def __init__(self):
            self.pending = np.zeros(4, dtype=bool)
            self.steps = np.zeros(4, dtype=int)
            self.reset_counts = np.zeros(4, dtype=int)
            self.action_manager = SimpleNamespace(get_term=lambda _name: self)

        def step(self, actions):
            assert not self.pending.any(), "Terminal slots require reset before stepping"
            self.steps += 1
            self.pending = self.steps >= np.arange(1, 5)
            return SimpleNamespace(
                reward=np.zeros(4), terminated=self.pending, truncated=np.zeros(4, dtype=bool)
            )

        def reset(self, ids):
            self.pending[ids] = False
            self.steps[ids] = 0
            self.reset_counts[ids] += 1

        def episode_report(self, env_id):
            return {"success": False, "failure_reason": "missed", "side": 1}

        def close(self):
            pass

    env = FakeEnv()
    cfg = OmegaConf.create(
        {"env": {"seed": 7, "auto_reset": True}, "training": {"task_name": "PantheraReturn"}}
    )
    monkeypatch.setattr(evaluator, "_compose_config", lambda _checkpoint: cfg)
    monkeypatch.setattr(config_adapter.BackendAdapter, "build_task_env_cfg_override", lambda _: {})
    monkeypatch.setattr(config_adapter, "create_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        rl, "RslRlVecEnvAdapter", lambda *args, **kwargs: SimpleNamespace(num_actions=2)
    )

    summary = evaluator.evaluate(
        argparse.Namespace(
            mode="zero",
            checkpoint=None,
            episodes=5,
            num_envs=4,
            seed=10007,
            output_dir=tmp_path,
            paired_serves=False,
        )
    )
    records = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert summary["episodes"] == len(records) == 5
    assert env.reset_counts[0] == 4  # Two counted serves, then two ignored dummy serves.
