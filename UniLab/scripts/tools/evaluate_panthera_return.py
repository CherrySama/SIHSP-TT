"""Evaluate the formal random-serve return task and keep every episode outcome.

Examples:
    uv run scripts/tools/evaluate_panthera_return.py --mode zero
    uv run scripts/tools/evaluate_panthera_return.py --checkpoint model_1000.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np


def wilson_interval(successes: int, episodes: int) -> list[float]:
    """Two-sided 95% Wilson interval, including the all-success boundary."""
    if episodes == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(record["success"]) for record in records)
    episodes = len(records)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "success_rate_95pct_ci": wilson_interval(successes, episodes),
        "failure_reasons": dict(
            Counter(
                record.get("failure_reason") or "unspecified_failure"
                for record in records
                if not record["success"]
            )
        ),
    }


def summarize_feed_bins(
    records: list[dict[str, Any]], *, field: str, coordinate: int, edges: list[float]
) -> dict[str, Any]:
    """Expose edge coverage using the actual sampled initial ball state."""
    groups: list[list[dict[str, Any]]] = [[] for _ in range(len(edges) - 1)]
    missing = 0
    outside = 0
    for record in records:
        state = record.get(field)
        if state is None:
            missing += 1
            continue
        value = float(state[coordinate])
        if field == "launch_velocity":
            value = abs(value)
        if value < edges[0] or value > edges[-1]:
            outside += 1
            continue
        index = min(int(np.searchsorted(edges, value, side="right")) - 1, len(groups) - 1)
        groups[index].append(record)
    return {
        "field": field,
        "coordinate": coordinate,
        "edges": edges,
        "missing_state": missing,
        "outside_configured_range": outside,
        "bins": [summarize_episodes(group) for group in groups],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def serve_seed(seed: int, episode: int) -> int:
    """A serve's random stream is independent of parallel termination order."""
    return int(np.random.SeedSequence([seed, episode]).generate_state(1, dtype=np.uint64)[0])


def _compose_config(checkpoint: Path | None, task: str = "panthera_return/mujoco") -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import unilab

    conf_dir = Path(unilab.__file__).parent / "conf/ppo"
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task}"])
    # Saved architecture and normalizer declarations must accompany the weights.
    if checkpoint is not None:
        snapshot_path = checkpoint.parent / "run_config.json"
        if not snapshot_path.is_file():
            raise ValueError(
                "Policy evaluation requires run_config.json beside the checkpoint "
                "to verify the trained observation normalization and architecture"
            )
        snapshot = json.loads(snapshot_path.read_text())
        saved_config = snapshot.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("run_config.json must contain the trained config")
        if (
            task == "panthera_horizontal/mujoco"
            and saved_config.get("training", {}).get("task_name") != "PantheraHorizontalReturn"
        ):
            raise ValueError("Explicit horizontal task conflicts with checkpoint task")
        cfg = OmegaConf.merge(cfg, saved_config)
        if (
            cfg.training.task_name not in {"PantheraReturn", "PantheraHorizontalReturn"}
            or cfg.training.sim_backend != "mujoco"
        ):
            raise ValueError("Checkpoint must belong to a Panthera return task on MuJoCo")
    return cfg


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from unilab.rl import RslRlVecEnvAdapter
    from unilab.tasks.manipulation.table_tennis.ball_launcher import (
        ServeLauncherCfg,
        sample_ball_launch,
    )
    from unilab.training import algo_config_dict
    from unilab.utils.sim2sim import policy_load_dim_guard, resolve_sim2sim_config

    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if args.mode == "policy" and (checkpoint is None or not checkpoint.is_file()):
        raise ValueError("Policy evaluation requires an existing --checkpoint")
    task = getattr(args, "task", "panthera_return/mujoco")
    cfg = (
        _compose_config(checkpoint)
        if task == "panthera_return/mujoco"
        else _compose_config(checkpoint, task)
    )
    if checkpoint is not None:
        resolve_sim2sim_config(checkpoint.parent, cfg, algo_name="ppo", strict=True)
    cfg.env.seed = args.seed
    cfg.env.auto_reset = False
    env_override = BackendAdapter(cfg, root_dir=Path.cwd()).build_task_env_cfg_override()
    env = cast(
        ManagerBasedRlEnv,
        create_env(
            cfg,
            num_envs=min(args.num_envs, args.episodes),
            env_cfg_override=env_override,
            sim_backend="mujoco",
        ),
    )
    env_cfg = cast(ManagerBasedRlEnvCfg, env.cfg)
    launch_event = env_cfg.events["launch_ball"]
    assert launch_event is not None
    launcher_cfg = cast(ServeLauncherCfg, launch_event.params["launcher_cfg"])
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    paired = args.paired_serves
    bank = (
        [
            sample_ball_launch(np.random.default_rng(serve_seed(args.seed, i)), launcher_cfg)
            for i in range(args.episodes)
        ]
        if paired
        else []
    )
    try:
        wrapped = RslRlVecEnvAdapter(env, device="cpu")
        if paired:
            for env_id in range(env.num_envs):
                env.seed(serve_seed(args.seed, env_id))
                env.reset(np.array([env_id], dtype=np.int32))
        policy = None
        if args.mode == "policy":
            runner = OnPolicyRunner(
                cast(Any, wrapped), algo_config_dict(cfg), log_dir=None, device="cpu"
            )
            with policy_load_dim_guard(
                env_obs_dim=wrapped.num_obs,
                env_action_dim=wrapped.num_actions,
                algo_name="ppo",
            ):
                runner.load(
                    str(checkpoint),
                    map_location="cpu",
                    load_cfg={
                        "actor": True,
                        "critic": False,
                        "optimizer": False,
                        "iteration": False,
                        "rnd": False,
                    },
                )
            policy = runner.get_inference_policy(device="cpu")
        term = cast(Any, env.action_manager.get_term("return_control"))
        # Assign exactly N serves; do not finish by selecting the fastest N
        # endings from a larger batch, which would bias towards early failures.
        episode_ids = np.arange(env.num_envs, dtype=np.int64)
        next_episode = env.num_envs
        actions = np.zeros((env.num_envs, wrapped.num_actions), dtype=np.float32)
        rewards = np.zeros(env.num_envs, dtype=np.float64)
        steps = np.zeros(env.num_envs, dtype=np.int64)
        rng = np.random.default_rng(args.seed + 1)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if paired:
            (args.output_dir / "serve_bank.json").write_text(
                json.dumps(
                    [
                        {
                            "serve_id": i,
                            "seed": serve_seed(args.seed, i),
                            "position": launch.position,
                            "velocity": launch.linear_velocity,
                        }
                        for i, launch in enumerate(bank)
                    ],
                    indent=2,
                )
            )
        with (args.output_dir / "episodes.jsonl").open("w") as output:
            while len(records) < args.episodes:
                if policy is not None:
                    with torch.inference_mode():
                        actions = policy(wrapped.get_observations()).cpu().numpy()
                elif args.mode == "random":
                    actions[:] = rng.uniform(-1.0, 1.0, size=actions.shape)
                else:
                    actions.fill(0.0)
                actions[episode_ids < 0] = 0.0
                state = env.step(actions)
                rewards += state.reward
                steps += 1
                done_ids = np.flatnonzero(state.terminated | state.truncated)
                for env_id in done_ids:
                    # Inactive slots still share the physics batch. Reset every
                    # terminal slot before its next step, but do not count its
                    # dummy serves in the N explicitly assigned evaluations.
                    if episode_ids[env_id] < 0:
                        continue
                    report = dict(term.episode_report(int(env_id)))
                    if paired:
                        expected = bank[int(episode_ids[env_id])]
                        for name, target in (
                            ("launch_position", expected.position),
                            ("launch_velocity", expected.linear_velocity),
                        ):
                            if not np.allclose(report[name], target, rtol=0, atol=2e-6):
                                raise ValueError(
                                    f"Serve {episode_ids[env_id]} did not match its fixed bank: {name}"
                                )
                    report["success"] = bool(report["success"])
                    if not report["success"] and not report.get("failure_reason"):
                        report["failure_reason"] = (
                            "timeout" if state.truncated[env_id] else "unspecified_failure"
                        )
                    report.update(
                        episode=int(episode_ids[env_id]),
                        env_id=int(env_id),
                        reward=float(rewards[env_id]),
                        control_steps=int(steps[env_id]),
                    )
                    records.append(report)
                    output.write(json.dumps(report, default=_json_default) + "\n")
                    output.flush()
                    if next_episode < args.episodes:
                        episode_ids[env_id] = next_episode
                        next_episode += 1
                        rewards[env_id] = 0.0
                        steps[env_id] = 0
                    else:
                        episode_ids[env_id] = -1
                if len(done_ids):
                    if paired:
                        for env_id in done_ids:
                            serve_id = int(episode_ids[env_id])
                            if serve_id < 0:
                                serve_id = args.episodes + int(env_id)
                            env.seed(serve_seed(args.seed, serve_id))
                            env.reset(np.array([env_id], dtype=np.int32))
                    else:
                        env.reset(done_ids.astype(np.int32))
    finally:
        env.close()
    summary = summarize_episodes(records)
    if any("horizontal_assessment" in row for row in records):
        accepted = sum(
            row.get("horizontal_assessment", {}).get("accepted", False) for row in records
        )
        summary.update(
            horizontal_legal=accepted,
            horizontal_legal_rate=accepted / len(records),
            horizontal_legal_95pct_ci=wilson_interval(accepted, len(records)),
        )
    summary.update(
        mode=args.mode,
        task_name=cfg.training.task_name,
        paired_serves=paired,
        seed=args.seed,
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        wall_seconds=time.perf_counter() - started,
        sides={
            str(side): summarize_episodes([row for row in records if row.get("side") == side])
            for side in (-1, 1)
        },
        feed_coverage={
            "lateral": summarize_feed_bins(
                records,
                field="launch_position",
                coordinate=1,
                edges=np.linspace(
                    launcher_cfg.lateral_range[0], launcher_cfg.lateral_range[1], 6
                ).tolist(),
            ),
            "forward_speed": summarize_feed_bins(
                records,
                field="launch_velocity",
                coordinate=0,
                edges=np.linspace(
                    launcher_cfg.speed_range[0], launcher_cfg.speed_range[1], 4
                ).tolist(),
            ),
        },
        acceptance_sample_size_met=len(records) >= 1000,
        acceptance_rate_met=summary.get("horizontal_legal_rate", summary["success_rate"]) >= 0.95,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--task",
        choices=("panthera_return/mujoco", "panthera_horizontal/mujoco"),
        default="panthera_return/mujoco",
    )
    parser.add_argument("--mode", choices=("policy", "zero", "random"), default="policy")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument(
        "--paired-serves",
        action="store_true",
        help="Assign each serve an independent seed and verify its initial state",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/panthera_return/evaluation")
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.num_envs <= 0 or args.seed < 0:
        parser.error("episodes and num-envs must be positive; seed must be non-negative")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
