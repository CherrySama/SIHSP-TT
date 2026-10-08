"""Repeat and record a scripted solution using the public UniLab runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from unilab.envs.manager_based_rl_env import make_manager_based_rl_env
from unilab.tasks.manipulation.table_tennis.return_feasibility import (
    DEFAULT_URDF,
    build_feasibility_env_cfg,
    horizontal_stroke_assessment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stroke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--physics-dt", type=float, default=0.00025)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    solution = json.loads(args.stroke.read_text())
    cfg = build_feasibility_env_cfg(
        lateral=solution["lateral"],
        stroke_mode=solution["stroke_mode"],
        preparation_reach=solution.get("preparation_reach"),
        reference_speed_fraction=solution.get("reference_speed_fraction", 0.9),
        preparation_position_gain=solution.get("preparation_position_gain"),
        preparation_side_deadband=solution.get("preparation_side_deadband"),
    )
    cfg.sim_dt = args.physics_dt
    env = make_manager_based_rl_env(cfg, num_envs=1)
    reports, states = [], []
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        control.record_trace = True
        import mujoco

        mujoco.mj_saveModel(
            env.physics_backend.get_playback_model(), str(args.output / "scene.mjb")
        )
        for repeat in range(args.repetitions):
            if repeat:
                env.reset()
                control.record_trace = False
            control.set_strokes(np.array([solution["stroke"]]))
            for _ in range(400):
                state = env.step(np.zeros((1, control.action_dim), dtype=np.float32))
                if not repeat:
                    states.append(env.get_physics_state_snapshot().copy())
                if state.terminated[0] or state.truncated[0]:
                    break
            report = control.episode_report(0)
            report["horizontal_assessment"] = horizontal_stroke_assessment(report)
            report.update(repeat=repeat, physics_dt=env.physics_dt)
            reports.append(report)
            if not repeat:
                np.savez_compressed(
                    args.output / "substeps.npz",
                    **{
                        key: np.asarray([row[key] for row in control.trace])
                        for key in control.trace[0]
                    },
                )
            print(
                json.dumps(
                    {
                        "repeat": repeat,
                        "success": report["success"],
                        "horizontal_accepted": report["horizontal_assessment"]["accepted"],
                        "failure": report["failure_reason"],
                        "landing": report["landing_position"],
                    }
                ),
                flush=True,
            )
        np.savez_compressed(args.output / "playback.npz", states=np.array(states))
        summary = {
            "solution": solution,
            "physics_dt": env.physics_dt,
            "records": reports,
            "legal": sum(r["success"] for r in reports),
            "horizontal_accepted": sum(r["horizontal_assessment"]["accepted"] for r in reports),
            "trials": len(reports),
            "urdf_sha256": hashlib.sha256(DEFAULT_URDF.read_bytes()).hexdigest(),
        }
        (args.output / "replay.json").write_text(json.dumps(summary, indent=2))
    finally:
        env.close()
    if args.video:
        import imageio.v2 as imageio
        from unisim.visualization.render_many import render_states_get_frames

        frames = render_states_get_frames(
            states,
            str(args.output / "scene.mjb"),
            num_processes=1,
            width=1280,
            height=720,
            cam_distance=4.5,
            cam_azimuth=115,
            cam_elevation=-24,
            cam_lookat=[-0.1, 0, 0.9],
        )
        if not frames:
            raise RuntimeError("No playback frames could be rendered")
        # Preserve every recorded frame, showing the motion at one third speed.
        filename = (
            "legal_return_slow.mp4"
            if summary["legal"] == summary["trials"]
            else "failed_return_slow.mp4"
        )
        imageio.mimsave(args.output / filename, frames, fps=round(1 / env.step_dt / 3))
        imageio.imwrite(args.output / "preview.png", frames[round(1.1 / env.step_dt) - 1])


if __name__ == "__main__":
    main()
