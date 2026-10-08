"""Search scripted strikes in the isolated URDF-bounded UniLab runtime."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from unilab.envs.manager_based_rl_env import make_manager_based_rl_env
from unilab.tasks.manipulation.table_tennis.return_feasibility import (
    HorizontalStrokeCriteria,
    build_feasibility_env_cfg,
    compact_motion_cost,
    horizontal_stroke_assessment,
    urdf_boundaries,
)


def score(record: dict, objective: str = "legal") -> float:
    """Search heuristic only; actual ReturnEpisode decides legal success."""
    if not record["urdf_boundary_passed"]:
        return -100.0
    if objective == "horizontal":
        assessment = horizontal_stroke_assessment(record)
        if assessment["accepted"]:
            return 1000.0 - 10 * compact_motion_cost(record)
        progress = 200 if record["success"] else min(150, score(record, "legal"))
        return float(progress - 80 * assessment.get("normalized_excess", 5))
    if record["success"]:
        if objective == "compact":
            cost = compact_motion_cost(record)
            # A bounded style adjustment cannot outrank actual legal success.
            return float(1000 - 100 * cost / (1 + cost))
        return 1000.0
    if record["crossing_time"] is not None:
        return 100.0
    if record["outgoing_velocity"] is None:
        return -30.0 if record["hit_time"] is None else -20.0
    p = np.array(record["outgoing_position"])
    v = np.array(record["outgoing_velocity"])
    flight = (v[2] + np.sqrt(v[2] ** 2 + 2 * 9.81 * max(0.0, p[2] - 0.78))) / 9.81
    landing = p + v * flight
    # Before the ball can reach the net, score the remaining horizontal gap.
    # Extrapolating below the first table landing creates an artificial cliff
    # and can rank a backward shot above a useful forward shot.
    net_gap = max(0.0, 0.026 - landing[0])
    if v[0] > 0.01:
        net_time = max(0.0, (0.026 - p[0]) / v[0])
        if net_time <= flight:
            net_z = p[2] + v[2] * net_time - 0.5 * 9.81 * net_time**2
            net_gap = max(0.0, 0.96 - net_z)
    return float(
        10.0
        - 3 * abs(landing[0] - 0.6)
        - 2 * abs(landing[1] - 0.35)
        - 2 * net_gap
        + 0.2 * v[0]
        + 0.2 * v[2]
    )


def evaluate(
    strokes: np.ndarray,
    lateral: float,
    mode: str = "velocity",
    *,
    preparation_reach: float | None = None,
    reference_speed_fraction: float = 0.9,
    preparation_position_gain: float | None = None,
    preparation_side_deadband: float | None = None,
) -> list[dict]:
    env = make_manager_based_rl_env(
        build_feasibility_env_cfg(
            lateral=lateral,
            stroke_mode=mode,
            preparation_reach=preparation_reach,
            reference_speed_fraction=reference_speed_fraction,
            preparation_position_gain=preparation_position_gain,
            preparation_side_deadband=preparation_side_deadband,
        ),
        num_envs=len(strokes),
    )
    records = {}
    try:
        env.init_state()
        control = env.action_manager.get_term("return_control")
        control.set_strokes(strokes)
        actions = np.zeros((len(strokes), control.action_dim), dtype=np.float32)
        for _ in range(400):
            state = env.step(actions)
            ended = np.flatnonzero(state.terminated | state.truncated)
            for i in ended:
                if int(i) not in records:
                    records[int(i)] = control.episode_report(int(i))
            if len(records) == len(strokes):
                break
            if len(ended):
                env.reset(ended.astype(np.int32))
        if len(records) != len(strokes):
            raise RuntimeError("Not every search episode produced an outcome")
    finally:
        env.close()
    return [records[i] for i in range(len(strokes))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lateral", type=float, default=0.35)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mode", choices=("velocity", "hermite"), default="velocity")
    parser.add_argument("--objective", choices=("legal", "compact", "horizontal"), default="legal")
    parser.add_argument("--initial-stroke", type=Path)
    parser.add_argument("--impact-offset-range", type=float, nargs=2)
    parser.add_argument("--pose-offset-range", type=float, nargs=2, default=(0, 0))
    parser.add_argument("--preparation-reach", type=float)
    parser.add_argument("--reference-speed-fraction", type=float, default=0.9)
    parser.add_argument("--preparation-position-gain", type=float)
    parser.add_argument("--preparation-side-deadband", type=float)
    args = parser.parse_args()
    if args.population < 6 or args.generations <= 0 or args.batch <= 0:
        parser.error("population must be >= 6; generations and batch must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lower = np.array([0.04, 0.08, 0.0, -4.5, -4.5, -3.78, -0.8, 0.05])
    upper = np.array([0.35, 0.4, 3.78, 4.5, 4.5, 3.78, 1.3, 0.4])
    mean = np.array([0.18, 0.28, 3.3, 0.0, 0.0, 0.0, 0.5, 0.28])
    if not np.isfinite(args.reference_speed_fraction) or not 0 < args.reference_speed_fraction <= 1:
        parser.error("reference speed fraction must be in (0, 1]")
    if args.preparation_reach is not None and (
        not np.isfinite(args.preparation_reach) or args.preparation_reach <= 0
    ):
        parser.error("preparation reach must be finite and positive")
    boundaries = urdf_boundaries()
    for column, name in zip((2, 3, 4, 5), ("joint1", "joint2", "joint3", "joint4"), strict=True):
        upper[column] = args.reference_speed_fraction * boundaries[name].velocity
        lower[column] = 0 if column == 2 else -upper[column]
    deviation = (upper - lower) / 3
    history = []
    default_offset = -0.01 if args.mode == "hermite" else 0.0
    offset_range = args.impact_offset_range or (default_offset, default_offset)
    if not np.isfinite(offset_range).all() or offset_range[0] > offset_range[1]:
        parser.error("impact offset range must have finite ordered bounds")
    lower = np.append(lower, offset_range[0])
    upper = np.append(upper, offset_range[1])
    mean = np.append(mean, np.clip(default_offset, *offset_range))
    deviation = np.append(deviation, (offset_range[1] - offset_range[0]) / 3)
    if (
        not np.isfinite(args.pose_offset_range).all()
        or args.pose_offset_range[0] > args.pose_offset_range[1]
    ):
        parser.error("pose offset range must have finite ordered bounds")
    if args.mode != "hermite" and any(args.pose_offset_range):
        parser.error("pose offsets require hermite mode")
    lower = np.r_[lower, np.repeat(args.pose_offset_range[0], 4)]
    upper = np.r_[upper, np.repeat(args.pose_offset_range[1], 4)]
    mean = np.r_[mean, np.zeros(4)]
    deviation = np.r_[deviation, (upper[9:] - lower[9:]) / 3]
    if args.initial_stroke:
        initial = json.loads(args.initial_stroke.read_text())
        if initial["stroke_mode"] != args.mode or initial["lateral"] != args.lateral:
            parser.error("initial stroke must have the selected mode and lateral")
        if (
            initial.get("preparation_reach") != args.preparation_reach
            or initial.get("reference_speed_fraction", 0.9) != args.reference_speed_fraction
            or initial.get("preparation_position_gain") != args.preparation_position_gain
            or initial.get("preparation_side_deadband") != args.preparation_side_deadband
        ):
            parser.error("initial stroke preparation and reference speed must match")
        values = np.asarray(initial["stroke"], dtype=float)
        if len(values) == 8:
            values = np.append(values, default_offset)
        if len(values) == 9:
            values = np.r_[values, np.zeros(4)]
        mean = np.clip(values, lower, upper)
        deviation = (upper - lower) * 0.12

    def rank(record: dict) -> float:
        return score(record, args.objective)

    if args.resume:
        previous = json.loads(args.resume.read_text())
        if previous["stroke_mode"] != args.mode or previous["lateral"] != args.lateral:
            parser.error("resume must have the selected mode and lateral")
        if (
            previous.get("preparation_reach") != args.preparation_reach
            or previous.get("reference_speed_fraction", 0.9) != args.reference_speed_fraction
            or previous.get("preparation_position_gain") != args.preparation_position_gain
            or previous.get("preparation_side_deadband") != args.preparation_side_deadband
        ):
            parser.error("resume preparation and reference speed must match")
        history = previous["records"]
        for record in history:
            if len(record["stroke"]) == 8:
                record["stroke"].append(default_offset)
            if len(record["stroke"]) == 9:
                record["stroke"].extend([0.0] * 4)
        if args.objective == "compact" and any("motion" not in r for r in history):
            parser.error("compact resume requires measured motion records")
        elites = sorted(history, key=rank, reverse=True)[: max(4, args.population // 6)]
        values = np.array([r["stroke"] for r in elites])
        mean = values.mean(axis=0)
        deviation = np.maximum(values.std(axis=0), (upper - lower) * 0.05)
        deviation[9:] = np.maximum(deviation[9:], (upper[9:] - lower[9:]) / 3)
    for generation in range(args.generations):
        candidates = np.clip(rng.normal(mean, deviation, (args.population, 13)), lower, upper)
        candidates[0] = mean
        if history:
            candidates[0] = max(history, key=rank)["stroke"]
        if generation == 0 and not history and not args.initial_stroke:
            # Cover face tilt and yaw before relying on a high-dimensional draw.
            for i, tilt in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
                candidates[i] = [0.14, 0.3, 3.7, 0, 0, 0, tilt, 0.35, default_offset, 0, 0, 0, 0]
        records = []
        for start in range(0, len(candidates), args.batch):
            batch = evaluate(
                candidates[start : start + args.batch],
                args.lateral,
                args.mode,
                preparation_reach=args.preparation_reach,
                reference_speed_fraction=args.reference_speed_fraction,
                preparation_position_gain=args.preparation_position_gain,
                preparation_side_deadband=args.preparation_side_deadband,
            )
            records.extend(batch)
            history.extend(batch)
            if args.objective == "horizontal":
                for record in batch:
                    record["horizontal_assessment"] = horizontal_stroke_assessment(record)
            payload = {
                "seed": args.seed,
                "lateral": args.lateral,
                "stroke_mode": args.mode,
                "objective": args.objective,
                "preparation_reach": args.preparation_reach,
                "reference_speed_fraction": args.reference_speed_fraction,
                "preparation_position_gain": args.preparation_position_gain,
                "preparation_side_deadband": args.preparation_side_deadband,
                "horizontal_criteria": (
                    HorizontalStrokeCriteria().__dict__ if args.objective == "horizontal" else None
                ),
                "generation": generation,
                "records": history,
                "best": max(history, key=rank),
            }
            (args.output / "search.json").write_text(json.dumps(payload, indent=2))
            print(
                json.dumps(
                    {
                        "trials": len(history),
                        "legal": sum(r["success"] for r in history),
                        "horizontal_accepted": sum(
                            r.get("horizontal_assessment", {}).get("accepted", False)
                            for r in history
                        ),
                        "crossed": sum(r["crossing_time"] is not None for r in history),
                        "best_score": rank(payload["best"]),
                        "best_motion": payload["best"]["motion"],
                        "failures": dict(Counter(r["failure_reason"] for r in history)),
                    }
                ),
                flush=True,
            )
            if args.objective == "legal" and any(r["success"] for r in batch):
                return
        elites = sorted(records, key=rank, reverse=True)[: max(4, args.population // 6)]
        values = np.array([r["stroke"] for r in elites])
        mean = values.mean(axis=0)
        deviation = np.maximum(values.std(axis=0), (upper - lower) * 0.04)


if __name__ == "__main__":
    main()
