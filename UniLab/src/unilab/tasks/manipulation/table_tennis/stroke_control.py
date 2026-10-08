"""Shared measured stroke control under the declared URDF boundaries.

Scripted diagnosis and policy control share preparation, tracking and physical
acceptance. Diagnostic scoring never determines real success.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from unilab.envs import ManagerBasedRlEnvCfg
    from unilab.managers._types import ManagerBasedRlEnv

from unilab.tasks.manipulation.table_tennis.prepared_scene import prepared_scene_path
from unilab.tasks.manipulation.table_tennis.return_control import (
    JOINT_NAMES,
    PantheraReturnAction,
    PantheraReturnActionCfg,
    _normal,
)

DEFAULT_URDF = Path(__file__).resolve().parents[6] / "assets/panthera-6dof.urdf"
EFFORT_SENSORS = tuple(f"feasibility_effort_{name}" for name in JOINT_NAMES)
HOLD_SENSOR = "feasibility_joint5_reaction"


@dataclass(frozen=True)
class HorizontalStrokeCriteria:
    """Exploratory motion gates, not values reconstructed from the video."""

    face_tilt_rad: float = np.deg2rad(15)
    velocity_elevation_rad: float = np.deg2rad(15)
    vertical_span_m: float = 0.05
    return_apex_above_table_m: float = 0.5
    complete_stroke_path_m: float = 0.45
    minimum_backswing_m: float = 0.01
    minimum_followthrough_m: float = 0.01
    followthrough_path_m: float = 0.15
    followthrough_vertical_span_m: float = 0.05
    followthrough_end_speed_m_s: float = 0.2

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) or value <= 0 for value in self.__dict__.values()):
            raise ValueError("Horizontal stroke criteria must be finite and positive")


def horizontal_stroke_assessment(
    record: dict, criteria: HorizontalStrokeCriteria | None = None
) -> dict:
    """Keep physical success and measured motion acceptance separate."""
    criteria = criteria or HorizontalStrokeCriteria()
    motion = record["motion"]
    velocity, normal = motion["hit_paddle_velocity"], motion["hit_paddle_normal"]
    if velocity is None or normal is None or not np.isfinite([velocity, normal]).all():
        return {"accepted": False, "motion_passed": False, "violations": ["no_contact"]}
    velocity = np.asarray(velocity)
    measurements = {
        "face_tilt_rad": motion.get("contact_face_tilt_max_rad"),
        "velocity_elevation_rad": motion.get("contact_velocity_elevation_max_rad"),
        "vertical_span_m": motion["swing_span_m"][2],
        "return_apex_above_table_m": motion["return_apex_above_table_m"],
        "complete_stroke_path_m": motion["swing_path_m"] + motion.get("followthrough_path_m", 0),
        "minimum_backswing_m": motion.get("backswing_backward_m"),
        "minimum_followthrough_m": motion.get("followthrough_forward_m"),
        "followthrough_path_m": motion.get("followthrough_path_m"),
        "followthrough_vertical_span_m": motion.get("followthrough_span_m", [None] * 3)[2],
        "followthrough_end_speed_m_s": motion.get("followthrough_end_speed_m_s"),
    }
    violations = []
    excess = 0.0
    for name, value in measurements.items():
        bound = getattr(criteria, name)
        if value is None or not np.isfinite(value):
            violations.append(name)
            excess += 1
            continue
        deficit = (
            (bound - value) / bound if name.startswith("minimum_") else (value - bound) / bound
        )
        if deficit > 0:
            violations.append(name)
            excess += deficit
    if velocity[0] <= 0:
        violations.append("forward_velocity")
        excess += 1
    motion_passed = not violations
    return {
        "accepted": bool(record["success"] and record["urdf_boundary_passed"] and motion_passed),
        "motion_passed": motion_passed,
        "violations": violations,
        "normalized_excess": float(excess),
        "measurements": measurements,
        "criteria": {name: float(value) for name, value in criteria.__dict__.items()},
    }


def compact_motion_cost(record: dict) -> float:
    """Rank measured strokes, not calibrated imitation or a PPO reward.

    Metre-scale apex/path and radian-scale velocity/face angles prefer a
    direct, short stroke. Legality remains an independent event decision.
    """
    motion = record["motion"]
    velocity = np.asarray(motion["hit_paddle_velocity"])
    normal = np.asarray(motion["hit_paddle_normal"])
    elevation = np.arctan2(abs(velocity[2]), max(0.01, velocity[0]))
    face_tilt = np.arcsin(np.clip(abs(normal[2]), 0, 1))
    return float(
        motion["return_apex_above_table_m"]
        + 0.25 * elevation
        + 0.25 * face_tilt
        + 0.2 * motion["swing_path_m"]
    )


@dataclass(frozen=True)
class JointBoundary:
    lower: float
    upper: float
    velocity: float
    effort: float


def urdf_boundaries(path: str | Path = DEFAULT_URDF) -> dict[str, JointBoundary]:
    """Read model declarations on the cold path, with no hardware inference."""
    result = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        values = JointBoundary(
            *(float(limit.attrib[key]) for key in ("lower", "upper", "velocity", "effort"))
        )
        if (
            not np.isfinite(tuple(values.__dict__.values())).all()
            or values.lower >= values.upper
            or values.velocity <= 0
            or values.effort <= 0
        ):
            raise ValueError(f"Invalid URDF limits for {joint.attrib['name']}")
        result[joint.attrib["name"]] = values
    for name in (*JOINT_NAMES[1:], "joint5"):
        if name not in result:
            raise ValueError(f"URDF is missing {name}")
    return result


@lru_cache(maxsize=4)
def feasibility_scene_path(urdf_path: str = str(DEFAULT_URDF)) -> str:
    """Keep native contact physics, but put gravity inside the torque budget."""
    source = Path(prepared_scene_path())
    boundaries = urdf_boundaries(urdf_path)
    fingerprint = hashlib.sha256(
        source.read_bytes() + Path(urdf_path).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / "unilab_panthera_feasibility" / fingerprint
    target = directory / "scene.xml"
    if target.exists():
        return str(target)
    root = ET.parse(source).getroot()
    for body in root.iter("body"):
        if body.get("name") in {f"link{i}" for i in range(1, 7)}:
            body.set("gravcomp", "0")
        if body.get("name") == "link5":
            # Joint5 was removed in the validated preparation profile. A sensor
            # at its original origin measures the reaction along local -Z.
            ET.SubElement(body, "site", name="feasibility_joint5_site", size="0.001")
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name in JOINT_NAMES[1:]:
            bound = boundaries[name]
            joint.set("range", f"{bound.lower} {bound.upper}")
            joint.set("actuatorfrcrange", f"{-bound.effort} {bound.effort}")
    actuators = root.find("actuator")
    assert actuators is not None
    for actuator in actuators:
        name = actuator.get("joint")
        if name in JOINT_NAMES[1:]:
            bound = boundaries[name]
            actuator.set("ctrlrange", f"{bound.lower} {bound.upper}")
            actuator.set("forcerange", f"{-bound.effort} {bound.effort}")
    sensors = root.find("sensor")
    assert sensors is not None
    for name, sensor in zip(JOINT_NAMES, EFFORT_SENSORS, strict=True):
        ET.SubElement(sensors, "jointactuatorfrc", name=sensor, joint=name)
    ET.SubElement(sensors, "torque", name=HOLD_SENSOR, site="feasibility_joint5_site")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=directory, delete=False) as output:
        output.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        temporary = output.name
    os.replace(temporary, target)
    return str(target)


def build_feasibility_env_cfg(
    urdf_path: str = str(DEFAULT_URDF),
    *,
    lateral: float = 0.35,
    stroke_mode: str = "velocity",
    preparation_reach: float | None = None,
    reference_speed_fraction: float = 0.9,
    preparation_position_gain: float | None = None,
    preparation_side_deadband: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """A fixed original-distribution serve, without learning or auto reset."""
    from unilab.tasks.manipulation.table_tennis.ball_launcher import ServeLauncherCfg
    from unilab.tasks.manipulation.table_tennis.training import build_panthera_return_env_cfg

    cfg = build_panthera_return_env_cfg()
    assert cfg.scene is not None
    cfg.scene.model_file = feasibility_scene_path(urdf_path)
    cfg.auto_reset = False
    cfg.rewards = {}
    cfg.actions["return_control"] = PantheraStrokeActionCfg(
        entity_name="robot",
        urdf_path=urdf_path,
        stroke_mode=stroke_mode,
        preparation_reach=preparation_reach,
        reference_speed_fraction=reference_speed_fraction,
        preparation_position_gain=preparation_position_gain,
        preparation_side_deadband=preparation_side_deadband,
    )
    launch = cfg.events["launch_ball"]
    assert launch is not None
    launch.params["launcher_cfg"] = ServeLauncherCfg(
        lateral_range=(lateral, lateral),
        speed_range=(2.9, 2.9),
        receiver_bounce_range=(-0.94, -0.94),
        receiver_peak_range=(0.32, 0.32),
    )
    return cfg


@dataclass(kw_only=True)
class PantheraStrokeActionCfg(PantheraReturnActionCfg):
    """Shared preparation, reference profile and measured physical guards."""

    urdf_path: str = str(DEFAULT_URDF)
    record_trace: bool = False
    stroke_mode: str = "velocity"
    preparation_reach: float | None = None
    reference_speed_fraction: float = 0.9
    preparation_position_gain: float | None = None
    preparation_side_deadband: float | None = None

    def build(self, env: ManagerBasedRlEnv) -> PantheraStrokeAction:
        return PantheraStrokeAction(self, env)


class PantheraStrokeAction(PantheraReturnAction):
    """Reuse measured preparation/events and search unrestricted arm strokes.

    Stroke columns are lead, duration, forward-yaw velocity, joint2/3/4
    velocities, face tilt relative to preparation, face-tilt lead, and an
    optional planned-impact offset relative to the incoming prediction. Four
    optional terminal offsets then move yaw/2/3/4 from the preparation pose;
    positive yaw offset is forward on either side. Velocities
    are in rad/s and times in seconds. The profile's acceleration values only
    smooth references; the URDF effort and measured velocity bound acceptance.
    """

    def __init__(self, cfg: PantheraStrokeActionCfg, env: ManagerBasedRlEnv) -> None:
        if cfg.preparation_reach is not None and (
            not np.isfinite(cfg.preparation_reach) or cfg.preparation_reach <= 0
        ):
            raise ValueError("Preparation reach must be finite and positive")
        if (
            not np.isfinite(cfg.reference_speed_fraction)
            or not 0 < cfg.reference_speed_fraction <= 1
        ):
            raise ValueError("Reference speed fraction must be in (0, 1]")
        for value in (cfg.preparation_position_gain, cfg.preparation_side_deadband):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Preparation tracking gain and side deadband must be positive")
        super().__init__(cfg, env)
        if cfg.stroke_mode not in {"velocity", "hermite"}:
            raise ValueError("Unknown scripted stroke mode")
        self.stroke_mode = cfg.stroke_mode
        self.preparation_reach = cfg.preparation_reach
        self.reference_speed_fraction = cfg.reference_speed_fraction
        self.preparation_position_gain = cfg.preparation_position_gain
        self.preparation_side_deadband = cfg.preparation_side_deadband
        if cfg.preparation_side_deadband is not None:
            from dataclasses import replace

            from rail_alignment import WaistPreparationController

            self._waist_cfg = replace(self._waist_cfg, side_deadband=cfg.preparation_side_deadband)
            self._waists = [
                WaistPreparationController(self.home[1], self._waist_cfg)
                for _ in range(env.num_envs)
            ]
        self.boundaries = urdf_boundaries(cfg.urdf_path)
        self._max_velocity = np.array(
            [1.5, *(self.boundaries[j].velocity for j in JOINT_NAMES[1:])]
        )
        self._max_acceleration = np.array([10.0, 40.0, 40.0, 40.0, 40.0, 60.0])
        self._effort_limit = np.array(
            [
                100.0,
                *(self.boundaries[j].effort for j in JOINT_NAMES[1:]),
                self.boundaries["joint5"].effort,
            ]
        )
        self._effort_sensors = self.backend.bind_sensor_data((*EFFORT_SENSORS, HOLD_SENSOR))
        self.strokes = np.zeros((env.num_envs, 13))
        self.strokes[:, 1] = 0.3
        self.strokes[:, 8] = -0.01 if self.stroke_mode == "hermite" else 0.0
        self._peak_effort = np.zeros((env.num_envs, 7))
        self._minimum_position = np.tile(self.home, (env.num_envs, 1))
        self._maximum_position = self._minimum_position.copy()
        self._boundary_failed = np.zeros(env.num_envs, dtype=bool)
        self.record_trace = cfg.record_trace
        self.trace: list[dict[str, object]] = []
        self._previous_paddle = np.zeros((env.num_envs, 3))
        self._swing_path = np.zeros(env.num_envs)
        self._swing_started = np.zeros(env.num_envs, dtype=bool)
        self._swing_lower = np.zeros((env.num_envs, 3))
        self._swing_upper = np.zeros((env.num_envs, 3))
        self._hit_paddle_velocity = np.full((env.num_envs, 3), np.nan)
        self._hit_paddle_normal = np.full((env.num_envs, 3), np.nan)
        self._return_apex = np.full(env.num_envs, np.nan)
        self._net_clearance = np.full(env.num_envs, np.nan)
        self._backswing_start_x = np.zeros(env.num_envs)
        self._backswing_backward = np.zeros(env.num_envs)
        self._contact_face_tilt = np.zeros(env.num_envs)
        self._contact_velocity_elevation = np.zeros(env.num_envs)
        self._outgoing_paddle = np.zeros((env.num_envs, 3))
        self._followthrough_lower = np.zeros((env.num_envs, 3))
        self._followthrough_upper = np.zeros((env.num_envs, 3))
        self._followthrough_path = np.zeros(env.num_envs)
        self._followthrough_end_speed = np.full(env.num_envs, np.nan)
        self._previous_ball = np.zeros((env.num_envs, 3))
        self._reach_arm_poses = np.empty((0, 3))
        if self.preparation_reach is not None:
            self._build_reach_preparation()

    def _build_reach_preparation(self) -> None:
        """Cold FK lookup at constant lateral reach, preserving the arm pitch.

        No physical state is rewritten. Unreachable heights are rejected;
        runtime prediction only interpolates this contiguous IK branch.
        """
        assert self.preparation_reach is not None
        seed = np.array([2.5, 2.5, 0.0])
        lower, upper = self._limits[2:5].T
        heights, arms = [], []

        def residual(configurations: np.ndarray, height: float) -> np.ndarray:
            poses = np.tile(self.home, (len(configurations), 1))
            poses[:, 1] += np.pi / 2
            poses[:, 2:5] = configurations
            positions, _ = self.backend.evaluate_body_poses(JOINT_NAMES, poses, ("paddle_frame",))
            return np.column_stack(
                (
                    5 * (positions[:, 0, 1] - self.preparation_reach),
                    5 * (positions[:, 0, 2] - height),
                    configurations[:, 0] - configurations[:, 1] - configurations[:, 2],
                )
            )

        grid = np.linspace(self._heights[0], self._heights[-1], 151)
        for height in grid:
            arm = seed.copy()
            for _ in range(40):
                error = residual(arm[None], height)[0]
                if np.linalg.norm(error) < 1e-7:
                    break
                delta = 1e-4
                perturbed = np.tile(arm, (6, 1))
                for joint in range(3):
                    perturbed[2 * joint, joint] += delta
                    perturbed[2 * joint + 1, joint] -= delta
                errors = residual(perturbed, height)
                jacobian = ((errors[::2] - errors[1::2]) / (2 * delta)).T
                direction = np.linalg.lstsq(jacobian, -error, rcond=None)[0]
                for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    trial = np.clip(arm + scale * direction, lower, upper)
                    if np.linalg.norm(residual(trial[None], height)[0]) < np.linalg.norm(error):
                        arm = trial
                        break
                else:
                    break
            if np.linalg.norm(residual(arm[None], height)[0]) < 1e-6:
                heights.append(height)
                arms.append(arm)
                seed = arm
        if len(heights) < 2 or np.max(np.diff(heights)) > 1.1 * (grid[1] - grid[0]):
            raise ValueError("Preparation reach has no contiguous height branch")
        self._phases = np.asarray(heights)
        self._reach_arm_poses = np.asarray(arms)
        for side in (-1, 1):
            poses = np.tile(self.home, (len(heights), 1))
            poses[:, 1] += side * np.pi / 2
            poses[:, 2:5] = self._reach_arm_poses
            positions, quaternions = self.backend.evaluate_body_poses(
                JOINT_NAMES, poses, ("paddle_frame",)
            )
            points = positions[:, 0]
            normals = _normal(quaternions[:, 0])
            if (
                np.any(np.diff(points[:, 2]) <= 0)
                or not np.allclose(points[:, 0], self.candidate_x, atol=1e-6)
                or not np.allclose(normals, normals[0], atol=1e-6)
            ):
                raise ValueError("Reach preparation changed the hit section or paddle normal")
            self._points[side] = points
        self._heights = self._points[1][:, 2]
        if not np.allclose(self._heights, self._points[-1][:, 2], atol=1e-6):
            raise ValueError("Reach preparation left/right height branches differ")

    def _reach_arm_target(self, i: int) -> np.ndarray:
        return np.array(
            [
                np.interp(self._height_target[i], self._phases, self._reach_arm_poses[:, joint])
                for joint in range(3)
            ]
        )

    def _advance_height(self, i: int, dt: float) -> None:
        if self.preparation_reach is None:
            super()._advance_height(i, dt)
            return
        if not np.isfinite(self._height_target[i]):
            return
        for joint, target in zip((2, 3, 4), self._reach_arm_target(i), strict=True):
            error = target - self._reference[i, joint]
            acceleration = self._max_acceleration[joint]
            step = acceleration * dt
            stopping = np.sqrt(step * step + 2 * acceleration * abs(error)) - step
            goal = np.copysign(
                min(self.reference_speed_fraction * self._max_velocity[joint], stopping), error
            )
            if self.preparation_position_gain is not None:
                # The stopping-distance law grows as sqrt(error). A linear
                # near-target cap avoids large velocity pulses on tiny updates.
                goal = np.copysign(
                    min(abs(goal), self.preparation_position_gain * abs(error)), error
                )
            self._advance_velocity(
                i,
                joint,
                float(goal),
                dt,
                float(self._limits[joint, 0]),
                float(self._limits[joint, 1]),
                acceleration,
            )

    def _check_ready(
        self,
        i: int,
        position: np.ndarray,
        velocity: np.ndarray,
        paddle: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        if self.preparation_reach is None:
            super()._check_ready(i, position, velocity, paddle, normal)
            return
        if not np.isfinite(self._height_target[i]):
            self._settled_since[i] = np.nan
            return
        target = self.home.copy()
        waist = self._waists[i]
        if waist.target is not None:
            target[1] = waist.target
        target[2:5] = self._reach_arm_target(i)
        ready = bool(
            self._prediction_valid[i]
            and self.elapsed[i] - self._last_prediction[i] <= 0.05
            and waist.ready(position=position[1], velocity=velocity[1])
            and np.max(np.abs(position[1:] - target[1:])) <= 0.015
            and np.max(np.abs(velocity[1:])) <= 0.05
            and abs(normal[0]) >= np.cos(np.deg2rad(8))
            and abs(paddle[2] - self._height_z[i]) <= 0.01
            and self._rails[i].ready(now=self.elapsed[i], paddle_y=paddle[1], velocity=velocity[0])
        )
        if not ready:
            self._settled_since[i] = np.nan
            return
        if not np.isfinite(self._settled_since[i]):
            self._settled_since[i] = self.elapsed[i]
        if self.elapsed[i] - self._settled_since[i] >= 0.03 - 1e-9:
            self.ready[i] = True
            self.prepared_time[i] = self.elapsed[i]
            self._reference[i] = position
            self._preparation_base[i] = position
            self._reference_velocity[i] = 0
            self._reference_acceleration[i] = 0

    def _update_guidance(self) -> None:
        if self.preparation_reach is None:
            super()._update_guidance()
        # The PPO envelope describes the original preparation branch. This
        # independent, reward-free experiment does not consume that guidance.

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        super().reset(env_ids)
        ids = np.arange(self._env.num_envs)[slice(None) if env_ids is None else env_ids]
        self._peak_effort[ids] = 0
        self._minimum_position[ids] = self.home
        self._maximum_position[ids] = self.home
        self._boundary_failed[ids] = False
        self.trace.clear()
        paddle, _ = self.backend.get_body_pose_w(self._paddle_id)
        self._previous_paddle[ids] = paddle[ids, 0]
        self._swing_lower[ids] = paddle[ids, 0]
        self._swing_upper[ids] = paddle[ids, 0]
        self._swing_path[ids] = 0
        self._swing_started[ids] = False
        self._hit_paddle_velocity[ids] = np.nan
        self._hit_paddle_normal[ids] = np.nan
        self._return_apex[ids] = np.nan
        self._net_clearance[ids] = np.nan
        self._backswing_start_x[ids] = 0
        self._backswing_backward[ids] = 0
        self._contact_face_tilt[ids] = 0
        self._contact_velocity_elevation[ids] = 0
        self._outgoing_paddle[ids] = 0
        self._followthrough_lower[ids] = 0
        self._followthrough_upper[ids] = 0
        self._followthrough_path[ids] = 0
        self._followthrough_end_speed[ids] = np.nan
        self._previous_ball[ids] = self.backend.get_state(("qpos",))["qpos"][
            np.ix_(ids, self._ball_qpos)
        ]

    def set_strokes(self, strokes: np.ndarray) -> None:
        values = np.asarray(strokes, dtype=np.float64)
        if (
            values.shape not in {(self._env.num_envs, n) for n in (8, 9, 13)}
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Expected a finite stroke array with 8, 9 or 13 columns per environment"
            )
        if np.any(values[:, (0, 1, 7)] < 0) or np.any(values[:, 1] <= 0):
            raise ValueError("Stroke duration must be positive and leads non-negative")
        if self.stroke_mode != "hermite" and values.shape[1] == 13 and np.any(values[:, 9:]):
            raise ValueError("Terminal pose offsets require hermite mode")
        self.strokes[:, :8] = values[:, :8]
        self.strokes[:, 8] = (
            values[:, 8]
            if values.shape[1] >= 9
            else (-0.01 if self.stroke_mode == "hermite" else 0.0)
        )
        self.strokes[:, 9:] = values[:, 9:] if values.shape[1] == 13 else 0

    def _stroke_goal(self, i: int, joint: int, flight: bool) -> float:
        lead, duration, yaw, q2, q3, q4, tilt, tilt_lead, offset = self.strokes[i, :9]
        if flight:
            return 0.0
        if joint == 5:
            if self._stroke_time(i) < self._arrival[i] - tilt_lead:
                return 0.0
            target = np.clip(self._preparation_base[i, 5] - self.side[i] * tilt, *self._limits[5])
            goal = 10 * (target - self._reference[i, 5])
        elif (
            self._arrival[i] + offset - lead
            <= self._stroke_time(i)
            < self._arrival[i] + offset - lead + duration
        ):
            goal = {1: -self.side[i] * yaw, 2: q2, 3: q3, 4: q4}[joint]
        else:
            goal = 0.0
        # Reserve tracking margin; acceptance still checks the full URDF limit.
        return float(
            np.clip(
                goal,
                -self.reference_speed_fraction * self._max_velocity[joint],
                self.reference_speed_fraction * self._max_velocity[joint],
            )
        )

    def _advance_swing(self, i: int, joint: int, command: float, dt: float) -> None:
        del command
        if self.stroke_mode == "hermite" and joint == 1:
            self._advance_hermite(i, joint, dt, self.episodes[i].outgoing_time is not None)
            return
        self._advance_velocity(
            i,
            joint,
            self._stroke_goal(i, joint, self.episodes[i].outgoing_time is not None),
            dt,
            float(self._limits[joint, 0]),
            float(self._limits[joint, 1]),
            self._max_acceleration[joint],
        )

    def _advance_adjustments(self, i: int, dt: float, flight: bool) -> None:
        self._reference[i, 0] = self._preparation_base[i, 0]
        self._reference_velocity[i, 0] = 0
        self._reference_acceleration[i, 0] = 0
        for joint in (2, 3, 4):
            if self.stroke_mode == "hermite":
                self._advance_hermite(i, joint, dt, flight)
                continue
            self._advance_velocity(
                i,
                joint,
                self._stroke_goal(i, joint, flight),
                dt,
                float(self._limits[joint, 0]),
                float(self._limits[joint, 1]),
                self._max_acceleration[joint],
            )
        self._peak_offset[i] = np.maximum(
            self._peak_offset[i],
            np.abs(self._reference[i, (0, 2, 3, 4)] - self._preparation_base[i, (0, 2, 3, 4)]),
        )

    def _stroke_time(self, i: int) -> float:
        return float(self.elapsed[i])

    def _stroke_clock(self, i: int, dt: float) -> tuple[float, float, float]:
        """Reference time, clock rate and rate acceleration for the chain rule."""
        return self._stroke_time(i) + dt, 1.0, 0.0

    def _advance_hermite(self, i: int, joint: int, dt: float, flight: bool) -> None:
        """Backswing continuously from rest, returning with a specified velocity."""
        lead, duration, yaw, q2, q3, q4, _, _, offset = self.strokes[i, :9]
        impact = self._arrival[i] + offset
        start = max(self.prepared_time[i], impact - lead)
        travel = impact - start
        now, clock_rate, clock_acceleration = self._stroke_clock(i, dt)
        if flight or now > impact + duration:
            self._advance_velocity(
                i,
                joint,
                0.0,
                dt,
                float(self._limits[joint, 0]),
                float(self._limits[joint, 1]),
                self._max_acceleration[joint],
            )
            return
        if now <= start or travel <= 0:
            self._set_stroke_reference(i, joint, self._preparation_base[i, joint], 0.0, 0.0)
            return
        terminal_velocity = {1: -self.side[i] * yaw, 2: q2, 3: q3, 4: q4}[joint]
        terminal_velocity = float(
            np.clip(
                terminal_velocity,
                -self.reference_speed_fraction * self._max_velocity[joint],
                self.reference_speed_fraction * self._max_velocity[joint],
            )
        )
        terminal_offset = self.strokes[i, 8 + joint]
        if joint == 1:
            terminal_offset *= -self.side[i]
        if now <= impact:
            phase = (now - start) / travel
            position = (
                self._preparation_base[i, joint]
                + travel * terminal_velocity * (phase**3 - phase**2)
                + terminal_offset * (3 * phase**2 - 2 * phase**3)
            )
            velocity = (
                terminal_velocity * (3 * phase**2 - 2 * phase)
                + terminal_offset * (6 * phase - 6 * phase**2) / travel
            )
            acceleration = (
                terminal_velocity * (6 * phase - 2) / travel
                + terminal_offset * (6 - 12 * phase) / travel**2
            )
        else:
            position = (
                self._preparation_base[i, joint]
                + terminal_offset
                + terminal_velocity * (now - impact)
            )
            velocity = terminal_velocity
            acceleration = 0.0
        self._set_stroke_reference(
            i,
            joint,
            position,
            velocity * clock_rate,
            acceleration * clock_rate**2 + velocity * clock_acceleration,
        )

    def _set_stroke_reference(
        self, i: int, joint: int, position: float, velocity: float, acceleration: float
    ) -> None:
        if not self._limits[joint, 0] <= position <= self._limits[joint, 1]:
            self._boundary_failed[i] = True
            self.episodes[i].fail("trajectory_reference_boundary")
            return
        self._reference[i, joint] = position
        self._reference_velocity[i, joint] = velocity
        self._reference_acceleration[i, joint] = acceleration

    def _observe_substep(self, backend: object) -> None:
        state = self.backend.get_state(("qpos", "qvel"))
        position = state["qpos"][:, self._qpos_ids]
        velocity = state["qvel"][:, self._qvel_ids]
        sensor = self._effort_sensors.read()
        effort = np.column_stack((sensor[:, :6], sensor[:, -1]))
        active = np.array([not e.done for e in self.episodes])
        self._peak_effort[active] = np.maximum(self._peak_effort[active], np.abs(effort[active]))
        self._minimum_position[active] = np.minimum(
            self._minimum_position[active], position[active]
        )
        self._maximum_position[active] = np.maximum(
            self._maximum_position[active], position[active]
        )
        tolerance = 1e-4  # Public state/sensor views are float32.
        failures = (
            np.any(
                (position < self._limits[:, 0] - tolerance)
                | (position > self._limits[:, 1] + tolerance),
                axis=1,
            ),
            np.any(np.abs(velocity) > self._max_velocity + tolerance, axis=1),
            np.any(np.abs(effort) > self._effort_limit + tolerance, axis=1),
        )
        for i in np.flatnonzero(active):
            for reason, failed in zip(
                ("urdf_position", "urdf_velocity", "urdf_effort"), failures, strict=True
            ):
                if failed[i]:
                    self._boundary_failed[i] = True
                    self.episodes[i].fail(reason)
                    break
        paddle, quaternion = self.backend.get_body_pose_w(self._paddle_id)
        paddle = paddle[:, 0].astype(np.float64)
        normal = _normal(quaternion[:, 0])
        paddle_velocity = (paddle - self._previous_paddle) / self._env.physics_dt
        ball = state["qpos"][:, self._ball_qpos]
        was_hit = np.array([episode.hit for episode in self.episodes])
        was_crossed = np.array([episode.crossed_net for episode in self.episodes])
        for i in np.flatnonzero(active & self.ready):
            if not self._swing_started[i]:
                self._swing_lower[i] = self._previous_paddle[i]
                self._swing_upper[i] = self._previous_paddle[i]
                self._swing_started[i] = True
                self._backswing_start_x[i] = self._previous_paddle[i, 0]
            if not self.episodes[i].hit:
                self._backswing_backward[i] = max(
                    self._backswing_backward[i], self._backswing_start_x[i] - paddle[i, 0]
                )
            if self.episodes[i].outgoing_time is None:
                self._swing_path[i] += np.linalg.norm(paddle[i] - self._previous_paddle[i])
                self._swing_lower[i] = np.minimum(self._swing_lower[i], paddle[i])
                self._swing_upper[i] = np.maximum(self._swing_upper[i], paddle[i])
            elif np.isnan(self._followthrough_end_speed[i]):
                outgoing_time = self.episodes[i].outgoing_time
                assert outgoing_time is not None
                # A fixed 200 ms after departure, independent of ball flight duration.
                self._followthrough_path[i] += np.linalg.norm(paddle[i] - self._previous_paddle[i])
                self._followthrough_lower[i] = np.minimum(self._followthrough_lower[i], paddle[i])
                self._followthrough_upper[i] = np.maximum(self._followthrough_upper[i], paddle[i])
                if self.elapsed[i] + self._env.physics_dt >= outgoing_time + 0.2:
                    self._followthrough_end_speed[i] = np.linalg.norm(paddle_velocity[i])
        if self.record_trace:
            self.trace.append(
                {
                    "time": float(self.elapsed[0] + self._env.physics_dt),
                    "qpos": state["qpos"].copy(),
                    "qvel": state["qvel"].copy(),
                    "effort": effort.copy(),
                    "paddle": paddle.copy(),
                    "normal": normal.copy(),
                    "contacts": self._contacts.read().copy(),
                    "ready": self.ready.copy(),
                    "reference": self._reference.copy(),
                    "reference_velocity": self._reference_velocity.copy(),
                }
            )
        super()._observe_substep(backend)
        for i in np.flatnonzero(active):
            episode = self.episodes[i]
            if self.preparation_reach is not None and episode.hit and not self.ready[i]:
                episode.fail("hit_before_preparation")
            if episode.hit and not was_hit[i]:
                self._hit_paddle_velocity[i] = paddle_velocity[i]
                self._hit_paddle_normal[i] = normal[i]
            if episode.hit and (
                episode.outgoing_time is None or episode.outgoing_time >= self.elapsed[i]
            ):
                self._contact_face_tilt[i] = max(
                    self._contact_face_tilt[i], float(np.arcsin(np.clip(abs(normal[i, 2]), 0, 1)))
                )
                self._contact_velocity_elevation[i] = max(
                    self._contact_velocity_elevation[i],
                    float(np.arctan2(abs(paddle_velocity[i, 2]), max(0.01, paddle_velocity[i, 0]))),
                )
            if episode.outgoing_time is not None and episode.outgoing_time >= self.elapsed[i]:
                self._outgoing_paddle[i] = paddle[i]
                self._followthrough_lower[i] = paddle[i]
                self._followthrough_upper[i] = paddle[i]
            if episode.outgoing_time is not None:
                self._return_apex[i] = np.fmax(self._return_apex[i], ball[i, 2])
            if episode.crossed_net and not was_crossed[i]:
                edge = episode.net_half_thickness + episode.ball_radius
                previous = self._previous_ball[i]
                fraction = (edge - previous[0]) / (ball[i, 0] - previous[0])
                height = previous[2] + fraction * (ball[i, 2] - previous[2])
                self._net_clearance[i] = height - episode.ball_radius - episode.net_top_z
        self._previous_paddle[:] = paddle
        self._previous_ball[:] = ball

    def episode_report(self, env_idx: int) -> dict[str, object]:
        return {
            **super().episode_report(env_idx),
            "stroke": self.strokes[env_idx].tolist(),
            "stroke_mode": self.stroke_mode,
            "preparation_reach": self.preparation_reach,
            "reference_speed_fraction": self.reference_speed_fraction,
            "preparation_position_gain": self.preparation_position_gain,
            "preparation_side_deadband": self.preparation_side_deadband,
            "urdf_boundary_passed": not bool(self._boundary_failed[env_idx]),
            "peak_effort_rail_arm_and_joint5": self._peak_effort[env_idx].tolist(),
            "minimum_joint_position": self._minimum_position[env_idx].tolist(),
            "maximum_joint_position": self._maximum_position[env_idx].tolist(),
            "motion": {
                "backswing_backward_m": float(self._backswing_backward[env_idx]),
                "contact_face_tilt_max_rad": float(self._contact_face_tilt[env_idx]),
                "contact_velocity_elevation_max_rad": float(
                    self._contact_velocity_elevation[env_idx]
                ),
                "followthrough_window_s": 0.2,
                "followthrough_path_m": float(self._followthrough_path[env_idx]),
                "followthrough_forward_m": float(
                    self._followthrough_upper[env_idx, 0] - self._outgoing_paddle[env_idx, 0]
                ),
                "followthrough_span_m": (
                    self._followthrough_upper[env_idx] - self._followthrough_lower[env_idx]
                ).tolist(),
                "followthrough_end_speed_m_s": (
                    float(self._followthrough_end_speed[env_idx])
                    if np.isfinite(self._followthrough_end_speed[env_idx])
                    else None
                ),
                "hit_paddle_velocity": (
                    self._hit_paddle_velocity[env_idx].tolist()
                    if self.episodes[env_idx].hit
                    else None
                ),
                "hit_paddle_normal": (
                    self._hit_paddle_normal[env_idx].tolist()
                    if self.episodes[env_idx].hit
                    else None
                ),
                "swing_path_m": float(self._swing_path[env_idx]),
                "swing_span_m": (self._swing_upper[env_idx] - self._swing_lower[env_idx]).tolist(),
                "return_apex_above_table_m": (
                    float(self._return_apex[env_idx] - self.episodes[env_idx].table_z)
                    if np.isfinite(self._return_apex[env_idx])
                    else None
                ),
                "net_bottom_clearance_m": (
                    float(self._net_clearance[env_idx])
                    if np.isfinite(self._net_clearance[env_idx])
                    else None
                ),
            },
        }
