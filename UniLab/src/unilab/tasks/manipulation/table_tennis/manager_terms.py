"""Manager terms owned by the Panthera table-tennis task."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers import ActionTerm, ActionTermCfg, ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_ARM_DOF = 6


def _real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{term} {name} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{term} {name} must be {relation} {minimum}")
    return result


def _names(term: str, name: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a sequence of {_ARM_DOF} strings")
    result = tuple(value)
    if len(result) != _ARM_DOF:
        raise ValueError(f"{term} {name} must contain {_ARM_DOF} names, got {len(result)}")
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{term} {name} must contain non-empty strings")
    if len(set(result)) != _ARM_DOF:
        raise ValueError(f"{term} {name} must contain unique names")
    return result


def _pair(term: str, name: str, value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a numeric (min, max) pair")
    if len(value) != 2:
        raise ValueError(f"{term} {name} must contain two values")
    lower = _real(term, f"{name}[0]", value[0])
    upper = _real(term, f"{name}[1]", value[1])
    if lower > upper:
        raise ValueError(f"{term} {name} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


def _env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.int32)[env_ids]
    return env_ids


@dataclass(kw_only=True)
class PantheraIncrementalPositionActionCfg(ActionTermCfg):
    """Six-axis position increments bounded by the SDK command velocity."""

    actuator_names: tuple[str, ...] | list[str]
    max_target_velocity: float
    raw_action_clip: tuple[float, float] | list[float] = (-1.0, 1.0)

    def build(self, env: ManagerBasedRlEnv) -> PantheraIncrementalPositionAction:
        return PantheraIncrementalPositionAction(self, env)


class PantheraIncrementalPositionAction(ActionTerm):
    """Integrate normalized policy deltas into rate- and range-limited targets."""

    cfg: PantheraIncrementalPositionActionCfg
    _entity: Entity

    def __init__(self, cfg: PantheraIncrementalPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        if cfg.clip is not None:
            raise NotImplementedError(
                f"{term} does not support actuator-name clip; use raw_action_clip"
            )
        actuator_selectors = _names(term, "actuator_names", cfg.actuator_names)
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(actuator_selectors)
        actuator_ids, resolved_actuators = self._entity.find_actuators(
            actuator_selectors, preserve_order=True
        )
        if len(joint_ids) != _ARM_DOF or len(actuator_ids) != _ARM_DOF:
            raise ValueError(f"{term} requires one joint and actuator per selector")
        expected_joint_names = [self._entity.joint_names[index] for index in joint_ids]
        if joint_names != expected_joint_names:
            raise ValueError(f"{term} actuator-to-joint mapping is incomplete")
        for selector, joint_name, actuator_name in zip(
            actuator_selectors, joint_names, resolved_actuators, strict=True
        ):
            if (
                re.fullmatch(selector, joint_name) is None
                or re.fullmatch(selector, actuator_name) is None
            ):
                raise ValueError(
                    f"{term} selector {selector!r} does not map one ordered joint/actuator pair"
                )

        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._joint_ids.setflags(write=False)
        actuator_ids_array = np.asarray(actuator_ids, dtype=np.intp)
        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        self._ctrl_lower = np.array(ranges[actuator_ids_array, 0], copy=True)
        self._ctrl_upper = np.array(ranges[actuator_ids_array, 1], copy=True)
        if np.any(self._ctrl_lower >= self._ctrl_upper):
            raise ValueError(f"{term} actuator control ranges must have lower < upper")

        max_velocity = _real(
            term,
            "max_target_velocity",
            cfg.max_target_velocity,
            minimum=0.0,
            strict_minimum=True,
        )
        step_dt = _real(term, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._max_target_step = max_velocity * step_dt
        self._raw_clip = _pair(term, "raw_action_clip", cfg.raw_action_clip)

        dtype = get_global_dtype()
        self._raw_action = np.zeros((env.num_envs, _ARM_DOF), dtype=dtype)
        self._clipped_action = np.zeros_like(self._raw_action)
        self._target = np.asarray(
            self._entity.data.default_joint_pos[:, self._joint_ids], dtype=dtype
        ).copy()

    @property
    def action_dim(self) -> int:
        return _ARM_DOF

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def max_target_step(self) -> float:
        return self._max_target_step

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"{type(self).__name__} expected np.ndarray actions")
        if actions.shape != self._raw_action.shape:
            raise ValueError(
                f"{type(self).__name__} expected action shape {self._raw_action.shape}, "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"{type(self).__name__} received NaN or Inf actions")
        self._raw_action[:] = actions
        np.clip(actions, self._raw_clip[0], self._raw_clip[1], out=self._clipped_action)
        self._target += self._max_target_step * self._clipped_action
        np.clip(self._target, self._ctrl_lower, self._ctrl_upper, out=self._target)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self._raw_action[ids] = 0.0
        self._clipped_action[ids] = 0.0
        self._target[ids] = self._entity.data.joint_pos[ids][:, self._joint_ids]


class PantheraJointSpeedLimit(ManagerTermBase):
    """Terminate only after measured joint speed remains clearly excessive."""

    _entity: Entity

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {
            "entity_name",
            "joint_names",
            "speed_limit",
            "tolerance",
            "duration_seconds",
        }
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = cfg.params.get("entity_name")
        if not isinstance(entity_name, str) or not entity_name:
            raise ValueError(f"{term} entity_name must be a non-empty string")
        self._entity = cast("Entity", env.scene[entity_name])
        joint_names = _names(term, "joint_names", cfg.params.get("joint_names"))
        joint_ids, resolved = self._entity.find_joints(joint_names, preserve_order=True)
        if tuple(resolved) != joint_names:
            raise ValueError(f"{term} joint selectors resolved out of order: {resolved}")
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._joint_ids.setflags(write=False)

        speed_limit = _real(
            term,
            "speed_limit",
            cfg.params.get("speed_limit"),
            minimum=0.0,
            strict_minimum=True,
        )
        tolerance = _real(term, "tolerance", cfg.params.get("tolerance"), minimum=0.0)
        duration = _real(
            term,
            "duration_seconds",
            cfg.params.get("duration_seconds"),
            minimum=0.0,
            strict_minimum=True,
        )
        step_dt = _real(term, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._threshold = speed_limit + tolerance
        self._steps_required = max(1, math.ceil(duration / step_dt))
        self._consecutive_steps = np.zeros(env.num_envs, dtype=np.int32)
        self._done = np.zeros(env.num_envs, dtype=np.bool_)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def steps_required(self) -> int:
        return self._steps_required

    @property
    def consecutive_steps(self) -> np.ndarray:
        return self._consecutive_steps

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self._consecutive_steps[ids] = 0
        self._done[ids] = False

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del env, params
        velocity = self._entity.data.joint_vel[:, self._joint_ids]
        excessive = np.any(np.abs(velocity) > self._threshold, axis=1)
        self._consecutive_steps[:] = np.where(excessive, self._consecutive_steps + 1, 0)
        self._done[:] = self._consecutive_steps >= self._steps_required
        return self._done


__all__ = [
    "PantheraIncrementalPositionAction",
    "PantheraIncrementalPositionActionCfg",
    "PantheraJointSpeedLimit",
]
