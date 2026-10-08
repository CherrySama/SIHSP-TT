"""Focused contracts for the Panthera table-tennis control terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from unisim.backend.base import SimBackend

from unilab.base.entity import EntityCfg, EntityScene
from unilab.managers import ManagerTermBaseCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.manipulation.table_tennis.manager_terms import (
    PantheraIncrementalPositionActionCfg,
    PantheraJointSpeedLimit,
)

_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
_ACTUATOR_NAMES = tuple(f"joint{index}_position" for index in range(1, 7))
_SELECTORS = tuple(f"joint{index}.*" for index in range(1, 7))
_HOME = np.asarray([0.0, 0.3, 0.3, 0.0, 0.0, 1.5], dtype=np.float32)
_RANGES = np.asarray(
    [
        [-2.4, 2.4],
        [-0.1, 3.2],
        [-0.1, 4.0],
        [-1.6, 1.6],
        [-1.7, 1.7],
        [-2.5, 2.5],
    ],
    dtype=np.float32,
)


class _PantheraBackend:
    backend_type = "fake"
    num_envs = 2
    num_actuators = 6

    def __init__(self) -> None:
        self.dof_pos = np.tile(_HOME, (self.num_envs, 1))
        self.dof_vel = np.zeros_like(self.dof_pos)

    def get_actuator_names(self) -> tuple[str, ...]:
        return _ACTUATOR_NAMES

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return _JOINT_NAMES

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return _RANGES.copy()

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        return np.asarray([_JOINT_NAMES.index(name) for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_dof_pos(self) -> np.ndarray:
        return self.dof_pos

    def get_dof_vel(self) -> np.ndarray:
        return self.dof_vel

    def get_default_dof_pos(self) -> np.ndarray:
        return _HOME.copy()

    def get_joint_range(self) -> np.ndarray:
        return _RANGES.copy()


def _env() -> tuple[ManagerBasedRlEnv, _PantheraBackend, np.ndarray]:
    backend = _PantheraBackend()
    control = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=_JOINT_NAMES,
                actuator_names=_ACTUATOR_NAMES,
            )
        },
        cast(SimBackend, backend),
        control,
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=backend.num_envs, scene=scene, step_dt=0.01),
    )
    return env, backend, control


def _action():
    env, backend, control = _env()
    cfg = PantheraIncrementalPositionActionCfg(
        entity_name="robot",
        actuator_names=_SELECTORS,
        max_target_velocity=1.0,
    )
    return cfg.build(env), backend, control


def test_incremental_action_clips_raw_delta_rate_and_joint_range() -> None:
    action, _, control = _action()
    raw = np.asarray(
        [[2.0] * 6, [-2.0] * 6],
        dtype=np.float32,
    )

    action.process_actions(raw)
    action.apply_actions()

    assert action.max_target_step == pytest.approx(0.01)
    np.testing.assert_array_equal(action.raw_action, raw)
    np.testing.assert_allclose(action.target[0], _HOME + 0.01)
    expected_second = np.maximum(_HOME - 0.01, _RANGES[:, 0])
    np.testing.assert_allclose(action.target[1], expected_second)
    np.testing.assert_allclose(control, action.target)

    for _ in range(1000):
        action.process_actions(raw)
    np.testing.assert_allclose(action.target[0], _RANGES[:, 1])
    np.testing.assert_allclose(action.target[1], _RANGES[:, 0])


def test_incremental_action_partial_reset_uses_measured_joint_position() -> None:
    action, backend, _ = _action()
    action.process_actions(np.ones((2, 6), dtype=np.float32))
    backend.dof_pos[1] = np.asarray([0.2, 0.4, 0.5, 0.1, -0.2, 1.2], dtype=np.float32)

    action.reset(np.asarray([1], dtype=np.int32))

    np.testing.assert_allclose(action.target[0], _HOME + 0.01)
    np.testing.assert_allclose(action.target[1], backend.dof_pos[1])
    np.testing.assert_array_equal(action.raw_action[0], 1.0)
    np.testing.assert_array_equal(action.raw_action[1], 0.0)


def test_incremental_action_rejects_bad_runtime_actions() -> None:
    action, _, control = _action()
    with pytest.raises(ValueError, match="expected action shape"):
        action.process_actions(np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        action.process_actions(np.full((2, 6), np.nan, dtype=np.float32))
    np.testing.assert_array_equal(control, 0.0)


def test_joint_speed_limit_requires_two_consecutive_control_steps() -> None:
    env, backend, _ = _env()
    cfg = ManagerTermBaseCfg(
        func=PantheraJointSpeedLimit,
        params={
            "entity_name": "robot",
            "joint_names": _JOINT_NAMES,
            "speed_limit": 1.0,
            "tolerance": 0.2,
            "duration_seconds": 0.02,
        },
    )
    term = PantheraJointSpeedLimit(cfg, env)
    backend.dof_vel[0, 2] = 1.21
    backend.dof_vel[1, 2] = 1.2

    np.testing.assert_array_equal(term(env), [False, False])
    np.testing.assert_array_equal(term.consecutive_steps, [1, 0])
    np.testing.assert_array_equal(term(env), [True, False])
    assert term.threshold == pytest.approx(1.2)
    assert term.steps_required == 2

    backend.dof_vel[0, 2] = 0.0
    np.testing.assert_array_equal(term(env), [False, False])
    np.testing.assert_array_equal(term.consecutive_steps, [0, 0])


def test_joint_speed_limit_partial_reset_clears_only_selected_rows() -> None:
    env, backend, _ = _env()
    cfg = ManagerTermBaseCfg(
        func=PantheraJointSpeedLimit,
        params={
            "entity_name": "robot",
            "joint_names": _JOINT_NAMES,
            "speed_limit": 1.0,
            "tolerance": 0.2,
            "duration_seconds": 0.02,
        },
    )
    term = PantheraJointSpeedLimit(cfg, env)
    backend.dof_vel[:, 0] = 1.3
    term(env)

    term.reset(np.asarray([1], dtype=np.int32))

    np.testing.assert_array_equal(term.consecutive_steps, [1, 0])
