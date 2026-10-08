"""Compatibility entrypoints for scripted URDF-bounded stroke diagnosis."""

from unilab.tasks.manipulation.table_tennis.stroke_control import (
    DEFAULT_URDF,
    JOINT_NAMES,
    HorizontalStrokeCriteria,
    JointBoundary,
    build_feasibility_env_cfg,
    compact_motion_cost,
    feasibility_scene_path,
    horizontal_stroke_assessment,
    urdf_boundaries,
)
from unilab.tasks.manipulation.table_tennis.stroke_control import (
    PantheraStrokeAction as FeasibilityAction,
)
from unilab.tasks.manipulation.table_tennis.stroke_control import (
    PantheraStrokeActionCfg as FeasibilityActionCfg,
)

__all__ = [
    "DEFAULT_URDF",
    "JOINT_NAMES",
    "HorizontalStrokeCriteria",
    "JointBoundary",
    "FeasibilityAction",
    "FeasibilityActionCfg",
    "build_feasibility_env_cfg",
    "compact_motion_cost",
    "feasibility_scene_path",
    "horizontal_stroke_assessment",
    "urdf_boundaries",
]
