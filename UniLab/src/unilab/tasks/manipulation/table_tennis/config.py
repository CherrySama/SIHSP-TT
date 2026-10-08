"""Manager-Based Panthera control with fixed or random incoming feeds."""

from __future__ import annotations

from typing import Literal

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.envs import ManagerBasedRlEnvCfg, mdp
from unilab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    TerminationTermCfg,
)
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.manipulation.table_tennis.ball_launcher import (
    BallLauncherCfg,
    ServeLauncherCfg,
    launch_random_ball,
)
from unilab.tasks.manipulation.table_tennis.manager_terms import (
    PantheraIncrementalPositionActionCfg,
    PantheraJointSpeedLimit,
)

PANTHERA_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
PANTHERA_ACTUATOR_NAMES = tuple(f"joint{index}_position" for index in range(1, 7))
PANTHERA_ACTUATOR_SELECTORS = tuple(f"joint{index}.*" for index in range(1, 7))


def build_panthera_table_tennis_env_cfg(
    *,
    ball_feed: Literal["random", "fixed"] = "random",
    launcher_cfg: BallLauncherCfg | ServeLauncherCfg | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build the unregistered reset/control environment; fixed feeds remain available."""
    if ball_feed not in ("random", "fixed"):
        raise ValueError(f"Unsupported Panthera ball_feed: {ball_feed!r}")
    if ball_feed == "fixed" and launcher_cfg is not None:
        raise ValueError("launcher_cfg only applies to random feeds")
    launch_event = (
        EventTermCfg(
            func=launch_random_ball,
            mode="reset",
            params={
                "launcher_cfg": launcher_cfg if launcher_cfg is not None else ServeLauncherCfg(),
                "asset_cfg": SceneEntityCfg("ball"),
            },
        )
        if ball_feed == "random"
        else EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {
                    "x": (-2.5, -2.5),
                    "y": (0.0, 0.0),
                    "z": (1.6, 1.6),
                },
                "asset_cfg": SceneEntityCfg("ball"),
            },
        )
    )
    robot_joints = SceneEntityCfg("robot", joint_names=PANTHERA_JOINT_NAMES)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "panthera" / "scene.xml"),
            default_keyframe_name="home",
            entities={
                "robot": EntityCfg(
                    root_body_name="rail_carriage",
                    joint_names=PANTHERA_JOINT_NAMES,
                    body_names=("rail_carriage", "link6", "paddle_frame"),
                    geom_names=("paddle_contact",),
                    actuator_names=PANTHERA_ACTUATOR_NAMES,
                ),
                "ball": EntityCfg(
                    root_body_name="ball",
                    body_names=("ball",),
                    geom_names=("ball_geom",),
                ),
            },
        ),
        sim_dt=0.00025,
        ctrl_dt=0.01,
        max_episode_seconds=2.0,
        seed=7,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(
                        func=mdp.joint_pos_rel,
                        params={"asset_cfg": robot_joints},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=mdp.joint_vel_rel,
                        params={"asset_cfg": robot_joints},
                    ),
                }
            )
        },
        actions={
            "arm_target_delta": PantheraIncrementalPositionActionCfg(
                entity_name="robot",
                actuator_names=PANTHERA_ACTUATOR_SELECTORS,
                max_target_velocity=1.0,
            )
        },
        events={
            "reset_scene_to_default": EventTermCfg(
                func=mdp.reset_scene_to_default,
                mode="reset",
            ),
            "launch_ball": launch_event,
        },
        rewards={},
        terminations={
            "joint_speed": TerminationTermCfg(
                func=PantheraJointSpeedLimit,
                params={
                    "entity_name": "robot",
                    "joint_names": PANTHERA_JOINT_NAMES,
                    "speed_limit": 1.0,
                    "tolerance": 0.2,
                    "duration_seconds": 0.02,
                },
            ),
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        policy_observation_group="policy",
    )


__all__ = [
    "PANTHERA_ACTUATOR_NAMES",
    "PANTHERA_ACTUATOR_SELECTORS",
    "PANTHERA_JOINT_NAMES",
    "build_panthera_table_tennis_env_cfg",
]
