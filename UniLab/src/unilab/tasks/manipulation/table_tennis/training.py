"""Panthera return registration and cold-path owner config materialization.

Importing the registry does not require the optional table-tennis prediction
package. The selected task resolves that dependency and its prepared scene only
when its config is materialized. All manager declarations stay in owner YAML.
"""

from pathlib import Path

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env


def build_panthera_return_env_cfg() -> ManagerBasedRlEnvCfg:
    from omegaconf import OmegaConf

    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.tasks.manipulation.table_tennis.prepared_scene import prepared_scene_path

    owner_path = Path(__file__).parents[3] / "conf/ppo/task/panthera_return/base.yaml"
    owner = OmegaConf.to_container(OmegaConf.load(owner_path), resolve=True)
    if not isinstance(owner, dict):
        raise TypeError("PantheraReturn owner YAML must contain a mapping")
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, {**owner["env"], "rewards": owner["reward"]})
    if cfg.scene is None:
        raise ValueError("PantheraReturn owner YAML must declare its scene")
    cfg.scene.model_file = prepared_scene_path()
    return cfg


registry.register_env_config("PantheraReturn", build_panthera_return_env_cfg)
registry.register_env("PantheraReturn", make_manager_based_rl_env, sim_backend="mujoco")


def build_panthera_horizontal_env_cfg() -> ManagerBasedRlEnvCfg:
    from omegaconf import OmegaConf

    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.tasks.manipulation.table_tennis.stroke_control import (
        PantheraStrokeActionCfg,
        feasibility_scene_path,
    )

    directory = Path(__file__).parents[3] / "conf/ppo/task"
    owner = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(directory / "panthera_return/base.yaml"),
            OmegaConf.load(directory / "panthera_horizontal/base.yaml"),
        ),
        resolve=True,
    )
    assert isinstance(owner, dict)
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, {**owner["env"], "rewards": owner["reward"]})
    assert cfg.scene is not None
    action = cfg.actions["return_control"]
    assert isinstance(action, PantheraStrokeActionCfg)
    cfg.scene.model_file = feasibility_scene_path(action.urdf_path)
    return cfg


registry.register_env_config("PantheraHorizontalReturn", build_panthera_horizontal_env_cfg)
registry.register_env("PantheraHorizontalReturn", make_manager_based_rl_env, sim_backend="mujoco")

__all__ = ["build_panthera_return_env_cfg", "build_panthera_horizontal_env_cfg"]
