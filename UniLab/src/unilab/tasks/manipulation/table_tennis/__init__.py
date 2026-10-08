"""Panthera task exports, loaded only when a caller selects this task."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unilab.tasks.manipulation.table_tennis.ball_launcher import (
        BallLauncherCfg as BallLauncherCfg,
    )
    from unilab.tasks.manipulation.table_tennis.ball_launcher import (
        ServeLauncherCfg as ServeLauncherCfg,
    )
    from unilab.tasks.manipulation.table_tennis.config import (
        build_panthera_table_tennis_env_cfg as build_panthera_table_tennis_env_cfg,
    )
    from unilab.tasks.manipulation.table_tennis.training import (
        build_panthera_return_env_cfg as build_panthera_return_env_cfg,
    )

_EXPORT_MODULES = {
    "BallLauncherCfg": "ball_launcher",
    "ServeLauncherCfg": "ball_launcher",
    "build_panthera_table_tennis_env_cfg": "config",
    "build_panthera_return_env_cfg": "training",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "BallLauncherCfg",
    "ServeLauncherCfg",
    "build_panthera_table_tennis_env_cfg",
    "build_panthera_return_env_cfg",
]
