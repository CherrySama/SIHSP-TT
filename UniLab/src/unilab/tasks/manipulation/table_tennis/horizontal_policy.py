"""Start residual PPO at the measured reference, then learn normally."""

import math
from typing import Any

import torch
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories


class ExpandableObservationMLPModel(MLPModel):
    """Warm-start a model after inserting one zero-initialized observation feature."""

    def __init__(
        self,
        *args: Any,
        inserted_observation_index: int | None = None,
        inserted_observation_dim: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if inserted_observation_dim < 1:
            raise ValueError("inserted_observation_dim must be positive")
        if inserted_observation_index is not None and not (
            0 <= inserted_observation_index <= self.mlp[0].in_features - inserted_observation_dim
        ):
            raise ValueError("inserted_observation_index is outside the observation vector")
        self._inserted_observation_index = inserted_observation_index
        self._inserted_observation_dim = inserted_observation_dim

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        state_dict = dict(state_dict)
        first_weight = state_dict.get("mlp.0.weight")
        current_weight = self.mlp[0].weight
        if (
            isinstance(first_weight, torch.Tensor)
            and first_weight.shape[0] == current_weight.shape[0]
            and first_weight.shape[1] + self._inserted_observation_dim == current_weight.shape[1]
        ):
            inserted = (
                first_weight.shape[1]
                if self._inserted_observation_index is None
                else self._inserted_observation_index
            )
            expanded = current_weight.detach().clone()
            expanded.zero_()
            expanded[:, :inserted] = first_weight[:, :inserted]
            expanded[:, inserted + self._inserted_observation_dim :] = first_weight[:, inserted:]
            state_dict["mlp.0.weight"] = expanded
            for name, fill in (
                ("obs_normalizer._mean", 0.0),
                ("obs_normalizer._var", 1.0),
                ("obs_normalizer._std", 1.0),
            ):
                saved = state_dict.get(name)
                current = self.state_dict().get(name)
                if isinstance(saved, torch.Tensor) and isinstance(current, torch.Tensor):
                    extended = current.detach().clone().fill_(fill)
                    extended[..., :inserted] = saved[..., :inserted]
                    extended[..., inserted + self._inserted_observation_dim :] = saved[..., inserted:]
                    state_dict[name] = extended
        return super().load_state_dict(state_dict, strict=strict)


class ReferenceResidualModel(ExpandableObservationMLPModel):
    """Zero initial action mean reproduces the nominal clock and zero residuals.

    The Gaussian still explores; all layers remain trainable by upstream PPO.
    Initialization is part of the actor declaration in the saved owner config.
    """

    def __init__(
        self,
        *args: Any,
        action_init_std: list[float] | tuple[float, ...] | None = None,
        reset_action_std_on_load: bool = False,
        reset_action_mean_indices: list[int] | tuple[int, ...] = (),
        lateral_strength_feature_index: int | None = None,
        lateral_strength_start: float = 0.25 / 0.55,
        lateral_strength_full: float = 0.50 / 0.55,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        head = self.mlp[-1]
        if not isinstance(head, torch.nn.Linear) or head.out_features < 6:
            raise ValueError("Reference residual initialization requires at least six actions")
        self._action_dim = head.out_features
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        self._reset_action_std_on_load = reset_action_std_on_load
        if any(index < 0 or index >= self._action_dim for index in reset_action_mean_indices):
            raise ValueError("reset_action_mean_indices must contain valid action indices")
        self._reset_action_mean_indices = tuple(dict.fromkeys(reset_action_mean_indices))
        if lateral_strength_feature_index is not None and lateral_strength_feature_index < 0:
            raise ValueError("lateral_strength_feature_index must be non-negative")
        if not 0 <= lateral_strength_start < lateral_strength_full:
            raise ValueError("Invalid lateral strength prior thresholds")
        self._lateral_strength_feature_index = lateral_strength_feature_index
        self._lateral_strength_start = lateral_strength_start
        self._lateral_strength_full = lateral_strength_full
        self._configured_action_std: torch.Tensor | None = None
        if action_init_std is not None:
            if len(action_init_std) != self._action_dim or any(
                not math.isfinite(value) or value <= 0 for value in action_init_std
            ):
                raise ValueError(
                    "action_init_std must contain one finite positive value per action"
                )
            self._configured_action_std = torch.tensor(
                action_init_std,
                dtype=self.distribution.std_param.dtype,
                device=self.distribution.std_param.device,
            )
            self._restore_configured_action_std()

    def forward(
        self,
        obs: Any,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        output = self.mlp(latent)
        if self._lateral_strength_feature_index is not None:
            lateral = obs["policy"][..., self._lateral_strength_feature_index].abs()
            prior = torch.clamp(
                (lateral - self._lateral_strength_start)
                / (self._lateral_strength_full - self._lateral_strength_start),
                0.0,
                1.0,
            )
            output = output.clone()
            output[..., 3] = output[..., 3] + prior
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def _restore_configured_action_std(self) -> None:
        if self._configured_action_std is None:
            return
        with torch.no_grad():
            self.distribution.std_param.copy_(self._configured_action_std)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        state_dict = dict(state_dict)
        head = self.mlp[-1]
        assert isinstance(head, torch.nn.Linear)
        head_prefix = f"mlp.{len(self.mlp) - 1}"
        saved_weight = state_dict.get(f"{head_prefix}.weight")
        if (
            isinstance(saved_weight, torch.Tensor)
            and saved_weight.shape[1] == head.in_features
            and saved_weight.shape[0] < head.out_features
        ):
            expanded = head.weight.detach().clone()
            expanded.zero_()
            expanded[: saved_weight.shape[0]] = saved_weight
            state_dict[f"{head_prefix}.weight"] = expanded
            saved_bias = state_dict.get(f"{head_prefix}.bias")
            if isinstance(saved_bias, torch.Tensor) and saved_bias.shape[0] < head.out_features:
                expanded_bias = head.bias.detach().clone()
                expanded_bias.zero_()
                expanded_bias[: saved_bias.shape[0]] = saved_bias
                state_dict[f"{head_prefix}.bias"] = expanded_bias
            saved_std = state_dict.get("distribution.std_param")
            current_std = self.state_dict().get("distribution.std_param")
            if (
                isinstance(saved_std, torch.Tensor)
                and isinstance(current_std, torch.Tensor)
                and saved_std.shape[0] < current_std.shape[0]
            ):
                expanded_std = current_std.detach().clone()
                expanded_std[: saved_std.shape[0]] = saved_std
                state_dict["distribution.std_param"] = expanded_std
        result = super().load_state_dict(state_dict, strict=strict)
        if self._reset_action_std_on_load:
            self._restore_configured_action_std()
        if self._reset_action_mean_indices:
            head = self.mlp[-1]
            assert isinstance(head, torch.nn.Linear)
            with torch.no_grad():
                head.weight[list(self._reset_action_mean_indices)] = 0
                head.bias[list(self._reset_action_mean_indices)] = 0
        return result
