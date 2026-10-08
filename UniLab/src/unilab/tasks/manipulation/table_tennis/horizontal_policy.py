"""Start residual PPO at the measured reference, then learn normally."""

from typing import Any

import torch
from rsl_rl.models import MLPModel


class ReferenceResidualModel(MLPModel):
    """Zero initial action mean reproduces the nominal clock and zero residuals.

    The Gaussian still explores; all layers remain trainable by upstream PPO.
    Initialization is part of the actor declaration in the saved owner config.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        head = self.mlp[-1]
        if not isinstance(head, torch.nn.Linear) or head.out_features != 6:
            raise ValueError("Reference residual initialization requires a six-action MLP head")
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
