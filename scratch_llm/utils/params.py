"""Parameter-count helpers."""

from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch module.
        trainable_only: If True, count only parameters with requires_grad.

    Returns:
        Number of parameters.
    """

    raise NotImplementedError("Implement parameter counting")
