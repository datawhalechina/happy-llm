"""Checkpoint contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped by DataParallel/DDP/compile.

    Args:
        model: A torch module, possibly wrapped.

    Returns:
        The module whose state_dict should be saved.
    """

    raise NotImplementedError("Implement wrapper unwrapping")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    step: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save model and optional optimizer state.

    Args:
        path: Output checkpoint file path.
        model: Model whose state_dict will be saved.
        optimizer: Optional optimizer to resume training.
        step: Optional global optimizer step.
        extra: Optional metadata, such as config dicts.
    """

    raise NotImplementedError("Implement torch.save checkpoint payload")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model and optional optimizer state.

    Args:
        path: Checkpoint file path.
        model: Model receiving the saved state_dict.
        optimizer: Optional optimizer to restore.
        map_location: Device mapping used by torch.load.
        strict: Whether model.load_state_dict should enforce exact keys.

    Returns:
        Remaining checkpoint metadata, such as step and extra.
    """

    raise NotImplementedError("Implement torch.load plus state restoration")
