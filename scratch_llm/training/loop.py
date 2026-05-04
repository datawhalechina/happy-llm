"""Training and evaluation loop contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from scratch_llm.config import TrainConfig


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Logger = Callable[[dict[str, Any]], None]


def move_batch_to_device(batch: Batch, device: str | torch.device) -> Batch:
    """Move x, y, and loss_mask tensors to a device.

    Args:
        batch: Tuple (x, y, loss_mask).
        device: Target torch device.

    Returns:
        Tuple with all tensors moved to device.
    """

    raise NotImplementedError("Implement tensor.to(device) for each batch item")


def compute_masked_loss(
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Reduce per-token loss with a binary mask.

    Args:
        per_token_loss: Loss tensor shaped (batch * seq_len,) or (batch, seq_len).
        loss_mask: 1/0 mask with a shape broadcastable to per_token_loss.

    Returns:
        Scalar average loss over positions where loss_mask is 1.
    """

    raise NotImplementedError("Implement masked mean while guarding empty masks")


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Batch],
    optimizer: Optimizer,
    config: TrainConfig,
    epoch: int,
    total_steps: int,
    start_step: int = 0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    logger: Optional[Logger] = None,
) -> int:
    """Train for one epoch.

    Args:
        model: ScratchLLM or compatible language model.
        dataloader: Iterable yielding (x, y, loss_mask).
        optimizer: Optimizer, usually AdamW.
        config: TrainConfig.
        epoch: Current epoch number.
        total_steps: Total planned optimizer steps for LR schedule.
        start_step: Global optimizer step before this epoch.
        scaler: Optional mixed-precision GradScaler.
        logger: Optional callback receiving metrics dictionaries.

    Returns:
        Updated global optimizer step.
    """

    raise NotImplementedError("Implement forward, masked loss, backward, clipping, optimizer step, and logs")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: Iterable[Batch],
    config: TrainConfig,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate masked language-model loss.

    Args:
        model: ScratchLLM or compatible language model.
        dataloader: Iterable yielding (x, y, loss_mask).
        config: TrainConfig.
        max_batches: Optional cap for quick validation.

    Returns:
        Metrics dictionary, for example {"loss": 1.23, "ppl": 3.42}.
    """

    raise NotImplementedError("Implement eval loop and perplexity calculation")
