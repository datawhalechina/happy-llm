"""Learning-rate schedules."""

from __future__ import annotations


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute warmup plus cosine-decay learning rate.

    Args:
        step: Current optimizer step, starting at 0.
        total_steps: Total planned optimizer steps.
        base_lr: Peak learning rate after warmup.
        warmup_steps: Number of linear warmup steps.
        min_lr_ratio: Final LR as base_lr * min_lr_ratio.

    Returns:
        Learning rate for this step.
    """

    raise NotImplementedError("Implement linear warmup and cosine decay")
