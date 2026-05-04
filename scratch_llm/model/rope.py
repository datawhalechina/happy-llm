"""Rotary position embedding contracts."""

from __future__ import annotations

from typing import Optional

import torch


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device | str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine RoPE tables.

    Args:
        head_dim: Per-head hidden dimension. Must be even.
        max_seq_len: Number of positions to precompute.
        theta: RoPE base frequency.
        device: Optional torch device for created tensors.

    Returns:
        Tuple (freqs_cos, freqs_sin), each shaped (max_seq_len, head_dim / 2).
    """

    raise NotImplementedError("Implement RoPE frequency precomputation")


def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a RoPE table so it can broadcast against q/k halves.

    Args:
        freqs: Tensor shaped (seq_len, head_dim / 2).
        x: Query or key half tensor shaped (batch, seq_len, heads, head_dim / 2).

    Returns:
        View of freqs shaped (1, seq_len, 1, head_dim / 2).
    """

    raise NotImplementedError("Implement broadcast reshaping for RoPE")


def apply_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to query and key tensors.

    Args:
        q: Query tensor shaped (batch, seq_len, n_heads, head_dim).
        k: Key tensor shaped (batch, seq_len, n_kv_heads, head_dim).
        freqs_cos: Cosine table for current sequence length.
        freqs_sin: Sine table for current sequence length.

    Returns:
        Tuple (rotated_q, rotated_k) with original shapes and dtypes.
    """

    raise NotImplementedError("Implement real/imaginary RoPE rotation")
