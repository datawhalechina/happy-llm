"""Attention layer contracts."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from scratch_llm.config import ModelConfig


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped-query attention.

    Args:
        x: Tensor shaped (batch, seq_len, n_kv_heads, head_dim).
        n_rep: Number of repeats per key/value head.

    Returns:
        Tensor shaped (batch, seq_len, n_kv_heads * n_rep, head_dim).
    """

    raise NotImplementedError("Implement grouped-query KV head repetition")


class CausalSelfAttention(nn.Module):
    """Masked self-attention with optional grouped-query attention.

    Args:
        config: ModelConfig with dim, n_heads, n_kv_heads, dropout, and
            max_seq_len.

    Expected forward input:
        x: Tensor shaped (batch, seq_len, dim).
        freqs_cos: RoPE cosine table shaped (seq_len, head_dim / 2).
        freqs_sin: RoPE sine table shaped (seq_len, head_dim / 2).
        attention_mask: Optional bool/int tensor shaped (batch, seq_len), where
            truthy values are valid tokens.

    Expected forward output:
        Tensor shaped (batch, seq_len, dim).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        raise NotImplementedError("Create q/k/v/output projections, dropout, and fallback causal mask")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run masked attention over the sequence."""

        raise NotImplementedError("Implement qkv projection, RoPE, causal attention, and output projection")
