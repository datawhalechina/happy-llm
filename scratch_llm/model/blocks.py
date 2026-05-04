"""Decoder block contract."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from scratch_llm.config import ModelConfig


class DecoderBlock(nn.Module):
    """One pre-norm Transformer decoder layer.

    Args:
        layer_id: Zero-based layer index, useful for debugging/logging.
        config: ModelConfig shared by all layers.

    Expected forward input:
        x: Tensor shaped (batch, seq_len, dim).
        freqs_cos: RoPE cosine table shaped (seq_len, head_dim / 2).
        freqs_sin: RoPE sine table shaped (seq_len, head_dim / 2).
        attention_mask: Optional bool/int tensor shaped (batch, seq_len).

    Expected forward output:
        Tensor shaped (batch, seq_len, dim).
    """

    def __init__(self, layer_id: int, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.layer_id = layer_id
        self.config = config
        raise NotImplementedError("Create attention, MLP, attention_norm, and ffn_norm")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention residual, then MLP residual."""

        raise NotImplementedError("Implement the two residual branches")
