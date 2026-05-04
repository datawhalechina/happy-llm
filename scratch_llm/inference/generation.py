"""Autoregressive generation contracts."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from scratch_llm.config import GenerationConfig


def top_k_filter(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    """Mask logits outside the top-k choices.

    Args:
        logits: Tensor shaped (batch, vocab_size).
        top_k: Number of highest logits to keep. None keeps all.

    Returns:
        Filtered logits with non-top-k entries set to -inf.
    """

    raise NotImplementedError("Implement top-k filtering")


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Choose the next token from logits.

    Args:
        logits: Tensor shaped (batch, vocab_size).
        temperature: 0 for greedy decoding, otherwise softmax temperature.
        top_k: Optional top-k filter before sampling.

    Returns:
        Token IDs shaped (batch, 1).
    """

    raise NotImplementedError("Implement greedy and multinomial sampling")


@torch.inference_mode()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate continuation tokens.

    Args:
        model: ScratchLLM or compatible causal language model.
        input_ids: Prompt token IDs shaped (batch, prompt_len).
        config: GenerationConfig.
        attention_mask: Optional mask shaped (batch, prompt_len).

    Returns:
        New token IDs shaped (batch, generated_len), excluding the prompt.
    """

    raise NotImplementedError("Implement autoregressive decoding loop without KV cache first")
