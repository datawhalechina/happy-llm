"""Top-level decoder-only language model contract."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from scratch_llm.config import ModelConfig


class ScratchLLM(nn.Module):
    """A small LLaMA-style decoder-only language model.

    Args:
        config: ModelConfig for embedding, blocks, norm, and LM head.

    Expected forward input:
        input_ids: Token IDs shaped (batch, seq_len).
        labels: Optional target IDs shaped (batch, seq_len). Use -100 or
            pad_token_id for ignored positions.
        attention_mask: Optional bool/int tensor shaped (batch, seq_len).

    Expected forward output:
        A dictionary with:
        logits: Tensor shaped (batch, seq_len, vocab_size) during training, or
            (batch, 1, vocab_size) if you optimize inference later.
        loss: Optional scalar or per-token loss, depending on your loop design.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        raise NotImplementedError("Create embeddings, decoder blocks, final norm, LM head, and RoPE buffers")

    def init_weights(self, module: nn.Module) -> None:
        """Initialize Linear and Embedding weights.

        Args:
            module: Submodule passed by nn.Module.apply.
        """

        raise NotImplementedError("Implement normal initialization and zero biases")

    def prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Normalize attention masks to bool shape (batch, seq_len).

        Args:
            attention_mask: None, 2D, 3D, or 4D mask from callers/tokenizers.
            input_ids: Token IDs used to validate final mask shape.

        Returns:
            None or a bool tensor shaped exactly like input_ids.
        """

        raise NotImplementedError("Implement attention mask shape normalization")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Run the Transformer and optionally compute language-model loss."""

        raise NotImplementedError("Implement embeddings, blocks, norm, logits, and optional CE loss")
