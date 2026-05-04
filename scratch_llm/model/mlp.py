"""Feed-forward layer contracts."""

from __future__ import annotations

import torch
from torch import nn


def derive_swiglu_hidden_dim(
    dim: int,
    hidden_dim: int | None = None,
    multiple_of: int = 64,
) -> int:
    """Derive the hidden size used by LLaMA-style SwiGLU MLPs.

    Args:
        dim: Residual stream hidden size.
        hidden_dim: Explicit hidden size. If provided, return it unchanged.
        multiple_of: Round derived size up to this multiple.

    Returns:
        Final hidden size.
    """

    raise NotImplementedError("Implement the 4x -> 2/3 -> rounded SwiGLU rule")


class SwiGLU(nn.Module):
    """LLaMA-style gated MLP.

    Args:
        dim: Residual stream hidden size.
        hidden_dim: Explicit hidden size or None to derive one.
        multiple_of: Round derived hidden size to this multiple.
        dropout: Dropout probability applied after the down projection.

    Expected forward input:
        x: Tensor shaped (batch, seq_len, dim).

    Expected forward output:
        Tensor shaped (batch, seq_len, dim).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None,
        multiple_of: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.dropout_p = dropout
        raise NotImplementedError("Create gate/up/down projections and dropout")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward transformation."""

        raise NotImplementedError("Implement silu(w1(x)) * w3(x), then w2")
