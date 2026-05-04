"""Dataset contracts for next-token prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset


def build_causal_lm_example(
    input_ids: list[int],
    max_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one shifted causal-LM training example.

    Args:
        input_ids: Token IDs before shifting. Include BOS/EOS if desired.
        max_length: Final unshifted sequence length before creating X and Y.
        pad_token_id: Token ID used to pad short examples.

    Returns:
        A tuple (x, y, loss_mask):
        x: Input IDs with shape (max_length - 1,).
        y: Target IDs shifted one token left, shape (max_length - 1,).
        loss_mask: 1 for real target positions and 0 for padding targets.
    """

    raise NotImplementedError("Implement truncation, padding, shift, and loss mask creation")


def build_sft_loss_mask(
    input_ids: list[int],
    assistant_prefix_ids: list[int],
    eos_token_id: int,
) -> list[int]:
    """Mark only assistant answer spans for supervised fine-tuning loss.

    Args:
        input_ids: Full chat prompt token IDs.
        assistant_prefix_ids: Token IDs that identify '<assistant> starts here'.
        eos_token_id: Token ID that closes an assistant answer.

    Returns:
        A list with the same length as input_ids where assistant answer tokens
        are 1 and all prompt/system/user tokens are 0.
    """

    raise NotImplementedError("Implement assistant-span masking for SFT")


class CausalLMDataset(Dataset):
    """JSONL pretraining dataset for next-token prediction.

    Args:
        data_path: JSONL file path. Each row should contain a text field.
        tokenizer: Tokenizer object with __call__, bos_token, and pad_token_id.
        max_length: Final sequence length before shifting.
        text_key: JSON key containing raw text.
        bos_token: Optional token string to prepend. Defaults to tokenizer.bos_token.
        pad_token_id: Optional padding token ID. Defaults to tokenizer.pad_token_id.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        max_length: int,
        text_key: str = "text",
        bos_token: Optional[str] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.bos_token = bos_token
        self.pad_token_id = pad_token_id
        raise NotImplementedError("Build byte offsets and store dataset metadata")

    def __len__(self) -> int:
        """Return the number of JSONL rows."""

        raise NotImplementedError("Return the precomputed row count")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one row, tokenize it, and return (x, y, loss_mask).

        Args:
            index: Integer row index.

        Returns:
            A training tuple from build_causal_lm_example.
        """

        raise NotImplementedError("Seek to the row offset, tokenize text, and build an example")


class SFTDataset(Dataset):
    """JSONL chat dataset for instruction tuning.

    Args:
        data_path: JSONL file path. Each row should be a list of chat messages
            or contain a messages field.
        tokenizer: Tokenizer object with apply_chat_template and __call__.
        max_length: Final sequence length before shifting.
        messages_key: JSON key containing messages when rows are dictionaries.
        assistant_prefix: String prefix used to locate assistant spans.
        pad_token_id: Optional padding token ID. Defaults to tokenizer.pad_token_id.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        max_length: int,
        messages_key: str = "messages",
        assistant_prefix: str = "<|im_start|>assistant\n",
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.messages_key = messages_key
        self.assistant_prefix = assistant_prefix
        self.pad_token_id = pad_token_id
        raise NotImplementedError("Build byte offsets and assistant prefix IDs")

    def __len__(self) -> int:
        """Return the number of JSONL rows."""

        raise NotImplementedError("Return the precomputed row count")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one chat row and return shifted tensors with SFT loss mask.

        Args:
            index: Integer row index.

        Returns:
            A tuple (x, y, loss_mask), each with shape (max_length - 1,).
        """

        raise NotImplementedError("Render chat template, tokenize, mask assistant labels, and shift")
