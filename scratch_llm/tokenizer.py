"""Tokenizer training and validation contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from scratch_llm.config import TokenizerConfig


DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def special_tokens(config: TokenizerConfig) -> list[str]:
    """Return special tokens in deterministic ID order.

    Args:
        config: TokenizerConfig containing token strings.

    Returns:
        List ordered as unk, optional legacy BOS/EOS, chat BOS, chat EOS.
    """

    return [config.unk_token, "<s>", "</s>", config.bos_token, config.eos_token]


def train_bpe_tokenizer(
    data_path: str | Path,
    output_dir: str | Path,
    config: TokenizerConfig,
) -> None:
    """Train and save a ByteLevel BPE tokenizer.

    Args:
        data_path: JSONL file with raw text.
        output_dir: Directory where tokenizer.json and configs are written.
        config: Tokenizer training config.

    TODO:
        Use tokenizers.Tokenizer(models.BPE), NFKC normalization, ByteLevel
        pre-tokenization/decoding, and BpeTrainer.
    """

    raise NotImplementedError("Implement BPE tokenizer training")


def write_tokenizer_configs(
    output_dir: str | Path,
    config: TokenizerConfig,
    chat_template: str = DEFAULT_CHAT_TEMPLATE,
) -> None:
    """Write tokenizer_config.json and special_tokens_map.json.

    Args:
        output_dir: Directory that already contains tokenizer.json.
        config: TokenizerConfig containing token strings.
        chat_template: Jinja chat template for transformers tokenizers.
    """

    raise NotImplementedError("Implement tokenizer sidecar config writing")


def load_tokenizer(tokenizer_dir: str | Path) -> Any:
    """Load a saved tokenizer through transformers.AutoTokenizer.

    Args:
        tokenizer_dir: Directory containing tokenizer.json and config files.

    Returns:
        A tokenizer object compatible with the dataset and scripts.
    """

    raise NotImplementedError("Import AutoTokenizer and load from tokenizer_dir")


def validate_tokenizer(
    tokenizer: Any,
    sample_text: str,
    messages: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """Run simple encode/decode checks.

    Args:
        tokenizer: Tokenizer returned by load_tokenizer.
        sample_text: Text used for round-trip encoding checks.
        messages: Optional chat messages for chat_template checks.

    Returns:
        Dictionary with vocab size, special tokens, encoded IDs, and decoded text.
    """

    raise NotImplementedError("Implement tokenizer smoke checks")
