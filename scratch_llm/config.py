"""Project-wide configuration objects.

These dataclasses are small on purpose. Start by changing values here before
touching model code, so every script reads the same source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class ModelConfig:
    """Decoder-only Transformer model parameters.

    Args:
        vocab_size: Number of tokenizer vocabulary entries.
        dim: Hidden size of the residual stream.
        n_layers: Number of decoder blocks.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads. Use None for standard MHA.
        hidden_dim: MLP hidden size. Use None to derive a SwiGLU size.
        multiple_of: Round the derived MLP size to a multiple of this value.
        norm_eps: Epsilon used by RMSNorm.
        max_seq_len: Maximum context length supported by RoPE buffers.
        dropout: Dropout probability used in attention, MLP, and embeddings.
        rope_theta: Base frequency for rotary position embeddings.
        pad_token_id: Token ID used for padding and ignored labels.
        bos_token_id: Beginning-of-sequence token ID.
        eos_token_id: End-of-sequence token ID.
        tie_embeddings: Whether token embedding and LM head share weights.
        use_flash_attention: Whether to prefer scaled_dot_product_attention.
    """

    vocab_size: int = 6144
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    hidden_dim: Optional[int] = None
    multiple_of: int = 64
    norm_eps: float = 1e-5
    max_seq_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    pad_token_id: int = 0
    bos_token_id: int = 3
    eos_token_id: int = 4
    tie_embeddings: bool = True
    use_flash_attention: bool = True

    @property
    def head_dim(self) -> int:
        """Return the per-head hidden size."""

        return self.dim // self.n_heads

    @property
    def effective_n_kv_heads(self) -> int:
        """Return key/value head count after filling the MHA default."""

        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    def validate(self) -> None:
        """Validate shape constraints before creating modules.

        Raises:
            ValueError: If a configuration would make attention shapes invalid.
        """

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.dim <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("dim, n_layers, and n_heads must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.effective_n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive when provided")
        if self.n_heads % self.effective_n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(slots=True)
class TokenizerConfig:
    """BPE tokenizer training parameters.

    Args:
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token pair frequency for BPE merges.
        text_key: JSONL key containing training text.
        unk_token: Unknown token string.
        bos_token: Beginning-of-sequence token string.
        eos_token: End-of-sequence token string.
        pad_token: Padding token string.
        user_role: Chat template user role name.
        assistant_role: Chat template assistant role name.
        system_role: Chat template system role name.
    """

    vocab_size: int = 6144
    min_frequency: int = 2
    text_key: str = "text"
    unk_token: str = "<unk>"
    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"
    user_role: str = "user"
    assistant_role: str = "assistant"
    system_role: str = "system"


@dataclass(slots=True)
class TrainConfig:
    """Training loop parameters.

    Args:
        data_path: JSONL dataset path.
        tokenizer_dir: Directory containing tokenizer files.
        output_dir: Directory for checkpoints and logs.
        device: Torch device string such as cpu, cuda, or cuda:0.
        dtype: Precision name: float32, float16, or bfloat16.
        batch_size: Number of samples per optimizer micro-step.
        max_seq_len: Number of input tokens after shifting.
        learning_rate: AdamW base learning rate.
        weight_decay: AdamW weight decay.
        beta1: AdamW first momentum coefficient.
        beta2: AdamW second momentum coefficient.
        epochs: Number of full passes over the dataset.
        grad_clip: Max gradient norm.
        accumulation_steps: Number of micro-steps per optimizer update.
        warmup_steps: Number of linear warmup steps.
        log_interval: Step interval for logging.
        save_interval: Step interval for saving checkpoints.
        num_workers: DataLoader worker count.
        seed: Random seed.
    """

    data_path: str = "data/pretrain.jsonl"
    tokenizer_dir: str = "scratch_llm_runs/tokenizer"
    output_dir: str = "scratch_llm_runs/checkpoints"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 8
    max_seq_len: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epochs: int = 1
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    warmup_steps: int = 100
    log_interval: int = 10
    save_interval: int = 1000
    num_workers: int = 0
    seed: int = 42


@dataclass(slots=True)
class GenerationConfig:
    """Autoregressive decoding parameters.

    Args:
        max_new_tokens: Maximum number of tokens to append.
        temperature: 0 means greedy decoding; >0 means sampling.
        top_k: Keep only the highest-k logits before sampling. None disables it.
        eos_token_id: Stop token ID. None disables early stopping.
        pad_token_id: Token ID used after a sequence has finished.
    """

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = None
    eos_token_id: Optional[int] = 4
    pad_token_id: Optional[int] = 0
