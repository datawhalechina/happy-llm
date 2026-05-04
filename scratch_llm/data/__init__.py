"""Dataset and JSONL helpers."""

from scratch_llm.data.dataset import (
    CausalLMDataset,
    SFTDataset,
    build_causal_lm_example,
    build_sft_loss_mask,
)
from scratch_llm.data.jsonl import count_jsonl_lines, iter_jsonl_records, iter_jsonl_texts

__all__ = [
    "CausalLMDataset",
    "SFTDataset",
    "build_causal_lm_example",
    "build_sft_loss_mask",
    "count_jsonl_lines",
    "iter_jsonl_records",
    "iter_jsonl_texts",
]
