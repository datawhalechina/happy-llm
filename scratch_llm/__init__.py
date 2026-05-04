"""Learning-first LLM scaffold.

This package is intentionally incomplete: it defines the architecture,
function boundaries, and parameters you need to implement a small decoder-only
LLM from scratch while reading Happy-LLM chapter 5.
"""

from scratch_llm.config import GenerationConfig, ModelConfig, TokenizerConfig, TrainConfig
from scratch_llm.model.transformer import ScratchLLM

__all__ = [
    "GenerationConfig",
    "ModelConfig",
    "ScratchLLM",
    "TokenizerConfig",
    "TrainConfig",
]
