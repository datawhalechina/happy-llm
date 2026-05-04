"""Interface contract tests for the scratch LLM scaffold.

These tests intentionally do not call unfinished implementations. They protect
function names and parameter order while you fill in the TODOs.
"""

from __future__ import annotations

import inspect
import unittest

from scratch_llm.config import GenerationConfig, ModelConfig, TokenizerConfig, TrainConfig
from scratch_llm.data.dataset import build_causal_lm_example, build_sft_loss_mask
from scratch_llm.data.jsonl import count_jsonl_lines, iter_jsonl_records, iter_jsonl_texts
from scratch_llm.inference.generation import generate, sample_next_token, top_k_filter
from scratch_llm.model.attention import CausalSelfAttention, repeat_kv
from scratch_llm.model.blocks import DecoderBlock
from scratch_llm.model.mlp import SwiGLU, derive_swiglu_hidden_dim
from scratch_llm.model.norm import RMSNorm
from scratch_llm.model.rope import (
    apply_rotary_embedding,
    precompute_rope_frequencies,
    reshape_for_broadcast,
)
from scratch_llm.model.transformer import ScratchLLM
from scratch_llm.tokenizer import (
    load_tokenizer,
    train_bpe_tokenizer,
    validate_tokenizer,
    write_tokenizer_configs,
)
from scratch_llm.training.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from scratch_llm.training.loop import compute_masked_loss, evaluate, move_batch_to_device, train_one_epoch
from scratch_llm.training.lr import cosine_lr
from scratch_llm.utils.params import count_parameters
from scratch_llm.utils.seed import seed_everything


def parameter_names(fn: object) -> list[str]:
    """Return callable parameter names in declaration order."""

    return list(inspect.signature(fn).parameters)


class ConfigContractTest(unittest.TestCase):
    def test_default_model_config_is_shape_valid(self) -> None:
        config = ModelConfig()
        config.validate()
        self.assertEqual(config.head_dim, config.dim // config.n_heads)
        self.assertEqual(config.effective_n_kv_heads, config.n_heads)

    def test_other_config_objects_construct(self) -> None:
        self.assertEqual(TokenizerConfig().text_key, "text")
        self.assertEqual(TrainConfig().device, "cpu")
        self.assertEqual(GenerationConfig().max_new_tokens, 128)


class FunctionSignatureContractTest(unittest.TestCase):
    def assert_params(self, fn: object, expected: list[str]) -> None:
        self.assertEqual(parameter_names(fn), expected)

    def test_data_signatures(self) -> None:
        self.assert_params(iter_jsonl_records, ["path", "encoding"])
        self.assert_params(iter_jsonl_texts, ["path", "text_key", "encoding"])
        self.assert_params(count_jsonl_lines, ["path"])
        self.assert_params(build_causal_lm_example, ["input_ids", "max_length", "pad_token_id"])
        self.assert_params(
            build_sft_loss_mask,
            ["input_ids", "assistant_prefix_ids", "eos_token_id"],
        )

    def test_tokenizer_signatures(self) -> None:
        self.assert_params(train_bpe_tokenizer, ["data_path", "output_dir", "config"])
        self.assert_params(write_tokenizer_configs, ["output_dir", "config", "chat_template"])
        self.assert_params(load_tokenizer, ["tokenizer_dir"])
        self.assert_params(validate_tokenizer, ["tokenizer", "sample_text", "messages"])

    def test_model_signatures(self) -> None:
        self.assert_params(precompute_rope_frequencies, ["head_dim", "max_seq_len", "theta", "device"])
        self.assert_params(reshape_for_broadcast, ["freqs", "x"])
        self.assert_params(apply_rotary_embedding, ["q", "k", "freqs_cos", "freqs_sin"])
        self.assert_params(RMSNorm.__init__, ["self", "dim", "eps"])
        self.assert_params(RMSNorm.forward, ["self", "x"])
        self.assert_params(derive_swiglu_hidden_dim, ["dim", "hidden_dim", "multiple_of"])
        self.assert_params(SwiGLU.__init__, ["self", "dim", "hidden_dim", "multiple_of", "dropout"])
        self.assert_params(SwiGLU.forward, ["self", "x"])
        self.assert_params(repeat_kv, ["x", "n_rep"])
        self.assert_params(CausalSelfAttention.__init__, ["self", "config"])
        self.assert_params(
            CausalSelfAttention.forward,
            ["self", "x", "freqs_cos", "freqs_sin", "attention_mask"],
        )
        self.assert_params(DecoderBlock.__init__, ["self", "layer_id", "config"])
        self.assert_params(
            DecoderBlock.forward,
            ["self", "x", "freqs_cos", "freqs_sin", "attention_mask"],
        )
        self.assert_params(ScratchLLM.__init__, ["self", "config"])
        self.assert_params(ScratchLLM.init_weights, ["self", "module"])
        self.assert_params(ScratchLLM.prepare_attention_mask, ["self", "attention_mask", "input_ids"])
        self.assert_params(ScratchLLM.forward, ["self", "input_ids", "labels", "attention_mask"])

    def test_training_and_inference_signatures(self) -> None:
        self.assert_params(cosine_lr, ["step", "total_steps", "base_lr", "warmup_steps", "min_lr_ratio"])
        self.assert_params(move_batch_to_device, ["batch", "device"])
        self.assert_params(compute_masked_loss, ["per_token_loss", "loss_mask"])
        self.assert_params(
            train_one_epoch,
            [
                "model",
                "dataloader",
                "optimizer",
                "config",
                "epoch",
                "total_steps",
                "start_step",
                "scaler",
                "logger",
            ],
        )
        self.assert_params(evaluate, ["model", "dataloader", "config", "max_batches"])
        self.assert_params(unwrap_model, ["model"])
        self.assert_params(save_checkpoint, ["path", "model", "optimizer", "step", "extra"])
        self.assert_params(
            load_checkpoint,
            ["path", "model", "optimizer", "map_location", "strict"],
        )
        self.assert_params(top_k_filter, ["logits", "top_k"])
        self.assert_params(sample_next_token, ["logits", "temperature", "top_k"])
        self.assert_params(generate, ["model", "input_ids", "config", "attention_mask"])
        self.assert_params(seed_everything, ["seed", "deterministic"])
        self.assert_params(count_parameters, ["model", "trainable_only"])


if __name__ == "__main__":
    unittest.main()
