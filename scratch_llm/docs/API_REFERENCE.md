# 函数参数速查

这份表只列你需要亲手实现的公开函数和方法。更完整的输入输出形状在各文件 docstring 里。

## 配置

| 位置 | 函数/类 | 关键参数 |
| --- | --- | --- |
| `scratch_llm/config.py` | `ModelConfig` | `vocab_size`, `dim`, `n_layers`, `n_heads`, `n_kv_heads`, `hidden_dim`, `max_seq_len`, `dropout` |
| `scratch_llm/config.py` | `TokenizerConfig` | `vocab_size`, `min_frequency`, `text_key`, `unk_token`, `bos_token`, `eos_token`, `pad_token` |
| `scratch_llm/config.py` | `TrainConfig` | `data_path`, `tokenizer_dir`, `output_dir`, `device`, `batch_size`, `learning_rate`, `epochs` |
| `scratch_llm/config.py` | `GenerationConfig` | `max_new_tokens`, `temperature`, `top_k`, `eos_token_id`, `pad_token_id` |

## 数据和 tokenizer

| 位置 | 函数/类 | 传入参数 | 返回 |
| --- | --- | --- | --- |
| `data/jsonl.py` | `iter_jsonl_records` | `path`, `encoding="utf-8"` | `Iterator[dict]` |
| `data/jsonl.py` | `iter_jsonl_texts` | `path`, `text_key="text"`, `encoding="utf-8"` | `Iterator[str]` |
| `data/jsonl.py` | `count_jsonl_lines` | `path` | `int` |
| `data/dataset.py` | `build_causal_lm_example` | `input_ids`, `max_length`, `pad_token_id` | `(x, y, loss_mask)` |
| `data/dataset.py` | `build_sft_loss_mask` | `input_ids`, `assistant_prefix_ids`, `eos_token_id` | `list[int]` |
| `data/dataset.py` | `CausalLMDataset` | `data_path`, `tokenizer`, `max_length`, `text_key`, `bos_token`, `pad_token_id` | `Dataset` |
| `data/dataset.py` | `SFTDataset` | `data_path`, `tokenizer`, `max_length`, `messages_key`, `assistant_prefix`, `pad_token_id` | `Dataset` |
| `tokenizer.py` | `train_bpe_tokenizer` | `data_path`, `output_dir`, `config` | `None` |
| `tokenizer.py` | `write_tokenizer_configs` | `output_dir`, `config`, `chat_template` | `None` |
| `tokenizer.py` | `load_tokenizer` | `tokenizer_dir` | tokenizer |
| `tokenizer.py` | `validate_tokenizer` | `tokenizer`, `sample_text`, `messages` | `dict` |

## 模型

| 位置 | 函数/类 | 传入参数 | 返回 |
| --- | --- | --- | --- |
| `model/norm.py` | `RMSNorm.__init__` | `dim`, `eps=1e-5` | module |
| `model/norm.py` | `RMSNorm.forward` | `x` | normalized tensor |
| `model/rope.py` | `precompute_rope_frequencies` | `head_dim`, `max_seq_len`, `theta`, `device` | `(cos, sin)` |
| `model/rope.py` | `reshape_for_broadcast` | `freqs`, `x` | reshaped tensor |
| `model/rope.py` | `apply_rotary_embedding` | `q`, `k`, `freqs_cos`, `freqs_sin` | `(q, k)` |
| `model/mlp.py` | `derive_swiglu_hidden_dim` | `dim`, `hidden_dim`, `multiple_of` | `int` |
| `model/mlp.py` | `SwiGLU.__init__` | `dim`, `hidden_dim`, `multiple_of`, `dropout` | module |
| `model/mlp.py` | `SwiGLU.forward` | `x` | tensor |
| `model/attention.py` | `repeat_kv` | `x`, `n_rep` | repeated tensor |
| `model/attention.py` | `CausalSelfAttention.__init__` | `config` | module |
| `model/attention.py` | `CausalSelfAttention.forward` | `x`, `freqs_cos`, `freqs_sin`, `attention_mask` | tensor |
| `model/blocks.py` | `DecoderBlock.__init__` | `layer_id`, `config` | module |
| `model/blocks.py` | `DecoderBlock.forward` | `x`, `freqs_cos`, `freqs_sin`, `attention_mask` | tensor |
| `model/transformer.py` | `ScratchLLM.__init__` | `config` | module |
| `model/transformer.py` | `ScratchLLM.init_weights` | `module` | `None` |
| `model/transformer.py` | `ScratchLLM.prepare_attention_mask` | `attention_mask`, `input_ids` | mask or `None` |
| `model/transformer.py` | `ScratchLLM.forward` | `input_ids`, `labels`, `attention_mask` | `dict` |

## 训练和推理

| 位置 | 函数 | 传入参数 | 返回 |
| --- | --- | --- | --- |
| `training/lr.py` | `cosine_lr` | `step`, `total_steps`, `base_lr`, `warmup_steps`, `min_lr_ratio` | `float` |
| `training/loop.py` | `move_batch_to_device` | `batch`, `device` | `batch` |
| `training/loop.py` | `compute_masked_loss` | `per_token_loss`, `loss_mask` | scalar tensor |
| `training/loop.py` | `train_one_epoch` | `model`, `dataloader`, `optimizer`, `config`, `epoch`, `total_steps`, `start_step`, `scaler`, `logger` | global step |
| `training/loop.py` | `evaluate` | `model`, `dataloader`, `config`, `max_batches` | metrics dict |
| `training/checkpoint.py` | `unwrap_model` | `model` | module |
| `training/checkpoint.py` | `save_checkpoint` | `path`, `model`, `optimizer`, `step`, `extra` | `None` |
| `training/checkpoint.py` | `load_checkpoint` | `path`, `model`, `optimizer`, `map_location`, `strict` | metadata dict |
| `inference/generation.py` | `top_k_filter` | `logits`, `top_k` | logits |
| `inference/generation.py` | `sample_next_token` | `logits`, `temperature`, `top_k` | token IDs |
| `inference/generation.py` | `generate` | `model`, `input_ids`, `config`, `attention_mask` | generated IDs |
| `utils/seed.py` | `seed_everything` | `seed`, `deterministic` | `None` |
| `utils/params.py` | `count_parameters` | `model`, `trainable_only` | `int` |
