# 实现顺序建议

## 第 1 步: 数据读写

先完成 `iter_jsonl_records`, `iter_jsonl_texts`, `count_jsonl_lines`。这一步只用标准库 `json` 和文件读写，不碰张量。

验收点：

```bash
python3 -m unittest tests/test_contracts.py
```

## 第 2 步: Tokenizer

完成 `train_bpe_tokenizer`, `write_tokenizer_configs`, `load_tokenizer`, `validate_tokenizer`。参考 `docs/chapter5/code/train_tokenizer.py`，但自己写一遍流程。

最小数据格式：

```jsonl
{"text": "hello world"}
{"text": "language model learns next tokens"}
```

## 第 3 步: Dataset

完成 `build_causal_lm_example` 后再写 `CausalLMDataset`。

关键形状：

```text
unshifted input_ids: max_length
x: max_length - 1
y: max_length - 1
loss_mask: max_length - 1
```

## 第 4 步: 模型小模块

建议顺序：

1. `RMSNorm`
2. `precompute_rope_frequencies`
3. `reshape_for_broadcast`
4. `apply_rotary_embedding`
5. `derive_swiglu_hidden_dim`
6. `SwiGLU`
7. `repeat_kv`

这些都可以单独造随机张量测试，不需要训练。

## 第 5 步: Attention 和 Block

先实现不带 padding mask 的 causal attention，再补 `attention_mask`。

注意形状变换：

```text
x:  (batch, seq_len, dim)
q:  (batch, seq_len, n_heads, head_dim)
k/v:(batch, seq_len, n_kv_heads, head_dim)
attn input after transpose: (batch, heads, seq_len, head_dim)
output: (batch, seq_len, dim)
```

## 第 6 步: ScratchLLM

把 embedding、decoder blocks、final RMSNorm、LM head 串起来。训练时返回全量 logits 和 loss；生成时先不用 KV cache，保持简单。

## 第 7 步: 训练 loop

先让 CPU 上的小模型跑通，再考虑 GPU、混合精度、多卡。

建议最小配置：

```bash
python3 scripts/02_pretrain.py \
  --data-path data/pretrain.jsonl \
  --device cpu \
  --dim 128 \
  --n-layers 2 \
  --n-heads 4 \
  --batch-size 2 \
  --max-seq-len 64
```

## 第 8 步: 生成

先实现 `temperature=0` 的 greedy decode，再实现 `top_k` 和随机采样。
