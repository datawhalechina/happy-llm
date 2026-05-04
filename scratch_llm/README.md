# Scratch LLM 学习工程

这个目录是给你边读 Happy-LLM 边手搓 LLM 用的骨架。原教程代码保留在 `docs/chapter5/code/`，这里新增的 `scratch_llm/` 不复制完整答案，只给出架构、函数签名、参数、返回值和 TODO。

## 目标

1. 从 JSONL 文本训练一个 ByteLevel BPE tokenizer。
2. 构造 causal language modeling 数据集。
3. 手写一个 LLaMA 风格 decoder-only Transformer。
4. 完成预训练 loop、checkpoint、生成函数。
5. 用 Git 分支把本地进度同步到 GitHub。

## 目录

```text
scratch_llm/
  config.py              # ModelConfig / TokenizerConfig / TrainConfig / GenerationConfig
  tokenizer.py           # BPE tokenizer 训练、保存、加载、校验
  data/
    jsonl.py             # JSONL 读取
    dataset.py           # Pretrain/SFT 数据集和 loss mask
  model/
    norm.py              # RMSNorm
    rope.py              # RoPE
    attention.py         # causal self-attention / GQA
    mlp.py               # SwiGLU
    blocks.py            # DecoderBlock
    transformer.py       # ScratchLLM
  training/
    lr.py                # warmup + cosine decay
    loop.py              # train/evaluate
    checkpoint.py        # save/load
  inference/
    generation.py        # top-k sampling / generate
  utils/
    seed.py              # 随机种子
    params.py            # 参数量统计
scripts/
  01_train_tokenizer.py
  02_pretrain.py
  03_generate.py
tests/
  test_contracts.py
```

## 推荐实现顺序

先写纯 Python 小函数，再写张量函数，最后写训练。

1. `scratch_llm/data/jsonl.py`
2. `scratch_llm/tokenizer.py`
3. `scratch_llm/data/dataset.py`
4. `scratch_llm/model/norm.py`
5. `scratch_llm/model/rope.py`
6. `scratch_llm/model/mlp.py`
7. `scratch_llm/model/attention.py`
8. `scratch_llm/model/blocks.py`
9. `scratch_llm/model/transformer.py`
10. `scratch_llm/training/lr.py`
11. `scratch_llm/training/loop.py`
12. `scratch_llm/training/checkpoint.py`
13. `scratch_llm/inference/generation.py`

## 本地运行

接口契约测试不依赖 pytest：

```bash
python3 -m unittest tests/test_contracts.py
```

语法检查：

```bash
python3 -m compileall scratch_llm scripts tests
```

当你开始填实现后，可以逐步跑：

```bash
python3 scripts/01_train_tokenizer.py --data-path data/pretrain.jsonl
python3 scripts/02_pretrain.py --data-path data/pretrain.jsonl --device cpu
python3 scripts/03_generate.py --checkpoint scratch_llm_runs/checkpoints/model.pt --prompt "你好"
```

## GitHub 同步

当前骨架建议放在独立分支：

```bash
git status --short --branch
git add scratch_llm scripts tests pyproject.toml .gitignore
git commit -m "Add scratch LLM learning scaffold"
git push -u origin scratch-llm-starter
```

之后你的节奏可以是：实现一个函数，跑一次测试，提交一次小 commit。
