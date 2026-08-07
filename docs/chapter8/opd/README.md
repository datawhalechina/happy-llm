# OPD 配套代码

> **代码来源：** 本目录代码引用并整理自本章作者维护的 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 中的 [02-opd](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/02-opd) 实现，当前版本适配 Happy-LLM 第八章与 PyTRIO 0.2.6。

本目录对应正文 8.2 节，以 DeepMath-103K 的 prompt 为例实现同步版 On-Policy Distillation。Student 先生成回答，Teacher 随后对同一条 Student 轨迹计算逐 token logprob，最终使用 reverse KL 构造稠密训练信号。

## 运行前准备

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

需要在线记录实验时，再执行 `swanlab login`。

## 最小试跑

```bash
python docs/chapter8/opd/train.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 1 \
    --max-tokens 512 \
    --sample-size 20 \
    --num-shards 1 \
    --swanlab-mode disabled
```

Teacher 默认使用 `Qwen/Qwen3.6-27B`。实际可用模型以 `ServiceClient.get_supported_models()` 返回结果为准，也可以通过 `--teacher-base-model` 或 `--teacher-model-path` 指定其他 Teacher。

## 代码阅读顺序

1. `build_prompt()`：把 prompt-only 数据渲染成模型输入。
2. `completion_teacher_logprobs()`：让 Teacher 对 Student completion 打分。
3. `build_opd_datum()`：把逐 token reverse KL 写入 advantage。
4. `main()`：保证采样器按步刷新并更新 Student。

Teacher 与 Student 需要使用可对齐的 tokenizer。代码不会使用 Teacher 自己生成的回答替代 Student rollout。
