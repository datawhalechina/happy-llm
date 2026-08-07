# GRPO 配套代码

> **代码来源：** 本目录代码引用并整理自本章作者维护的 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 中的 [01-grpo](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/01-grpo) 实现，当前版本适配 Happy-LLM 第八章与 PyTRIO 0.2.6。

本目录对应正文 8.1 节，以 GSM8K 为例实现一条完整的 PyTRIO GRPO 训练链路。代码包含同题分组采样、规则奖励、组内相对优势、`Datum` 对齐、策略更新和 SwanLab 指标记录。

## 运行前准备

从 Happy-LLM 仓库根目录创建 Python 3.13 环境并安装第八章公共依赖：

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

需要在线记录实验时，再执行 `swanlab login`。

## 最小试跑

```bash
python docs/chapter8/grpo/train.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

训练第一次启动时会下载 GSM8K。正式实验应增大 `steps`、`batch-size` 和 `group-size`，并固定其余配置后再比较不同 loss。

## 代码阅读顺序

1. `grade_answer()`：抽取 `\boxed{}` 并计算规则奖励。
2. `run_rollout_group()`：对同一道题采样一组回答并计算相对优势。
3. `build_grpo_datum()`：完成自回归右移和 prompt mask。
4. `main()`：串联 rollout、`forward_backward()` 与 `optim_step()`。
