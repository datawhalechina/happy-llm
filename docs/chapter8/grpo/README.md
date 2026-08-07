# GRPO 配套代码

> **代码来源：** 本目录代码引用并整理自本章作者维护的 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 中的 [01-grpo](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/01-grpo) 实现，当前版本适配 Happy-LLM 第八章与 PyTRIO 0.2.6。

本目录对应正文 8.1 节，以 GSM8K 为例实现完整的 PyTRIO GRPO 训练链路。同步版与异步版使用相同的 prompt、规则奖励、组内相对优势、Datum 对齐和 loss。

## 文件说明

| 文件 | 作用 | 正文定位 |
| --- | --- | --- |
| `01-demo-sync.py` | 按 prompt 顺序完成 rollout 和训练 | 逐段讲解 |
| `02-demo-async.py` | 使用 `asyncio.gather()` 并发执行 batch 内 rollout | 提供完整代码，只说明接口差异 |

## 运行前准备

从 Happy-LLM 仓库根目录创建 Python 3.13 环境并安装第八章公共依赖：

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

需要在线记录实验时，再执行 `swanlab login`。

## 运行同步版

```bash
python docs/chapter8/grpo/01-demo-sync.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

训练第一次启动时会下载 GSM8K。正式实验应增大 `steps`、`batch-size` 和 `group-size`，并固定其余配置后再比较 `importance_sampling` 与 `ppo`。

## 运行异步版

```bash
python docs/chapter8/grpo/02-demo-async.py \
    --steps 1 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

异步版会并发处理同一 batch 中不同 prompt 的 rollout。reward、advantage、Datum 和 loss 与同步版保持一致。

## 同步版代码阅读顺序

1. `parse_args()` 与 `RolloutSample`：定义训练配置和一条 rollout 的数据。
2. `grade_answer()`：抽取 `\boxed{}` 并计算规则奖励。
3. `run_rollout_group()`：对同一道题采样一组回答并计算相对优势。
4. `build_grpo_datum()`：完成自回归右移和 prompt mask。
5. `main()`：选择内置 loss，并串联 sampler 刷新、rollout、策略更新、日志和权重保存。

异步版在对应函数中使用 `sample_async()`、`forward_backward_async()`、`optim_step_async()` 和 `asyncio.gather()`。
