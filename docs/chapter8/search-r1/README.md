# Search-R1 配套代码

> **代码来源：** 本目录代码引用并整理自本章作者维护的 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 中的 [03-search-r1](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/03-search-r1) 实现，当前版本适配 Happy-LLM 第八章与 PyTRIO 0.2.6。

本目录对应正文 8.3 节。代码保留数据、工具协议、搜索环境、多轮 rollout、reward、训练和评测的边界，便于观察 Agentic RL 如何在普通 GRPO 训练循环上增加环境交互。

## 运行前准备

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

需要在线记录实验时，再执行 `swanlab login`。

## 准备数据

```bash
python docs/chapter8/search-r1/prepare_data.py
```

该命令会准备 NQ 与 HotpotQA 训练、开发和测试数据。Wikipedia 后端免费且不需要 API Key，适合先验证流程。

## 最小试跑

```bash
python docs/chapter8/search-r1/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --group-size 4 \
    --max-search-calls 2 \
    --search-backend wikipedia \
    --search-concurrency 3 \
    --swanlab-mode disabled
```

在线搜索结果会随时间变化。比较 Base Model 与 checkpoint 时，必须保持题集、搜索后端、最大搜索次数和 sampling 参数一致。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `protocol.py` | 定义工具协议，解析搜索动作和最终答案 |
| `search.py` | 封装 DeepSeek Search、Wikipedia 和知乎搜索后端 |
| `rollout.py` | 执行多轮“生成—搜索—观察—继续生成”状态机 |
| `reward.py` | 计算格式奖励和答案精确匹配奖励 |
| `train.py` | 构造 observation mask、PyTRIO Datum 并更新策略 |
| `eval.py` | 使用相同环境评测 Base Model 或 checkpoint |
| `analyse.py` | 汇总评测结果 |
