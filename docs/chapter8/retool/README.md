# ReTool 配套代码

> **代码来源：** 本目录代码引用并整理自本章作者维护的 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 中的 [05-retool](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/05-retool) 实现，当前版本适配 Happy-LLM 第八章与 PyTRIO 0.2.6。

本目录对应正文 8.4 节。代码实现数学题上的多轮代码解释器 rollout、结果奖励、observation mask、PPO 更新和统一评测。

> 安全提示：`sandbox.py` 通过独立 subprocess、超时和资源限制控制意外消耗，但不能提供可信安全隔离。模型生成的代码仍可能访问本机文件、网络和继承的环境变量。处理不可信代码时，请使用一次性容器、低权限虚拟机或专用沙箱服务，并移除所有凭证。

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
python docs/chapter8/retool/prepare_data.py
```

## 最小试跑

请先在隔离环境中确认执行器权限，再运行：

```bash
python docs/chapter8/retool/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --group-size 4 \
    --max-code-calls 2 \
    --sandbox-workers 2 \
    --swanlab-mode disabled
```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `protocol.py` | 定义 `code_interpreter` 工具和消息拼接规则 |
| `sandbox.py` | 执行 Python 代码并限制时间、进程和输出大小 |
| `rollout.py` | 执行多轮“生成—运行代码—观察—继续生成”状态机 |
| `reward.py` | 抽取最后一个 `\boxed{}` 并判断数学等价性 |
| `train.py` | 构造 observation mask、PyTRIO Datum 并执行 PPO 更新 |
| `eval.py` | 统一评测 text-only 与 ReTool 模式 |
| `analysis.py` | 汇总不同 checkpoint 的指标 |
