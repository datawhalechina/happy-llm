# OPD 配套代码

> **代码来源：** 本目录代码引用并整理自本章作者维护的 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 中的 [02-opd](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/02-opd) 实现，当前版本适配 Happy-LLM 第八章与 PyTRIO 0.2.6。

本目录对应正文 8.2 节，以 DeepMath-103K prompt 为例实现完整的 On-Policy Distillation。Student 生成回答，Teacher 对同一条 Student 轨迹计算逐 token logprob，再由 reverse KL 构造训练信号。

## 文件说明

| 文件 | 作用 | 正文定位 |
| --- | --- | --- |
| `01-demo-sync.py` | 顺序执行 Student rollout、Teacher 打分和 Student 更新 | 逐段讲解 |
| `02-demo-async.py` | 并发执行 batch 内 rollout 与 Teacher logprob 请求 | 提供完整代码，只说明接口差异 |

## 运行前准备

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

需要在线记录实验时，再执行 `swanlab login`。

## 运行同步版

```bash
python docs/chapter8/opd/01-demo-sync.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 1 \
    --max-tokens 512 \
    --sample-size 20 \
    --num-shards 1 \
    --swanlab-mode disabled
```

脚本会从 ModelScope 下载 DeepMath-103K parquet。`--num-shards 1` 适合小成本试跑，`--num-shards 10` 会使用全部分片。

## 运行异步版

```bash
python docs/chapter8/opd/02-demo-async.py \
    --steps 1 \
    --batch-size 4 \
    --group-size 2 \
    --max-tokens 512 \
    --sample-size 20 \
    --num-shards 1 \
    --swanlab-mode disabled
```

异步版在 batch 内并发执行 Student rollout，并在同一 prompt 内并发提交 Teacher logprob 请求。token 对齐、reverse KL 与 `importance_sampling` Datum 保持不变。

## 同步版代码阅读顺序

1. `parse_args()` 与 `load_deepmath()`：准备训练参数和 prompt-only 数据。
2. `build_prompt()`：把问题渲染成 Student 输入。
3. `completion_teacher_logprobs()`：让 Teacher 对 Student completion 打分。
4. `build_opd_datum()`：把逐 token reverse KL 写入 advantage。
5. `main()`：刷新 Student sampler，更新 Student，记录指标并保存权重。

Teacher 默认使用 `Qwen/Qwen3.6-27B`。实际可用模型以 PyTRIO 服务返回结果为准，也可以通过 `--teacher-base-model` 或 `--teacher-model-path` 指定其他 Teacher。Teacher 与 Student 的 token id 必须兼容。
