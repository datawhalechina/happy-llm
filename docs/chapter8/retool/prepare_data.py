"""下载并整理 ReTool 训练数据（DAPO-Math-17k）。

跟随官方 recipe 使用 BytedTsinghua-SIA/DAPO-Math-17k：
- question 取 prompt[0].content
- answer 取 reward_model.ground_truth
按固定 seed 打乱后切出 50 条 dev.jsonl（用于快速验证，不参与训练），
其余写入 train.jsonl。

运行：
python docs/chapter8/retool/prepare_data.py
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"

# DAPO-Math-17k 的题目外面统一包了一层模板，要求用 "Answer: $Answer" 格式作答，
# 与我们 system prompt / reward 要求的 \boxed{} 格式直接冲突，清洗时剥掉。
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def strip_dapo_template(question: str) -> str:
    """剥掉 DAPO 统一外层模板，只保留题目本体；模板不完整时原样保留。"""
    if question.startswith(PROMPT_PREFIX):
        question = question[len(PROMPT_PREFIX) :]
    if question.endswith(PROMPT_SUFFIX):
        question = question[: -len(PROMPT_SUFFIX)]
    return question.strip()


def normalize_row(index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    """把 verl 格式的原始样本转成简洁的训练格式。"""
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return None
    question = strip_dapo_template(str(prompt[0].get("content") or ""))
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else None
    answer = str(ground_truth or "").strip()
    if not question or not answer:
        return None
    return {
        "id": str(row.get("extra_info", {}).get("index") or index),
        "question": question,
        "answer": answer,
        "data_source": str(row.get("data_source") or "dapo_math"),
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """把处理后的样本写入 JSONL 文件。"""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """解析数据准备命令行参数。"""
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=base_dir / "datasets" / "raw" / "dapo-math-17k.parquet",
        help="已下载的原始 parquet；不存在时回退到从 Hub 加载",
    )
    parser.add_argument("--output-dir", type=Path, default=base_dir / "datasets")
    parser.add_argument("--dev-size", type=int, default=50, help="切出多少条作为 dev 集")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """下载 DAPO-Math-17k 并生成 train.jsonl 与 dev.jsonl。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw.exists():
        # 网络不稳定时先用 curl 下载到 datasets/raw/，再从本地 parquet 加载。
        dataset = load_dataset("parquet", data_files=str(args.raw), split="train")
    else:
        dataset = load_dataset(DATASET_ID, split="train")
    records = [
        record
        for index, row in enumerate(dataset)
        if (record := normalize_row(index, row)) is not None
    ]
    if not records:
        raise ValueError("数据清洗后为空，请检查 DAPO-Math-17k 的字段结构")

    random.Random(args.seed).shuffle(records)
    dev_records = records[: args.dev_size]
    train_records = records[args.dev_size :]
    write_jsonl(train_records, args.output_dir / "train.jsonl")
    write_jsonl(dev_records, args.output_dir / "dev.jsonl")

    lengths = sorted(len(record["answer"]) for record in records)
    print(f"train: total={len(train_records)}")
    print(f"dev: total={len(dev_records)}")
    print(
        "answer length: "
        f"min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}"
    )
    print("sample:", json.dumps(train_records[0], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
