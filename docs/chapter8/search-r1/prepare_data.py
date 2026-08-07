"""从 ModelScope 下载并整理 Search-R1 数据集。

运行：
python docs/chapter8/search-r1/prepare_data.py
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from modelscope import dataset_snapshot_download


DATASET_ID = "zhuangzhuang2023/nq_hotpotqa_train"
DATASET_REVISION = "aa2da0496c1b1a50a66af7acabdf09c07a0cb79e"
RAW_COLUMNS = ["id", "question", "golden_answers", "data_source", "reward_model"]


def download_dataset(raw_dir: Path) -> Path:
    """从 ModelScope 下载固定版本的两个 Parquet 文件。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_snapshot_download(
        DATASET_ID,
        revision=DATASET_REVISION,
        local_dir=str(raw_dir),
        allow_patterns=["train.parquet", "test.parquet"],
    )
    return Path(path)


def clean_answers(values: Any) -> list[str]:
    """清理答案列表并保持原有顺序。"""
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    answers: list[str] = []
    for value in values:
        answer = str(value).strip()
        if answer and answer not in answers:
            answers.append(answer)
    return answers


def extract_answers(row: dict[str, Any]) -> list[str]:
    """优先读取 golden_answers，并兼容 reward_model 字段。"""
    answers = clean_answers(row.get("golden_answers"))
    if answers:
        return answers
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target")
    return clean_answers(ground_truth)


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """把原始样本转成训练和评测共用的简洁格式。"""
    question = str(row.get("question") or "").strip()
    answers = extract_answers(row)
    if not question or not answers:
        return None
    return {
        "id": str(row.get("id") or ""),
        "question": question,
        "answers": answers,
        "data_source": str(row.get("data_source") or "unknown"),
    }


def prepare_split(
    parquet_path: Path,
    output_path: Path,
    collect_records: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """逐条清洗一个 Parquet split 并写成 JSONL。"""
    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
        columns=RAW_COLUMNS,
    )
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as file:
        for row in dataset:
            record = normalize_row(row)
            if record is None:
                continue
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if collect_records:
                records.append(record)
            counts[record["data_source"]] += 1
    return records, counts

# 按来源等量抽样；默认从 7 个 benchmark 各抽 10 条，共 70 条
def select_dev(records: list[dict[str, Any]], per_source: int, seed: int) -> list[dict[str, Any]]:
    """从每个测试来源固定抽取一小份开发集。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["data_source"]].append(record)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for source in sorted(groups):
        rng.shuffle(groups[source])
        selected.extend(groups[source][:per_source])
    return selected


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """把处理后的样本写入 JSONL 文件。"""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_counts(name: str, counts: Counter[str]) -> None:
    """打印一个 split 的样本总量和来源分布。"""
    details = ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))
    print(f"{name}: total={sum(counts.values())}; {details}")


def parse_args() -> argparse.Namespace:
    """解析数据准备命令行参数。"""
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=base_dir / "datasets" / "raw")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "datasets")
    parser.add_argument("--dev-per-source", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """下载原始数据并生成训练集、固定评测集和完整评测池。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = download_dataset(args.raw_dir)

    _, train_counts = prepare_split(raw_dir / "train.parquet", args.output_dir / "train.jsonl")
    test_records, test_counts = prepare_split(
        raw_dir / "test.parquet",
        args.output_dir / "test.jsonl",
        collect_records=True,
    )
    dev_records = select_dev(test_records, args.dev_per_source, args.seed)
    write_jsonl(dev_records, args.output_dir / "dev.jsonl")

    print_counts("train", train_counts)
    print_counts("test", test_counts)
    print_counts("dev", Counter(record["data_source"] for record in dev_records))


if __name__ == "__main__":
    main()
