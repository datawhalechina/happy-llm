"""读取 prepare_data.py 生成的本地 Search-R1 数据。"""

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchExample:
    """保存一道搜索问答题及其标准答案。"""

    id: str
    question: str
    answers: list[str]
    data_source: str


def load_examples(path: str | Path) -> list[SearchExample]:
    """从本地 JSONL 文件读取全部样本。"""
    examples: list[SearchExample] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            examples.append(
                SearchExample(
                    id=str(row["id"]),
                    question=row["question"],
                    answers=list(row["answers"]),
                    data_source=row["data_source"],
                )
            )
    return examples


def shuffled_examples(path: str | Path, seed: int) -> list[SearchExample]:
    """读取并按固定随机种子打乱训练样本。"""
    examples = load_examples(path)
    random.Random(seed).shuffle(examples)
    return examples


def take_batch(examples: list[SearchExample], start: int, batch_size: int) -> list[SearchExample]:
    """循环取出一个固定大小的训练问题批次。"""
    if not examples:
        return []
    return [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
