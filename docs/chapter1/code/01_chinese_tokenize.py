#!/usr/bin/env python3
"""Demo for §1.3.1 中文分词 and §1.3.2 子词切分.

Runs jieba word segmentation on textbook-style examples, then trains a tiny
character-level BPE tokenizer on a small Chinese corpus (no network needed).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import jieba


CHAPTER_EXAMPLES = [
    "今天天气真好，适合出去游玩。",
    "雍和宫的荷花开的很好。",
    "李雷和韩梅梅是北京市海淀区的居民。",
    "苹果公司发布了新款 Macbook，配备了最新的m3芯片。",
]

# Tiny in-repo corpus for BPE training (synthetic / textbook-aligned).
BPE_CORPUS = [
    "今天天气真好适合出去游玩",
    "雍和宫的荷花开的很好",
    "自然语言处理让计算机理解人类语言",
    "文本分类情感分析机器翻译自动问答",
    "词向量语言模型Word2Vec与子词切分",
    "深度学习推动了自然语言处理的发展",
    "北京市海淀区的居民计划去上海旅行",
    "湖人和勇士将在季后赛首轮对决",
] * 20


def demo_jieba() -> None:
    print("=== 1.3.1 中文分词（jieba）===")
    for text in CHAPTER_EXAMPLES:
        words = list(jieba.cut(text))
        print(f"输入: {text}")
        print(f"分词: {' | '.join(words)}")
        print()


def get_stats(vocab: dict[tuple[str, ...], int]) -> Counter:
    pairs: Counter = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    merged: dict[tuple[str, ...], int] = defaultdict(int)
    a, b = pair
    for word, freq in vocab.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merged[tuple(new_word)] += freq
    return dict(merged)


def train_bpe(corpus: Iterable[str], num_merges: int = 40) -> tuple[list[tuple[str, str]], list[str]]:
    vocab: dict[tuple[str, ...], int] = Counter()
    for sent in corpus:
        chars = tuple(list(sent) + ["</w>"])
        vocab[chars] += 1

    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        stats = get_stats(vocab)
        if not stats:
            break
        best = stats.most_common(1)[0][0]
        vocab = merge_vocab(best, vocab)
        merges.append(best)

    tokens = sorted({tok for word in vocab for tok in word})
    return merges, tokens


def apply_bpe(text: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(text) + ["</w>"]
    for a, b in merges:
        i = 0
        out: list[str] = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    if symbols and symbols[-1] == "</w>":
        symbols = symbols[:-1]
    elif symbols and symbols[-1].endswith("</w>"):
        symbols[-1] = symbols[-1][: -len("</w>")]
    return symbols


def demo_bpe() -> None:
    print("=== 1.3.2 子词切分（自实现迷你 BPE 训练）===")
    merges, tokens = train_bpe(BPE_CORPUS, num_merges=50)
    print(f"训练完成: {len(merges)} 次 merge，词表约 {len(tokens)} 个符号")
    print("前 10 次 merge:", merges[:10])
    sample = "自然语言处理"
    pieces = apply_bpe(sample, merges)
    print(f"示例输入: {sample}")
    print(f"BPE 切分: {pieces}")
    print()


def main() -> None:
    demo_jieba()
    demo_bpe()
    print("OK: tokenization demos finished.")


if __name__ == "__main__":
    main()
