#!/usr/bin/env python3
"""Demo for §1.4.2 语言模型（N-gram）.

Trains a smoothed bigram/trigram language model on a tiny Chinese corpus and
scores held-out sentences. Pure Python + jieba; finishes in under a second.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import jieba


CORPUS = [
    "今天天气真好适合出去游玩",
    "今天天气不错我们去公园散步",
    "自然语言处理让计算机理解人类语言",
    "自然语言处理是人工智能的重要分支",
    "文本分类可以把新闻分到体育政治或科技",
    "机器翻译把中文句子翻译成英文",
    "词向量把词语变成计算机可计算的数字",
    "语言模型可以估计一句话出现的概率",
    "深度学习推动自然语言处理快速发展",
    "雍和宫的荷花开的很好",
    "湖人和勇士将在季后赛首轮对决",
    "苹果公司发布了新款笔记本电脑",
] * 15


class NGramLM:
    def __init__(self, n: int = 3, alpha: float = 0.5) -> None:
        self.n = n
        self.alpha = alpha
        self.ngram_counts: dict[tuple[str, ...], int] = Counter()
        self.context_counts: dict[tuple[str, ...], int] = Counter()
        self.vocab: set[str] = set()

    def _tokenize(self, text: str) -> list[str]:
        return list(jieba.lcut(text))

    def fit(self, sentences: list[str]) -> None:
        for sent in sentences:
            tokens = ["<s>"] * (self.n - 1) + self._tokenize(sent) + ["</s>"]
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i : i + self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def log_prob(self, text: str) -> float:
        tokens = ["<s>"] * (self.n - 1) + self._tokenize(text) + ["</s>"]
        v = len(self.vocab)
        total = 0.0
        for i in range(self.n - 1, len(tokens)):
            ngram = tuple(tokens[i - self.n + 1 : i + 1])
            context = ngram[:-1]
            num = self.ngram_counts[ngram] + self.alpha
            den = self.context_counts[context] + self.alpha * v
            total += math.log(num / den)
        return total

    def perplexity(self, text: str) -> float:
        tokens = self._tokenize(text) + ["</s>"]
        lp = self.log_prob(text)
        return math.exp(-lp / max(len(tokens), 1))


def main() -> None:
    print("=== 1.4.2 N-gram 语言模型训练 ===")
    lm = NGramLM(n=3, alpha=0.5)
    lm.fit(CORPUS)
    print(f"词表大小: {len(lm.vocab)}, n={lm.n}")

    probes = [
        "今天天气真好适合出去游玩",
        "自然语言处理让计算机理解人类语言",
        "量子力学研究黑洞信息悖论",  # OOV-heavy / out-of-domain
    ]
    for text in probes:
        print(f"句子: {text}")
        print(f"  logP={lm.log_prob(text):.3f}  perplexity={lm.perplexity(text):.2f}")
    print("\nOK: N-gram LM trained and scored.")


if __name__ == "__main__":
    main()
