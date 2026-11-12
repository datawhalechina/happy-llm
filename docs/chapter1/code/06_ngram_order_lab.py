#!/usr/bin/env python3
"""§1.4 第 B 课：词袋丢词序 vs N-gram 语言模型。

教学目标：词序为何重要；N-gram 如何用局部条件概率补上；稀疏爆炸的直觉。
"""

from __future__ import annotations

import math
from collections import Counter

import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


CORPUS = [
    "今天天气真好适合出去游玩",
    "今天天气不错我们去公园散步",
    "自然语言处理让计算机理解人类语言",
    "语言模型可以估计句子出现的概率",
    "狗咬人是一条新闻标题",
    "人咬狗也是一条新闻标题",
    "国王和王后出席国家典礼",
    "电脑和计算机都是电子设备",
] * 20


class NGramLM:
    def __init__(self, n: int = 3, alpha: float = 0.5) -> None:
        self.n = n
        self.alpha = alpha
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocab: set[str] = set()

    def tok(self, text: str) -> list[str]:
        return list(jieba.lcut(text))

    def fit(self, sents: list[str]) -> None:
        for s in sents:
            toks = ["<s>"] * (self.n - 1) + self.tok(s) + ["</s>"]
            self.vocab.update(toks)
            for i in range(len(toks) - self.n + 1):
                ng = tuple(toks[i : i + self.n])
                self.ngram_counts[ng] += 1
                self.context_counts[ng[:-1]] += 1

    def log_prob(self, text: str) -> float:
        toks = ["<s>"] * (self.n - 1) + self.tok(text) + ["</s>"]
        v = len(self.vocab)
        total = 0.0
        for i in range(self.n - 1, len(toks)):
            ng = tuple(toks[i - self.n + 1 : i + 1])
            ctx = ng[:-1]
            total += math.log((self.ngram_counts[ng] + self.alpha) / (self.context_counts[ctx] + self.alpha * v))
        return total

    def ppl(self, text: str) -> float:
        n_tok = len(self.tok(text)) + 1  # + </s>
        return math.exp(-self.log_prob(text) / max(n_tok, 1))

    def next_topk(self, context: str, k: int = 5) -> list[tuple[str, float]]:
        """给定上文，列出最可能的下一个词（示意）。"""
        ctx_toks = (["<s>"] * (self.n - 1) + self.tok(context))[-(self.n - 1) :]
        ctx = tuple(ctx_toks)
        v = len(self.vocab)
        scores = []
        for w in self.vocab:
            if w in {"<s>", "</s>"}:
                continue
            ng = ctx + (w,)
            p = (self.ngram_counts[ng] + self.alpha) / (self.context_counts[ctx] + self.alpha * v)
            scores.append((w, p))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def demo_bow_order_blind() -> None:
    section("B1. 词袋分不清「狗咬人」和「人咬狗」")
    s1, s2 = "狗咬人", "人咬狗"
    vec = CountVectorizer(tokenizer=lambda s: list(jieba.lcut(s)), token_pattern=None)
    x = vec.fit_transform([s1, s2])
    print("词表:", list(vec.get_feature_names_out()))
    print("计数向量:")
    print(x.toarray())
    cos = cosine_similarity(x[0], x[1])[0, 0]
    print(f"余弦相似度 = {cos:.4f}  （词集合相同 → 词袋认为几乎一样）")
    print("【应观察到】词序被扔掉了。新闻标题语义完全不同，向量却一样。")


def demo_ngram_prefers_fluent() -> None:
    section("B2. N-gram：通顺句概率高，乱序句更糟")
    lm = NGramLM(n=3, alpha=0.5)
    lm.fit(CORPUS)
    probes = [
        ("通顺", "今天天气真好适合出去游玩"),
        ("通顺", "自然语言处理让计算机理解人类语言"),
        ("乱序", "游玩出去适合真好天气今天"),
        ("乱序", "语言人类理解计算机让处理自然语言"),
    ]
    print(f"训练句数≈{len(CORPUS)}  词表={len(lm.vocab)}  n={lm.n}")
    for tag, text in probes:
        print(f"  [{tag}] logP={lm.log_prob(text):8.3f}  PPL={lm.ppl(text):7.2f}  | {text}")

    print("\n给定上文「今天天气」时，模型更想接什么？")
    for w, p in lm.next_topk("今天天气", k=6):
        print(f"  P(·|今天天气)  {w!r}: {p:.4f}")

    print("【应观察到】局部词序进入模型；通顺续写概率更高。")


def demo_sparsity_explosion() -> None:
    section("B3. N 变大 → 可能的 N-gram 组合暴增（稀疏直觉）")
    tokens = [t for s in CORPUS for t in jieba.lcut(s)]
    uniq = len(set(tokens))
    print(f"语料约 {len(tokens)} 个 token，词种 ≈ {uniq}")
    for n in (1, 2, 3, 4, 5):
        theoretical = uniq**n
        observed = len({tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)})
        print(f"  N={n}: 理论最多约 {theoretical:>12} 种，实际见到 {observed:>5} 种")
    print(
        "【应观察到】N 稍大，理论空间远大于实际计数 → 大量零计数，必须平滑，"
        "且仍难捕捉长距离依赖。痛点：离散符号，没有稠密语义几何。"
    )


def quiz() -> None:
    section("B 课自测")
    print("Q1. 词袋丢掉了什么？N-gram 用什么补上？")
    print("Q2. 为什么小语料上 N=5 往往不稳？")
    print("答案：① 词序；局部条件概率/前 N-1 词  ② 组合爆炸+零计数稀疏。")


def main() -> None:
    print("【第 B 课】N-gram 语言模型：找回局部词序")
    print("问题链位置：稀疏 VSM 之后 —— 让模型知道‘谁在谁后面’。")
    demo_bow_order_blind()
    demo_ngram_prefers_fluent()
    demo_sparsity_explosion()
    quiz()
    print("\nOK: n-gram order lab finished. 下一课：Word2Vec 稠密语义空间。")


if __name__ == "__main__":
    main()
