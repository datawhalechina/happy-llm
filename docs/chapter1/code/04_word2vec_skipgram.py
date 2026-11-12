#!/usr/bin/env python3
"""Demo for §1.4.3 Word2Vec（Skip-Gram + negative sampling）.

Trains a tiny embedding model with NumPy on an in-memory Chinese corpus.
No PyTorch required — finishes in a few seconds on CPU.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import jieba
import numpy as np


CORPUS = [
    "自然语言处理让计算机理解人类语言",
    "文本表示把词语变成向量方便计算",
    "词向量可以刻画语义相近的词语",
    "国王和王后在语义空间中距离很近",
    "男人女人国王王后是经典类比例子",
    "深度学习推动自然语言处理快速发展",
    "语言模型估计句子出现的概率",
    "文本分类识别体育政治科技新闻",
    "机器翻译把中文翻译成英文",
    "子词切分处理罕见词和未登录词",
    "今天天气真好适合出去游玩",
    "雍和宫的荷花开的很好",
    "湖人和勇士将在季后赛首轮对决",
    "苹果公司发布了新款笔记本电脑",
    "美国总统宣布将提高关税引发贸易争端",
] * 40


def build_dataset(window: int = 2):
    tokenized = [jieba.lcut(s) for s in CORPUS]
    counts = Counter(tok for sent in tokenized for tok in sent)
    vocab = ["<unk>"] + [w for w, c in counts.most_common() if c >= 2]
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = [[stoi.get(tok, 0) for tok in sent] for sent in tokenized]

    pairs: list[tuple[int, int]] = []
    for sent in ids:
        for i, center in enumerate(sent):
            left = max(0, i - window)
            right = min(len(sent), i + window + 1)
            for j in range(left, right):
                if i == j:
                    continue
                pairs.append((center, sent[j]))
    return ids, stoi, vocab, pairs


def noise_probs(ids: list[list[int]], vocab_size: int) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.float64)
    for sent in ids:
        for tok in sent:
            counts[tok] += 1.0
    probs = np.power(counts, 0.75)
    probs /= probs.sum()
    return probs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    ids, stoi, vocab, pairs = build_dataset()
    v = len(vocab)
    print("=== 1.4.3 Word2Vec Skip-Gram 训练（NumPy）===")
    print(f"词表={v} 训练对={len(pairs)}")

    # Input / output embeddings
    scale = 0.5 / args.dim
    w_in = rng.uniform(-scale, scale, size=(v, args.dim)).astype(np.float64)
    w_out = np.zeros((v, args.dim), dtype=np.float64)
    noise = noise_probs(ids, v)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        for center, pos in pairs:
            neg = rng.choice(v, size=args.neg, replace=True, p=noise)
            v_c = w_in[center]

            # positive
            score = float(np.dot(v_c, w_out[pos]))
            pred = sigmoid(np.array([score]))[0]
            total_loss += -np.log(pred + 1e-9)
            grad = args.lr * (pred - 1.0)
            g_pos = grad * v_c
            g_in = grad * w_out[pos]

            # negatives
            for n_id in neg:
                score_n = float(np.dot(v_c, w_out[n_id]))
                pred_n = sigmoid(np.array([score_n]))[0]
                total_loss += -np.log(1.0 - pred_n + 1e-9)
                grad_n = args.lr * pred_n
                w_out[n_id] -= grad_n * v_c
                g_in += grad_n * w_out[n_id]

            w_out[pos] -= g_pos
            w_in[center] -= g_in

        avg = total_loss / max(len(pairs), 1)
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.4f}")

    def sim(a: str, b: str) -> float:
        if a not in stoi or b not in stoi:
            return float("nan")
        return cosine(w_in[stoi[a]], w_in[stoi[b]])

    probes = [("国王", "王后"), ("男人", "女人"), ("自然语言", "处理"), ("体育", "科技")]
    print("\n余弦相似度抽查:")
    for a, b in probes:
        print(f"  sim({a}, {b}) = {sim(a, b):.3f}")

    query = "语言" if "语言" in stoi else vocab[min(1, v - 1)]
    q = w_in[stoi[query]]
    norms = np.linalg.norm(w_in, axis=1) + 1e-9
    scores = (w_in @ q) / (norms * (np.linalg.norm(q) + 1e-9))
    top = np.argsort(-scores)[:6]
    print(f"\n与「{query}」最近的词:")
    for idx in top:
        print(f"  {vocab[idx]}: {scores[idx]:.3f}")

    print("\nOK: Word2Vec Skip-Gram training finished.")


if __name__ == "__main__":
    main()
