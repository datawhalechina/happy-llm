#!/usr/bin/env python3
"""§1.4 第 C 课：Word2Vec 静态稠密向量 vs One-Hot。

教学目标：近义词靠近；但一词一向量 → 多义词无法分身。
"""

from __future__ import annotations

import random
from collections import Counter

import jieba
import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


CORPUS = [
    "国王和王后一起出席国家典礼",
    "王后与国王在宫殿中会面",
    "男人女人国王王后是经典类比",
    "电脑和计算机其实是同类设备",
    "我买了一台新电脑办公",
    "计算机科学需要学习编程",
    "我今天中午吃了一个苹果",
    "水果店的苹果香蕉都很新鲜",
    "苹果很好吃富含维生素",
    "苹果公司发布了新款手机",
    "苹果公司的电脑产品很受欢迎",
    "科技公司苹果发布会吸引关注",
    "自然语言处理让计算机理解语言",
    "词向量把词语映射到稠密空间",
] * 50


def build_pairs(window: int = 2):
    tokenized = [jieba.lcut(s) for s in CORPUS]
    counts = Counter(t for s in tokenized for t in s)
    vocab = ["<unk>"] + [w for w, c in counts.most_common() if c >= 2]
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = [[stoi.get(t, 0) for t in s] for s in tokenized]
    pairs = []
    for sent in ids:
        for i, c in enumerate(sent):
            for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                if i != j:
                    pairs.append((c, sent[j]))
    return stoi, vocab, pairs, ids


def noise_probs(ids, v: int) -> np.ndarray:
    counts = np.zeros(v)
    for s in ids:
        for t in s:
            counts[t] += 1
    p = np.power(counts, 0.75)
    return p / p.sum()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20))))


def train_skipgram(dim=32, epochs=5, neg=5, lr=0.05, seed=0):
    stoi, vocab, pairs, ids = build_pairs()
    v = len(vocab)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    w_in = rng.uniform(-0.5 / dim, 0.5 / dim, size=(v, dim))
    w_out = np.zeros((v, dim))
    noise = noise_probs(ids, v)

    for ep in range(1, epochs + 1):
        random.shuffle(pairs)
        loss = 0.0
        for center, pos in pairs:
            negs = rng.choice(v, size=neg, replace=True, p=noise)
            vc = w_in[center]
            # positive
            sp = float(np.dot(vc, w_out[pos]))
            pp = sigmoid(sp)
            loss += -np.log(pp + 1e-9)
            g = lr * (pp - 1.0)
            g_in = g * w_out[pos]
            w_out[pos] -= g * vc
            for n_id in negs:
                sn = float(np.dot(vc, w_out[n_id]))
                pn = sigmoid(sn)
                loss += -np.log(1.0 - pn + 1e-9)
                gn = lr * pn
                w_out[n_id] -= gn * vc
                g_in += gn * w_out[n_id]
            w_in[center] -= g_in
        print(f"  epoch {ep}/{epochs}  loss={loss / max(len(pairs), 1):.4f}")
    return stoi, vocab, w_in


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    print("【第 C 课】Word2Vec：稠密语义空间（静态词向量）")
    print("问题链位置：N-gram 之后 —— 让近义词共享几何，而不是正交。")

    section("C1. 训练迷你 Skip-Gram")
    stoi, vocab, emb = train_skipgram()
    print(f"词表大小={len(vocab)}")

    section("C2. 近义词靠近了吗？（对比 One-Hot 的 0）")
    related = [("国王", "王后"), ("电脑", "计算机"), ("苹果", "香蕉"), ("苹果", "公司")]
    random_pairs = [("国王", "香蕉"), ("王后", "电脑"), ("语言", "香蕉")]
    print("相关/主题相关词对:")
    for a, b in related:
        if a in stoi and b in stoi:
            print(f"  cos({a},{b}) = {cos(emb[stoi[a]], emb[stoi[b]]):.3f}   (One-Hot 恒为 0)")
    print("较无关词对:")
    for a, b in random_pairs:
        if a in stoi and b in stoi:
            print(f"  cos({a},{b}) = {cos(emb[stoi[a]], emb[stoi[b]]):.3f}")

    print("【应观察到】训练后相关词对相似度通常高于乱配；语义几何出现了。")

    section("C3. 静态之痛：同一「苹果」，两句共用一个向量")
    s_fruit = "我今天中午吃了一个苹果"
    s_company = "苹果公司发布了新款手机"
    if "苹果" not in stoi:
        raise SystemExit("语料中缺少「苹果」，请检查 CORPUS")
    v_static = emb[stoi["苹果"]]
    print(f"句1: {s_fruit}")
    print(f"句2: {s_company}")
    print("静态 Word2Vec 查表:")
    print(f"  vec(苹果|句1) is vec(苹果|句2) ?  {np.allclose(v_static, v_static)}  (永远 True)")
    print(f"  向量前 8 维: {np.round(v_static[:8], 4)}")
    print(
        "【应观察到】不管句子说水果还是公司，「苹果」只有一个点。"
        "这就是一词多义痛点 → 需要上下文相关表示（下一课）。"
    )

    section("C 课自测")
    print("Q1. Skip-Gram 用什么预测什么？")
    print("Q2. 「静态词向量」为何搞不定「苹果」？")
    print("答案：① 中心词→上下文（或相反 CBOW）  ② 查表与句子无关，多义共用一点。")
    print("\nOK: static embedding lab finished. 下一课：上下文向量（ELMo 风格）。")


if __name__ == "__main__":
    main()
