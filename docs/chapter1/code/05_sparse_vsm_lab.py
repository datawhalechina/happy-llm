#!/usr/bin/env python3
"""§1.4 第 A 课：One-Hot / TF-IDF 稀疏向量空间实验。

教学目标：看到稀疏性、近义词正交、TF-IDF 调权 ≠ 语义几何。
"""

from __future__ import annotations

import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_onehot_orthogonality() -> None:
    section("A1. One-Hot：不同词永远正交（语义丢失）")
    vocab = ["国王", "王后", "电脑", "计算机", "苹果", "香蕉"]
    stoi = {w: i for i, w in enumerate(vocab)}
    eye = np.eye(len(vocab))
    print("词表:", vocab)
    print("One-Hot 矩阵 (每行一词):")
    print(eye.astype(int))
    nnz = int(eye.sum())
    print(f"非零元={nnz}/{eye.size}  稀疏率={(1 - nnz / eye.size) * 100:.1f}%")

    pairs = [("国王", "王后"), ("电脑", "计算机"), ("苹果", "香蕉"), ("国王", "电脑")]
    print("\n余弦相似度（One-Hot）:")
    for a, b in pairs:
        va, vb = eye[stoi[a]], eye[stoi[b]]
        cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        print(f"  cos({a}, {b}) = {cos:.4f}   ← 不同词必为 0")

    print(
        "\n【应观察到】近义词与无关词一样正交。"
        "痛点：能区分‘是哪个词’，但不能表达‘意思近不近’。"
    )


def demo_tfidf_weights() -> None:
    section("A2. TF-IDF：抬高区分性词，压低虚词（仍非语义嵌入）")
    docs = [
        "国王和王后出席了国家典礼",
        "电脑和计算机其实指同类电子设备",
        "我今天吃了一个苹果和一根香蕉",
        "苹果公司发布了新款电脑",
        "的 了 在 是 很常见的虚词也会反复出现 的 了 的 了",
    ]
    # 空格分词方便 Count/Tfidf；中文用 jieba
    tokenized = [" ".join(jieba.lcut(d)) for d in docs]

    count = CountVectorizer()
    tfidf = TfidfVectorizer()
    x_count = count.fit_transform(tokenized)
    x_tfidf = tfidf.fit_transform(tokenized)
    names = tfidf.get_feature_names_out()

    print(f"文档数={len(docs)}  词表大小={len(names)}")
    print(f"TF 矩阵形状={x_count.shape}  非零={x_count.nnz}  稀疏率={(1 - x_count.nnz / np.prod(x_count.shape)) * 100:.1f}%")
    print(f"TF-IDF 矩阵形状={x_tfidf.shape}  非零={x_tfidf.nnz}")

    # 对比某一常见词 vs 主题词在各文档的平均权重
    def avg_weight(mat, term: str) -> float:
        if term not in names:
            return float("nan")
        j = list(names).index(term)
        col = mat[:, j].toarray().ravel()
        return float(col.mean())

    for term in ["的", "了", "国王", "电脑", "苹果"]:
        if term in names:
            print(
                f"  词「{term}」: 平均TF={avg_weight(x_count, term):.3f}  "
                f"平均TF-IDF={avg_weight(x_tfidf, term):.3f}"
            )

    print("\n文档两两余弦（TF-IDF）:")
    sim = cosine_similarity(x_tfidf)
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            print(f"  doc{i} vs doc{j}: {sim[i, j]:.3f}")
            print(f"    · {docs[i][:22]}…")
            print(f"    · {docs[j][:22]}…")

    print(
        "\n【应观察到】TF-IDF 会改变词权重、影响文档相似度；"
        "但它仍是稀疏词袋几何，不会让「电脑≈计算机」自动变成近邻词向量。"
    )


def demo_synonym_bag_failure() -> None:
    section("A3. 同义改写：词袋/TF-IDF 未必认出‘同一意思’")
    pairs = [
        ("我买了一台电脑", "我购买了一台计算机"),
        ("国王会见王后", "天气预报说今晚有雨"),
    ]
    vec = TfidfVectorizer()
    corpus = [" ".join(jieba.lcut(t)) for pair in pairs for t in pair]
    x = vec.fit_transform(corpus)
    print("同义句 vs 无关句 的 TF-IDF 余弦：")
    print(f"  同义改写: {cosine_similarity(x[0], x[1])[0, 0]:.3f}")
    print(f"  无关句子: {cosine_similarity(x[2], x[3])[0, 0]:.3f}")
    print(
        "【应观察到】若用词不同，同义句相似度可能并不高；"
        "稀疏 VSM 缺的是跨词的语义共享。→ 这推动了 Word2Vec。"
    )


def quiz() -> None:
    section("A 课自测（先想再看答案）")
    print("Q1. One-Hot 维度由什么决定？")
    print("Q2. 为什么任意两个不同词的 One-Hot 余弦一定是 0？")
    print("Q3. TF-IDF 主要改进的是语义几何，还是词对文档的区分权重？")
    print("\n答案要点：")
    print("A1. 词表大小 V。")
    print("A2. 不同标准基向量点积为 0。")
    print("A3. 主要是区分权重（检索/分类常用），不是近义词嵌入。")


def main() -> None:
    print("【第 A 课】稀疏向量空间（One-Hot / TF-IDF）")
    print("问题链位置：文本→数字（能算了）——但语义与词序仍弱。")
    demo_onehot_orthogonality()
    demo_tfidf_weights()
    demo_synonym_bag_failure()
    quiz()
    print("\nOK: sparse VSM lab finished. 下一课：N-gram 找回局部词序。")


if __name__ == "__main__":
    main()
