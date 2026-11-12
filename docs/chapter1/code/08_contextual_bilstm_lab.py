#!/usr/bin/env python3
"""§1.4 第 D 课：上下文相关表示（ELMo 风格类比）。

两条对照：
1) 静态查表：emb(苹果) 永远相同
2) 上下文编码器：用左右邻居（不含自己）聚合成向量 → 水果句 / 公司句明显分开
另训练一个迷你双向 LSTM LM，说明「隐状态随整句变化」与 ELMo 同构。
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# 空格分词，保证「苹果」独立成词，便于一词多义对照
CORPUS = [
    "我 今天 中午 吃了 一个 苹果",
    "水果店 的 苹果 香蕉 都 很 新鲜",
    "这个 苹果 比 那个 香蕉 更 好吃",
    "他 削了 个 苹果 分享 给 朋友",
    "红 苹果 富含 维生素 很 营养",
    "苹果 公司 发布了 新款 手机",
    "苹果 公司 的 电脑 产品 很 受欢迎",
    "科技 巨头 苹果 召开 春季 发布会",
    "投资者 关注 苹果 公司 的 股价 表现",
    "苹果 股价 随着 新品 发布 上涨",
    "国王 和 王后 出席 国家 典礼",
    "电脑 和 计算机 都是 电子 设备",
    "自然语言 处理 让 计算机 理解 语言",
    "词向量 把 词语 映射 到 向量 空间",
] * 40


def tokenize(text: str) -> list[str]:
    return [t for t in text.strip().split() if t]


def build_vocab(corpus: list[str]):
    tokenized = [tokenize(s) for s in corpus]
    counts = Counter(t for s in tokenized for t in s)
    vocab = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common()]
    stoi = {w: i for i, w in enumerate(vocab)}
    return stoi, vocab, tokenized


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


class ContextEncoder(nn.Module):
    """CBOW 风格：邻居词嵌入均值 → MLP → 预测中心词。

    中心位置的「上下文向量」= MLP(mean(neighbors))，天然随邻居变化。
    """

    def __init__(self, vocab_size: int, dim: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.out = nn.Linear(dim, vocab_size)

    def context_vector(self, neighbor_ids: torch.Tensor) -> torch.Tensor:
        # neighbor_ids: [B, K]
        mask = (neighbor_ids != 0).float().unsqueeze(-1)
        e = self.emb(neighbor_ids) * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean = e.sum(dim=1) / denom
        return self.mlp(mean)

    def forward(self, neighbor_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context_vector(neighbor_ids)
        return self.out(ctx), ctx


class TinyBiLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hid: int = 48) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hid * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(self.emb(x))
        return self.out(h), h


def make_cbow_examples(tokenized, stoi, window: int = 2):
    xs, ys = [], []
    for sent in tokenized:
        ids = [stoi.get(t, 1) for t in sent]
        for i, center in enumerate(ids):
            neigh = []
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    neigh.append(ids[j])
            if not neigh:
                continue
            # pad neighbors to fixed length
            while len(neigh) < window * 2:
                neigh.append(0)
            xs.append(neigh[: window * 2])
            ys.append(center)
    return torch.tensor(xs), torch.tensor(ys)


def train_context_encoder(model, x, y, epochs=12, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 64):
            idx = perm[i : i + 64]
            logits, _ = model(x[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        print(f"  [CBOW上下文编码器] epoch {ep}/{epochs}  loss={total / len(x):.4f}")


def train_bilm(model, tokenized, stoi, epochs=6, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs + 1):
        random.shuffle(tokenized)
        total, n = 0.0, 0
        for sent in tokenized:
            ids = torch.tensor([[stoi.get(t, 1) for t in sent]])
            if ids.size(1) < 3:
                continue
            logits, _ = model(ids)
            loss = F.cross_entropy(logits.squeeze(0), ids.squeeze(0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"  [BiLSTM-LM] epoch {ep}/{epochs}  loss={total / max(n, 1):.4f}")


@torch.no_grad()
def ctx_vec_for_word(model: ContextEncoder, stoi, sentence: str, word: str, window: int = 2) -> np.ndarray:
    toks = tokenize(sentence)
    idx = toks.index(word)
    ids = [stoi.get(t, 1) for t in toks]
    neigh = []
    for j in range(max(0, idx - window), min(len(ids), idx + window + 1)):
        if j != idx:
            neigh.append(ids[j])
    while len(neigh) < window * 2:
        neigh.append(0)
    t = torch.tensor([neigh[: window * 2]])
    return model.context_vector(t)[0].detach().numpy()


@torch.no_grad()
def bilm_vec_for_word(model: TinyBiLM, stoi, sentence: str, word: str) -> np.ndarray:
    toks = tokenize(sentence)
    idx = toks.index(word)
    ids = torch.tensor([[stoi.get(t, 1) for t in toks]])
    _, h = model(ids)
    return h[0, idx].detach().numpy()


def main() -> None:
    print("【第 D 课】上下文相关表示（ELMo 风格类比）")
    print("映射：ELMo≈双向 LM 的位置隐状态；本课先用「邻居聚合」把差异放大讲清，再给迷你 BiLSTM。")

    stoi, vocab, tokenized = build_vocab(CORPUS)
    print(f"词表={len(vocab)}  句数={len(tokenized)}")

    section("D1. 训练上下文编码器（邻居 → 预测中心词）")
    x, y = make_cbow_examples(tokenized, stoi)
    enc = ContextEncoder(len(vocab))
    train_context_encoder(enc, x, y, epochs=12)

    section("D2. 关键对照：静态查表 vs 上下文向量")
    s_fruit = "我 今天 中午 吃了 一个 苹果"
    s_company = "苹果 公司 发布了 新款 手机"
    enc.eval()
    v_fruit = ctx_vec_for_word(enc, stoi, s_fruit, "苹果")
    v_company = ctx_vec_for_word(enc, stoi, s_company, "苹果")
    v_static = enc.emb.weight[stoi["苹果"]].detach().numpy()

    print(f"句1: {s_fruit}")
    print(f"句2: {s_company}")
    print(f"静态查表 cos(emb, emb) = {cos(v_static, v_static):.3f}  （恒为 1，两句共用）")
    print(f"上下文向量 cos(苹果|水果句, 苹果|公司句) = {cos(v_fruit, v_company):.3f}")
    print(f"  水果句 ctx 前8维: {np.round(v_fruit[:8], 4)}")
    print(f"  公司句 ctx 前8维: {np.round(v_company[:8], 4)}")
    print(
        "【应观察到】上下文余弦明显低于 1（两句邻居不同 → 向量分离）；"
        "静态表永远相同。这就是一词多义需要「随句变向量」的原因。"
    )

    section("D3. 同构补充：迷你 BiLSTM-LM 隐状态也会随句变化")
    bilm = TinyBiLM(len(vocab))
    train_bilm(bilm, tokenized, stoi, epochs=6)
    bilm.eval()
    b1 = bilm_vec_for_word(bilm, stoi, s_fruit, "苹果")
    b2 = bilm_vec_for_word(bilm, stoi, s_company, "苹果")
    print(f"BiLSTM 隐状态 cos(苹果|句1, 苹果|句2) = {cos(b1, b2):.3f}")
    print(
        "说明：原版 ELMo 是大规模双向 LM 的层组合；这里只保留「位置隐状态随上下文变」这一核心直觉。"
        "更强的全局上下文 → 第 2 章 Transformer。"
    )

    section("D 课自测")
    print("Q1. 为什么上下文向量能处理一词多义，而 Word2Vec 查表不能？")
    print("Q2. 本实验与原版 ELMo / Transformer 差在哪里？")
    print("答案：① 向量是上下文的函数，不是词 ID 的唯一查表  ② 规模、深层、自注意力并行与长程建模。")
    print("\nOK: contextual lab finished. 演进链四课跑通。")


if __name__ == "__main__":
    main()
