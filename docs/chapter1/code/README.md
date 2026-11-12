# 第一章代码实践（uv）

对应正文：[第一章 NLP 基础概念](../第一章%20NLP基础概念.md)  
**文本表示演进实验课（推荐精读+实操）：** [../1.4-文本表示演进实验课.md](../1.4-文本表示演进实验课.md)

本章示例刻意保持小规模：本地合成/教材风格语料，无需联网下载数据集。

## 环境（请用 uv，不要用 conda）

```bash
cd docs/chapter1/code
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
# 若 torch 很慢：
# uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 路径 A：搞清「稀疏 → N-gram → Word2Vec → 上下文」这条链（优先）

| 脚本 | 课 | 你要亲眼看到 |
| --- | --- | --- |
| `05_sparse_vsm_lab.py` | A. One-Hot/TF-IDF | 近义词正交、稀疏矩阵、TF-IDF≠语义嵌入 |
| `06_ngram_order_lab.py` | B. N-gram | 词袋分不清词序；通顺句 logP 更高；N 变大稀疏爆炸 |
| `07_static_embedding_lab.py` | C. Word2Vec | 近义词靠近；「苹果」在两句仍是同一向量 |
| `08_contextual_bilstm_lab.py` | D. ELMo 风格 | 迷你 BiLSTM：同一「苹果」在水果句/公司句向量分离 |

```bash
bash run_lesson_14.sh
```

每课结构均为：概念打印 → 数值实验 →「应观察到」→ 自测题。

## 路径 B：更早的「能跑通」示例

| 脚本 | 对应章节 | 做什么 |
| --- | --- | --- |
| `01_chinese_tokenize.py` | 1.3.1 / 1.3.2 | jieba + 迷你 BPE |
| `02_tfidf_classify.py` | 1.3.4 / 1.4.1 | TF-IDF 文本分类 |
| `03_ngram_lm.py` | 1.4.2 | trigram LM |
| `04_word2vec_skipgram.py` | 1.4.3 | NumPy Skip-Gram |

```bash
bash run_all.sh
```

## 说明

- 演进课目标是**教学直觉**，不是刷榜。  
- `08` 是 **ELMo 思想的迷你类比**（双向 LM 隐状态），不是加载原版 ELMo 权重。  
- 全链路可在 CPU 上数秒到约一分钟内跑完。
