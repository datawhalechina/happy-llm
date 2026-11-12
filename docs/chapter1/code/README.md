# 第一章代码实践（uv）

对应正文：[第一章 NLP 基础概念](../第一章%20NLP基础概念.md)

本章示例刻意保持小规模：全部使用本地合成/教材风格语料，无需联网下载数据集，可在普通 CPU 上几秒到几十秒内跑完。

## 环境（请用 uv，不要用 conda）

在仓库根目录执行：

```bash
cd docs/chapter1/code
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

若系统尚未安装 uv：`curl -LsSf https://astral.sh/uv/install.sh | sh`

## 示例一览

| 脚本 | 对应章节 | 做什么 |
| --- | --- | --- |
| `01_chinese_tokenize.py` | 1.3.1 / 1.3.2 | jieba 中文分词 + 迷你 BPE 训练与切分 |
| `02_tfidf_classify.py` | 1.3.4 / 1.4.1 | TF-IDF 向量空间 + 逻辑回归文本分类（训练+评估） |
| `03_ngram_lm.py` | 1.4.2 | 加平滑的 trigram 语言模型训练与困惑度 |
| `04_word2vec_skipgram.py` | 1.4.3 | 迷你 Skip-Gram + 负采样词向量训练（纯 NumPy） |

## 运行

```bash
python 01_chinese_tokenize.py
python 02_tfidf_classify.py
python 03_ngram_lm.py
python 04_word2vec_skipgram.py
```

或一键：

```bash
bash run_all.sh
```

## 说明

- 示例目标是**把正文概念跑通**，不是刷榜；分类准确率与词向量相似度仅作直观演示。
- 依赖刻意保持轻量（NumPy / scikit-learn / jieba），CPU 即可；Word2Vec 用纯 NumPy 实现，便于对照正文公式。
- ELMo / 大规模预训练不在本章范围（见后续章节）。
