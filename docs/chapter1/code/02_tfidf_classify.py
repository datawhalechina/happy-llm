#!/usr/bin/env python3
"""Demo for §1.3.4 文本分类 + §1.4.1 词向量/VSM（TF-IDF）.

Trains a small TF-IDF + linear classifier on synthetic Chinese news snippets
aligned with the chapter's 体育 / 政治 / 科技 examples. CPU-only, seconds.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import jieba


def tokenize(text: str) -> str:
    return " ".join(jieba.lcut(text))


# Tiny labeled corpus (expanded variants of §1.3.4 examples). Enough to train.
RAW_DATA: list[tuple[str, str]] = [
    ("NBA季后赛将于下周开始，湖人和勇士将在首轮对决。", "体育"),
    ("足球世界杯预选赛今晚开战，国足力争三分。", "体育"),
    ("网球公开赛决赛上演五盘大战，冠军再度加冕。", "体育"),
    ("篮球联赛常规赛积分榜竞争激烈，卫冕冠军领先。", "体育"),
    ("游泳世锦赛打破多项纪录，中国队收获金牌。", "体育"),
    ("马拉松周末开跑，万名跑者挑战全马距离。", "体育"),
    ("羽毛球汤姆斯杯小组赛出线形势明朗。", "体育"),
    ("田径锦标赛百米飞人大战备受关注。", "体育"),
    ("冰球联赛总决赛加时绝杀，主队夺冠。", "体育"),
    ("乒乓球亚洲杯男女单打均闯进四强。", "体育"),
    ("电竞世界赛小组出线，粉丝涌入线下观赛。", "体育"),
    ("排球联赛主客场连胜，冲击联赛冠军。", "体育"),
    ("美国总统宣布将提高关税，引发国际贸易争端。", "政治"),
    ("国会就预算案展开辩论，两党分歧明显。", "政治"),
    ("外交部回应边境局势，呼吁各方保持克制。", "政治"),
    ("地方政府发布民生政策，推进公共服务改革。", "政治"),
    ("联合国安理会召开紧急会议讨论地区冲突。", "政治"),
    ("选举委员会公布投票结果，新一届议会产生。", "政治"),
    ("多国签署气候协议，承诺减少碳排放。", "政治"),
    ("议会通过网络安全法案，加强数据保护。", "政治"),
    ("内阁改组后公布新任部长名单。", "政治"),
    ("双边会谈就签证与留学政策达成共识。", "政治"),
    ("市政厅听证会上市民就住房问题发言。", "政治"),
    ("国际峰会闭幕，发表联合声明强调合作。", "政治"),
    ("苹果公司发布了新款 Macbook，配备了最新的m3芯片。", "科技"),
    ("开源大模型社区发布新版本，推理速度显著提升。", "科技"),
    ("量子计算初创公司宣布实现关键纠错里程碑。", "科技"),
    ("智能手机影像系统升级，夜景拍摄更清晰。", "科技"),
    ("自动驾驶路测里程突破千万公里，安全性提升。", "科技"),
    ("芯片代工厂扩产先进制程，缓解算力供给紧张。", "科技"),
    ("云计算厂商推出新一代 GPU 实例服务科研训练。", "科技"),
    ("操作系统发布安全更新，修复多个高危漏洞。", "科技"),
    ("数据库内核优化后查询延迟明显下降。", "科技"),
    ("机器人实验室展示新型双足行走样机。", "科技"),
    ("卫星互联网星座完成新一轮组网发射。", "科技"),
    ("浏览器引入本地端侧 AI 摘要功能。", "科技"),
]


def main() -> None:
    texts = [tokenize(t) for t, _ in RAW_DATA]
    labels = [y for _, y in RAW_DATA]

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=1)),
            ("clf", MultinomialNB()),
        ]
    )
    # 3-fold CV on the tiny corpus; then fit on all data for demo inference.
    cv_pred = cross_val_predict(pipe, texts, labels, cv=3)
    pipe.fit(texts, labels)

    print("=== 1.3.4 / 1.4.1 TF-IDF 文本分类训练 ===")
    print(f"样本数: {len(texts)}（3-fold 交叉验证）")
    print(classification_report(labels, cv_pred, digits=3))

    demos = [
        "湖人队在季后赛首轮战胜对手晋级。",
        "两国领导人会晤讨论关税与贸易问题。",
        "新一代芯片让笔记本性能大幅提升。",
    ]
    print("推理示例（全量拟合后）:")
    for text in demos:
        label = pipe.predict([tokenize(text)])[0]
        print(f"  [{label}] {text}")

    # Show sparse VSM flavor: top TF-IDF terms for one doc
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    matrix = vec.transform([tokenize(demos[0])])
    feature_names = vec.get_feature_names_out()
    row = matrix.toarray()[0]
    top_idx = row.argsort()[::-1][:8]
    print("\nVSM/TF-IDF 示例（首条推理文本的高权重词）:")
    for i in top_idx:
        if row[i] > 0:
            print(f"  {feature_names[i]!r}: {row[i]:.4f}")
    print("\nOK: TF-IDF classifier trained and evaluated.")


if __name__ == "__main__":
    main()
