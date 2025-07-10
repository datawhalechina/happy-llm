<div align='center'>
    <img src="./images/head.jpg" alt="alt text" width="100%">
    <h1>Happy-LLM</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/datawhalechina/happy-llm?style=flat&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/datawhalechina/happy-llm?style=flat&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/badge/language-Chinese-brightgreen?style=flat" alt="Language"/>
  <a href="https://github.com/datawhalechina/happy-llm"><img src="https://img.shields.io/badge/GitHub-Project-blue?style=flat&logo=github" alt="GitHub Project"></a>
  <a href="https://swanlab.cn/@kmno4/Happy-LLM/overview"><img src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" alt="SwanLab"></a>
</div>

<div align="center">
  <a href="https://trendshift.io/repositories/14175" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14175" alt="datawhalechina%2Fhappy-llm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
  <h3>📚 从零开始的大语言模型原理与实践教程</h3>
  <p><em>深入理解 LLM 核心原理，动手实现你的第一个大模型</em></p>
</div>

---

## 🎯 项目介绍

> &emsp;&emsp;*很多小伙伴在看完 Datawhale开源项目： [self-llm 开源大模型食用指南](https://github.com/datawhalechina/self-llm) 后，感觉意犹未尽，想要深入了解大语言模型的原理和训练过程。于是我们（Datawhale）决定推出《Happy-LLM》项目，旨在帮助大家深入理解大语言模型的原理和训练过程。*

&emsp;&emsp;本项目是一个**系统性的 LLM 学习教程**，将从 NLP 的基本研究方法出发，根据 LLM 的思路及原理逐层深入，依次为读者剖析 LLM 的架构基础和训练过程。同时，我们会结合目前 LLM 领域最主流的代码框架，演练如何亲手搭建、训练一个 LLM，期以实现授之以鱼，更授之以渔。希望大家能从这本书开始走入 LLM 的浩瀚世界，探索 LLM 的无尽可能。

### ✨ 你将收获什么？

- 📚 **Datawhale 开源免费** 完全免费的学习本项目所有内容
- 🔍 **深入理解** Transformer 架构和注意力机制
- 📚 **掌握** 预训练语言模型的基本原理
- 🧠 **了解** 现有大模型的基本结构
- 🏗️ **动手实现** 一个完整的 LLaMA2 模型
- ⚙️ **掌握训练** 从预训练到微调的全流程
- 🚀 **实战应用** RAG、Agent 等前沿技术

## 📖 内容导航

| 章节 | 关键内容 | 状态 |
| --- | --- | --- |
| [前言](./前言.md) | 本项目的缘起、背景及读者建议 | ✅ |
| [第一章 NLP 基础概念](./chapter1/第一章%20NLP基础概念.md) | 什么是 NLP、发展历程、任务分类、文本表示演进 | ✅ |
| [第二章 Transformer 架构](./chapter2/第二章%20Transformer架构.md) | 注意力机制、Encoder-Decoder、手把手搭建 Transformer | ✅ |
| [第三章 预训练语言模型](./chapter3/第三章%20预训练语言模型.md) | Encoder-only、Encoder-Decoder、Decoder-Only 模型对比 | ✅ |
| [第四章 大语言模型](./chapter4/第四章%20大语言模型.md) | LLM 定义、训练策略、涌现能力分析 | ✅ |
| [第五章 动手搭建大模型](./chapter5/第五章%20动手搭建大模型.md) | 实现 LLaMA2、训练 Tokenizer、预训练小型 LLM | ✅ |
| [第六章 大模型训练实践](./chapter6/第六章%20大模型训练流程实践.md) | 预训练、有监督微调、LoRA/QLoRA 高效微调 | 🚧 |
| [第七章 大模型应用](./chapter7/第七章%20大模型应用.md) | 模型评测、RAG 检索增强、Agent 智能体 | ✅ |

### 模型下载

| 模型名称 | 下载地址 |
| --- | --- |
| Happy-LLM-Chapter5-Base-215M | [🤖 ModelScope](https://www.modelscope.cn/models/kmno4zx/happy-llm-215M-base) |
| Happy-LLM-Chapter5-SFT-215M | [🤖 ModelScope](https://www.modelscope.cn/models/kmno4zx/happy-llm-215M-sft) |

> *ModelScope 创空间体验地址：[🤖 创空间](https://www.modelscope.cn/studios/kmno4zx/happy_llm_215M_sft)*

### PDF 版本下载

&emsp;&emsp;***本 Happy-LLM PDF 教程完全开源免费。为防止各类营销号加水印后贩卖给大模型初学者，我们特地在 PDF 文件中预先添加了不影响阅读的 Datawhale 开源标志水印，敬请谅解～***

> *Happy-LLM PDF : https://github.com/datawhalechina/happy-llm/releases/tag/PDF*  
> *Happy-LLM PDF 国内下载地址 : https://www.datawhale.cn/learn/summary/179* 

## 💡 如何学习

&emsp;&emsp;本项目适合大学生、研究人员、LLM 爱好者。在学习本项目之前，建议具备一定的编程经验，尤其是要对 Python 编程语言有一定的了解。最好具备深度学习的相关知识，并了解 NLP 领域的相关概念和术语，以便更轻松地学习本项目。

&emsp;&emsp;本项目分为两部分——基础知识与实战应用。第1章～第4章是基础知识部分，从浅入深介绍 LLM 的基本原理。其中，第1章简单介绍 NLP 的基本任务和发展，为非 NLP 领域研究者提供参考；第2章介绍 LLM 的基本架构——Transformer，包括原理介绍及代码实现，作为 LLM 最重要的理论基础；第3章整体介绍经典的 PLM，包括 Encoder-Only、Encoder-Decoder 和 Decoder-Only 三种架构，也同时介绍了当前一些主流 LLM 的架构和思想；第4章则正式进入 LLM 部分，详细介绍 LLM 的特点、能力和整体训练过程。第5章～第7章是实战应用部分，将逐步带领大家深入 LLM 的底层细节。其中，第5章将带领大家者基于 PyTorch 层亲手搭建一个 LLM，并实现预训练、有监督微调的全流程；第6章将引入目前业界主流的 LLM 训练框架 Transformers，带领学习者基于该框架快速、高效地实现 LLM 训练过程；第7章则将介绍 基于 LLM 的各种应用，补全学习者对 LLM 体系的认知，包括 LLM 的评测、检索增强生产（Retrieval-Augmented Generation，RAG）、智能体（Agent）的思想和简单实现。你可以根据个人兴趣和需求，选择性地阅读相关章节。

&emsp;&emsp;在阅读本书的过程中，建议你将理论和实际相结合。LLM 是一个快速发展、注重实践的领域，我们建议你多投入实战，复现本书提供的各种代码，同时积极参加 LLM 相关的项目与比赛，真正投入到 LLM 开发的浪潮中。我们鼓励你关注 Datawhale 及其他 LLM 相关开源社区，当遇到问题时，你可以随时在本项目的 issue 区提问。

&emsp;&emsp;最后，欢迎每一位读者在学习完本项目后加入到 LLM 开发者的行列。作为国内 AI 开源社区，我们希望充分聚集共创者，一起丰富这个开源 LLM 的世界，打造更多、更全面特色 LLM 的教程。星火点点，汇聚成海。我们希望成为 LLM 与普罗大众的阶梯，以自由、平等的开源精神，拥抱更恢弘而辽阔的 LLM 世界。

## 🤝 如何贡献

我们欢迎任何形式的贡献！

- 🐛 **报告 Bug** - 发现问题请提交 Issue
- 💡 **功能建议** - 有好想法就告诉我们
- 📝 **内容完善** - 帮助改进教程内容
- 🔧 **代码优化** - 提交 Pull Request

## 🙏 致谢

### 核心贡献者
- [宋志学-项目负责人](https://github.com/KMnO4-zx) (Datawhale成员-中国矿业大学(北京))
- [邹雨衡-项目负责人](https://github.com/logan-zou) (Datawhale成员-对外经济贸易大学)
- [朱信忠-指导专家](https://xinzhongzhu.github.io/)（Datawhale首席科学家-浙江师范大学杭州人工智能研究院教授）

### 特别感谢
- 感谢 [@Sm1les](https://github.com/Sm1les) 对本项目的帮助与支持
- 感谢所有为本项目做出贡献的开发者们 ❤️

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/happy-llm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/happy-llm" />
  </a>
</div>

## Star History

<div align='center'>
    <img src="./images/star-history-2025710.png" alt="Datawhale" width="90%">
</div>

<div align="center">
  <p>⭐ 如果这个项目对你有帮助，请给我们一个 Star！</p>
</div>

## 关于 Datawhale

<div align='center'>
    <img src="./images/datawhale.png" alt="Datawhale" width="30%">
    <p>扫描二维码关注 Datawhale 公众号，获取更多优质开源内容</p>
</div>

---

## 📜 开源协议

本作品采用[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/)进行许可。
