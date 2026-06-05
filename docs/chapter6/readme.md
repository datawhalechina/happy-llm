# 第六章实践说明

第六章聚焦如何基于 Transformers 生态完成大模型训练实践，正文主线为 Pretrain、SFT 与 PEFT 三部分，适合作为从手写模型实现过渡到工业界训练框架的桥梁章节。

## 1. 本章包含什么

- [`第六章 大模型训练流程实践`](./第六章%20大模型训练流程实践.md)：第六章正文，介绍训练框架、数据处理、Trainer、DeepSpeed 与 LoRA/QLoRA 等核心内容。
- [`6.4 偏好对齐专题补充`](./6.4%5BWIP%5D%20偏好对齐.md)：偏好对齐的补充阅读材料，可在完成正文学习后继续深入。
- [`code/`](./code)：本章配套代码目录，包含数据下载、预训练、SFT 与 DeepSpeed 配置示例。

## 2. 推荐学习顺序

1. 阅读第六章正文的 `6.1 模型预训练`，理解 Transformers、Trainer 与 DeepSpeed 的基础用法。
2. 阅读 `6.2 模型有监督微调`，对比 Pretrain 与 SFT 的数据构造与 loss 计算差异。
3. 阅读 `6.3 高效微调`，掌握 LoRA/QLoRA 的基本思路和 `peft` 的接入方式。
4. 完成上述内容后，再阅读偏好对齐补充材料，建立从 SFT 到 Post-Training 的完整认识。

## 3. 代码入口

本章的主要代码文件如下：

- [`code/download_model.py`](./code/download_model.py)：下载基座模型。
- [`code/download_dataset.py`](./code/download_dataset.py)：下载或准备训练数据。
- [`code/pretrain.py`](./code/pretrain.py)：预训练脚本。
- [`code/finetune.py`](./code/finetune.py)：有监督微调脚本。
- [`code/pretrain.sh`](./code/pretrain.sh)：DeepSpeed 预训练启动示例。
- [`code/finetune.sh`](./code/finetune.sh)：DeepSpeed 微调启动示例。
- [`code/ds_config_zero2.json`](./code/ds_config_zero2.json)：DeepSpeed 配置文件。

## 4. 环境建议

- 依赖文件：[`code/requirements.txt`](./code/requirements.txt)
- 推荐 Python 版本：3.10 或 3.11
- 推荐硬件：多卡 GPU；如果资源有限，建议优先使用小样本或单卡环境调试数据处理与训练流程

可以参考仓库中的 [`学习与环境准备`](../学习与环境准备.md) 页面统一准备环境。

## 5. 实践建议

### 5.1 先调通路径，再扩大规模

本章脚本中的 `autodl-tmp`、显卡编号、batch size 和输出目录都属于示例配置。建议先把模型路径、数据路径和输出路径替换为本地实际路径，再使用小样本验证流程是否跑通。

### 5.2 先看懂数据，再启动训练

无论是 Pretrain 还是 SFT，数据格式都直接决定了训练效果。建议先阅读正文中的数据处理部分，并在运行前打印 1 到 2 条样本进行检查。

### 5.3 将偏好对齐作为进阶主题

第六章正文已经覆盖了核心训练主线。若你希望进一步理解 RLHF、DPO、KTO 与奖励模型，可以在完成正文后继续阅读偏好对齐专题补充，而不必在第一次阅读时一次性掌握所有后训练细节。