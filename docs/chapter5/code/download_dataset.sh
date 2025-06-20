#!/bin/bash

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# dataset dir 下载到本地目录
dataset_dir="your local dataset dir"

# 下载预训练数据集， 需要预先安装modelscope，使用pip3 install modelscope安装
modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir ${dataset_dir}

# 解压预训练数据集
tar -xvf "${dataset_dir}/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2" -C "${dataset_dir}"

# 下载SFT数据集
huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  BelleGroup/train_3.5M_CN \
  --local-dir "${dataset_dir}/BelleGroup"
