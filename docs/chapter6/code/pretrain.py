'''
预训练脚本
'''

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.datapipes.iter import IterableWrapper
from itertools import chain
import deepspeed
from typing import Optional,List

import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
import swanlab


logger = logging.getLogger(__name__)


# 超参类
@dataclass
class ModelArguments:
    """
    关于模型的参数
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "后训练使用，为预训练模型参数地址"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练使用，Config 文件地址"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练 Tokenizer 地址"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型训练使用的数据类型，推荐 bfloat16"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    关于训练的参数
    """

    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "设置的文本块长度"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使用线程数."},
    )

                
def main():

    # 加载脚本参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化 SwanLab
    swanlab.init(project="pretrain", experiment_name="from_scrach")
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 将日志级别设置为 INFO
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 训练整体情况记录
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查 checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出路径 ({training_args.output_dir}) 非空 "
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"从 {last_checkpoint}恢复训练"
            )

    # 设置随机数种子.
    set_seed(training_args.seed)

    # 初始化模型
    if model_args.config_name is not None:
        # from scrach
        config = AutoConfig.from_pretrained(model_args.config_name)
        logger.warning("你正在从零初始化一个模型")
        logger.info(f"模型参数配置地址：{model_args.config_name}")
        logger.info(f"模型参数：{config}")
        model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"预训练一个新模型 - Total size={n_params/2**20:.2f}M params")
    elif model_args.model_name_or_path is not None:
        logger.warning("你正在初始化一个预训练模型")
        logger.info(f"模型参数地址：{model_args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")
    else:
        logger.error("config_name 和 model_name_or_path 不能均为空")
        raise ValueError("config_name 和 model_name_or_path 不能均为空")

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    logger.info("完成 tokenzier 加载")
    logger.info(f"tokenzier 配置地址：{model_args.tokenizer_name}")

    # 加载预训练数据
    ds = load_dataset('json', data_files=data_args.train_files)
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练文件总数:{len(ds["train"])}')
    # logger.info(f"训练集采样：{ds["train"][0]}")

    # 文本 tokenize
    column_names = list(ds["train"].features)
    logger.info('训练集特征：', column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # tokenize 函数
    def tokenize_function(examples):
        output = tokenizer([item for item in examples[text_column_name]])
        return output

    # 仅主进程进行数据预处理
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = ds.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset"
        )

    # 文本切块
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "tokenizer 支持大于 1K 的上下文长度，默认设置为 1K"
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"设定的块长为 ({data_args.block_size}) ，大于模型的上下文长度"
                f"将块长设置为模型上下文长度：{tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        # 将文本段拼接起来
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # 计算拼起来的整体长度
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # 如果长度太长，进行分块
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }    
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="文本分块"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"文本分块到{block_size}",
            batch_size = 40000,
        )
        logger.info("完成数据预处理")
        train_dataset = lm_datasets["train"]
    
    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= IterableWrapper(train_dataset),
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    # 从 checkpoint 加载
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
            checkpoint = last_checkpoint

    logger.info("开始训练")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model() 

if __name__ == "__main__":
    main()