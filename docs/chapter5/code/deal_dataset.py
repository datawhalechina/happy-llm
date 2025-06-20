import os
import json
from tqdm import tqdm

# pretrain_data 为运行download_dataset.sh时，下载的pretrain_data本地路径
pretrain_data = 'your local pretrain_data'
output_pretrain_data = 'seq_monkey_datawhale.jsonl'

# sft_data 为运行download_dataset.sh时，下载的sft_data本地路径
sft_data = 'your local sft_data'
output_sft_data = 'BelleGroup_sft.jsonl'

# 1 处理预训练数据
def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

with open(output_pretrain_data, 'a', encoding='utf-8') as pretrain:
    with open(pretrain_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data, desc=f"Processing lines in {pretrain_data}", leave=False):  # 添加行级别的进度条
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 2 处理SFT数据
def convert_message(data):
    """
    将原始数据转换为标准格式
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

with open(output_sft_data, 'a', encoding='utf-8') as sft:
    with open(sft_data, 'r') as f:
        data = f.readlines()
        for item in tqdm(data, desc="Processing", unit="lines"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')