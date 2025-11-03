
当调用搭建好的`MiniMindForCausalLM`类实例化一个模型之后，模型的参数是随机的，这个阶段的模型没有任何语言能力，无法进行有意义的文本生成或理解。

**预训练**使用大规模的无监督语料对模型进行训练，使其具备“理解和生成自然语言”的基础能力，为后续的**微调**提供一个好的起点。

# 一、查看预训练数据集格式

MiniMind预训练使用的数据集为`pretrain_hq.jsonl`，这是一个1.55GB的文件，里面包含了非常多条数据，这里查看其中的第一条数据作为示例：
```python
import json
pretrain_dataset_path=r'D:\MyFile\github\minimind-master\minimind_dataset\pretrain_hq.jsonl'
with open(pretrain_dataset_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line.strip())
        break
        
print(data.keys()) # dict_keys(['text'])
print(data)
```

```
{'text': '<|im_start|>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。<|im_end|> <|im_start|>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？<|im_end|> <|im_start|>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。<|im_end|> <|im_start|>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。<|im_end|> <|im_start|>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。<|im_end|> <|im_start|>帮我想一个有趣的标题。这个挺有趣的："如何成为一名成功的魔术师" 调皮的标题往往会吸引读者的注意力。<|im_end|> <|im_start|>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。<|im_end|> <|im_start|>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。<|im_end|>'}
```
可以看到，每一条数据都是一个字典格式，只包含一个键值对，key是固定的'text'，value是用于预训练的“一段文本”，这是一个以 
`<|im_start|>`和`<|im_end|>`为对话边界token的多轮指令-回答对话数据集片段。

# 二、准备预训练数据加载器

构建符合PyTorch的Dataloader的Dataset类：
```python
import json
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer      # 分词器，用于将文本转为token ID
        self.max_length = max_length    # 每条样本的最大token长度
        self.samples = self.load_data(data_path)  # 加载数据

    def load_data(self, path):
        """从文件中加载数据，每一行为一条JSON格式的样本"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 读取每一行，解析成字典结构
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        """返回样本数量"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        返回第 index 个样本：
        - X: 模型输入（input_ids[:-1]）
        - Y: 目标输出（input_ids[1:]）
        - loss_mask: 哪些token位置参与loss计算（去除padding部分）
        """
        sample = self.samples[index]

        # 将样本中的文本字段进行tokenize
        encoding = self.tokenizer(
            str(sample['text']),                 # 转为字符串（确保数据类型一致）
            max_length=self.max_length,          # 限制最大长度
            padding='max_length',                # 不足部分补pad
            truncation=True,                     # 超出部分截断
            return_tensors='pt'                  # 返回PyTorch tensor形式（包含batch维度）
        )

        # 获取input_ids张量，并去除batch维度（变成一维）
        input_ids = encoding.input_ids.squeeze()  # shape: [max_length]
        
        # 计算loss_mask：pad的位置不参与loss
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # shape: [max_length]，bool类型

        # 语言模型是自回归的，使用前一个token预测下一个
        X = torch.tensor(input_ids[:-1], dtype=torch.long)         # 输入：[0, ..., n-2]
        Y = torch.tensor(input_ids[1:], dtype=torch.long)          # 目标：[1, ..., n-1]
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # loss_mask对齐目标Y

        return X, Y, loss_mask
```

构建数据加载器：
```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


max_length=512
data_path=r'D:\MyFile\github\minimind-master\minimind_dataset\pretrain_hq.jsonl'
tokenizer = AutoTokenizer.from_pretrained(r'D:\MyFile\github\minimind-master\model')
train_ds = PretrainDataset(data_path, tokenizer, max_length)

train_loader = DataLoader(
    train_ds,
    batch_size=2,
    pin_memory=True,
    drop_last=False,
    shuffle=False,
    num_workers=0,
)
```

查看数据总量以及数据的维度信息：
```python
print(len(train_loader)) # 706552
for item in train_loader:
    print([i.shape for i in item]) # [torch.Size([2, 511]), torch.Size([2, 511]), torch.Size([2, 511])]
    break
```

通过打印看到，数据总量为706552，每一条数据都包含3个PyTorch Tensor，分别是X, Y以及Y对应的padding mask（用于掩掉padding token的loss），shape都是`2x511`，2是batch_size，511是max_length-1，因为X和Y是正好是偏移一位的。

# 三、开始预训练

预训练代码和常规的模型训练代码几乎没有区别，核心代码段如下：
```python
# 定义交叉熵损失函数（不做reduction，保留每个token位置的loss）
loss_fct = nn.CrossEntropyLoss(reduction='none')

# CPU 不支持 float16 加速计算
ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

# 遍历训练数据加载器
for step, (X, Y, loss_mask) in enumerate(train_loader):
    # print(step)  # 可用于调试

    # 将数据转移到目标设备（如GPU）
    X = X.to(args.device)             # 输入 token 序列，形状: [batch_size, seq_len]
    Y = Y.to(args.device)             # 目标 token 序列，形状: [batch_size, seq_len]
    loss_mask = loss_mask.to(args.device)  # 用于遮蔽padding位置，形状: [batch_size, seq_len]

    # 使用自定义学习率调度函数更新学习率
    lr = get_lr(
        epoch * iter_per_epoch + step,                # 当前训练步数（全局step）
        args.epochs * iter_per_epoch,                 # 总训练步数
        args.learning_rate                            # 初始学习率
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr                        # 动态更新优化器中的学习率

    # 自动混合精度上下文（提高训练速度，降低显存）
    with ctx:  # ctx = autocast() 之类
        res = model(X)  # 前向传播，res.logits: [batch, seq_len, vocab_size]

        # 计算token级别的交叉熵损失（不做平均）
        loss = loss_fct(
            res.logits.view(-1, res.logits.size(-1)),  # 转为2D: [batch*seq_len, vocab_size]
            Y.view(-1)                                 # 展平目标: [batch*seq_len]
        ).view(Y.size())  # reshape回[batch, seq_len]

        # 仅在非pad的位置计算损失（通过loss_mask筛选）
        loss = (loss * loss_mask).sum() / loss_mask.sum()  # 平均有效token上的loss

        # 加入模型可能返回的辅助损失（如正则项等）
        loss += res.aux_loss

        # 梯度累积：将loss缩小为1/N，以模拟更大的batch
        loss = loss / args.accumulation_steps

    # 使用GradScaler进行反向传播，支持AMP混合精度
    scaler.scale(loss).backward()

    # 累积一定步数后才进行一次参数更新
    if (step + 1) % args.accumulation_steps == 0:
        # 取消scale，准备裁剪梯度（clip操作要求原始梯度）
        scaler.unscale_(optimizer)

        # 裁剪梯度，防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # 执行优化器步进
        scaler.step(optimizer)

        # 更新scaler内部状态
        scaler.update()

        # 清空梯度，准备下一次累计
        optimizer.zero_grad(set_to_none=True)
```
