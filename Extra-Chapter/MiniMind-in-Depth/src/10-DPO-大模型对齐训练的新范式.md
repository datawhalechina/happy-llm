
**DPO（Direct Preference Optimization）** 是一种用于有监督指令微调后模型偏好对齐的训练方法，目标是让模型更倾向于输出人类偏好的回答（`chosen`），而不是次优回答（`rejected`）。

# 一、查看DPO训练数据集格式
```python
import json
pretrain_dataset_path=r'D:\MyFile\github\minimind-master\minimind_dataset\dpo.jsonl'
with open(pretrain_dataset_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line.strip())
        break
        
print(data.keys()) # dict_keys(['chosen', 'rejected'])
print(data)
```

```
{
    'chosen': 
        [
            {
                'content': 'How many moles of HBr are required to react ...', 
                'role': 'user'
            }, 
            
            {
                'content': 'To determine the number of moles of HBr ...',
            
                'role': 'assistant'
            }
        ],
        
    'rejected': 
        [
            {
                'content': 'How many moles of HBr are required to react ...', 

                'role': 'user'
            }, 
                
            {
                'content': 'To answer this question, we need to write  ...', 
            
                'role': 'assistant'
            }
        ]
}
```

用于DPO训练的数据集中，每一条是数据都包含至少两个assistant回答，一个优、一个劣，“chosen”对应优，“rejected”对应劣。

在DPO训练时，模型会学习让“chosen”回答的概率高于“rejected”回答，从而实现偏好对齐。

# 二、准备DPO训练数据加载器

构建符合PyTorch的Dataloader的Dataset类：
```python
import json
import torch
from torch.utils.data import Dataset

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 特殊标记 <|im_start|>assistant 和 <|im_end|> 的 token ids（一般是开头和结尾的边界符）
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids  # list[int]
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids              # list[int]

        # 加载 JSONL 格式数据：每行为一个 dict，有 chosen 和 rejected
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        chosen = item['chosen']
        rejected = item['rejected']

        # 拼接成字符串（不 tokenize，只生成 prompt 文本）
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # 编码为 input_ids（截断 + 填充）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )

        # 转换为 token ID 列表，长度为 max_length
        chosen_input_ids = chosen_encoding['input_ids']           # shape: (max_length,)
        rejected_input_ids = rejected_encoding['input_ids']       # shape: (max_length,)

        # 构造 loss mask：仅在 assistant 段落（<|im_start|>assistant ... <|im_end|>）中的 token 参与损失
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)     # shape: (max_length,)
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids) # shape: (max_length,)

        # （MiniMind没有将padding的token掩掉）

        # 构造训练数据：左移一位预测（即 y 是 x 的下一位）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)      # shape: (max_length - 1,)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)       # shape: (max_length - 1,)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)    # shape: (max_length - 1,)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)  # shape: (max_length - 1,)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)   # shape: (max_length - 1,)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)# shape: (max_length - 1,)

        return {
            'x_chosen': x_chosen,           # shape: (max_length - 1,)
            'y_chosen': y_chosen,           # shape: (max_length - 1,)
            'mask_chosen': mask_chosen,     # shape: (max_length - 1,)

            'x_rejected': x_rejected,       # shape: (max_length - 1,)
            'y_rejected': y_rejected,       # shape: (max_length - 1,)
            'mask_rejected': mask_rejected  # shape: (max_length - 1,)
        }

    def _generate_loss_mask(self, input_ids):
        """
        根据 <|im_start|>assistant 和 <|im_end|> 的位置标记哪些 token 应该参与损失计算。
        返回一个和 input_ids 等长的 0/1 mask。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 匹配一个 assistant 段落开头
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 查找 assistant 的回答终止符 <|im_end|>
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 在 <|im_start|>assistant 和 <|im_end|> 之间部分启用 loss
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
```

`DPODataset`和之前的`SFTDataset`的处理逻辑是完全一致的，只不过`DPODataset`中需要处理两次(chosen和rejected)，因此上述代码中包含的函数介绍可以去看`SFTDataset`，这里不再重复介绍。

# 三、DPO 损失函数

DPO的目标是让训练后模型更偏好人类认为更好的答案（chosen），而不是差的答案（rejected），并且这种偏好是在对比参考模型(refrence model)的基础上学来的。

这里的参考模型，一般指的是微调前的模型，比如做了预训练和SFT之后的模型。

参考：https://allam.vercel.app/post/dpo/

DPO旨在以一种更简单、更稳定的方式替代传统RLHF中复杂的奖励建模过程。它的核心在于：使用一个直接可微的损失函数，来优化模型对人类偏好的响应倾向，而无需训练单独的奖励模型或使用复杂的强化学习方法（如PPO）。

具体来说，DPO在一对偏好样本上进行优化：它增加人类偏好响应中token的对数概率，同时减少非偏好响应中的对数概率，从而促使模型产生更符合人类意图的输出。

从数学角度看，这一过程相当于为模型引入了一个隐式奖励函数，该函数通过log-ratio的差值衡量当前策略相对于参考策略对人类偏好的一致程度，并直接用于梯度优化。


设：

- $\pi$ 是当前模型（policy model）
- $\pi_\text{ref}$ 是参考模型（reference model）
- $x$ 是输入 prompt
- $y^+$ 是人类偏好的回答（`chosen`）
- $y^-$ 是次优回答（`rejected`）
- $\beta$ 是温度超参（调节梯度幅度）

DPO loss 如下：

$$
\mathcal{L}_{\text{DPO}} = \mathbb{E}_{(x, y^+, y^-) \sim \mathcal{D}} \left[ -\log \sigma \left( \beta \cdot \left( \log \frac{\pi(y^+|x)}{\pi_{\text{ref}}(y^+|x)} - \log \frac{\pi(y^-|x)}{\pi_{\text{ref}}(y^-|x)} \right) \right) \right]
$$

其中 $\sigma$ 是 sigmoid 函数。

在上述公式的log差值项中，前一个表示模型对于人类偏好`chosen`回复$y^+$的对数概率，后一个表示模型对于`rejected`回复$y^-$的对数概率，DPO loss的目标是最大化两者的差值，也就是鼓励模型$\pi$相较于$\pi_\text{ref}$更加偏好$y^+$而非$y^-$。其中除以$\pi_\text{ref}$的作用是作为一个正则化因子，确保训练后的模型过度偏离原始模型。

在MiniMind的代码实现中，根据对数运算的性质，调换了DPO loss中的对数项顺序，如下：

$$
\mathcal{L}_{\text{DPO}} = \mathbb{E}_{(x, y^+, y^-) \sim \mathcal{D}} \left[ -\log \sigma \left( \beta \cdot \left( \log \frac{\pi(y^+|x)}{\pi(y^-|x)} - \log \frac{\pi_{\text{ref}}(y^+|x)}{\pi_{\text{ref}}(y^-|x)} \right) \right) \right]
$$

代码实现上述DPO loss：
```python
def dpo_loss(ref_probs, probs, mask, beta):
    # ref_probs: (batch_size, seq_len) 来自参考模型（Reference Model）的 log-probs
    # probs: (batch_size, seq_len)     来自当前策略模型（Policy Model）的 log-probs
    # mask: (batch_size, seq_len)      用于标记哪些 token 被计入损失（如生成部分）
    # beta: float                      DPO 的超参数控制分布偏移强度

    # Step 1: 每个样本的有效长度（非 padding 部分 token 的数量）
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

    # Step 2: 对每个样本计算平均 log-probs，仅在 mask == 1 的位置有效
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze(1)  # (batch_size,)
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze(1)          # (batch_size,)

    # Step 3: 将 batch 划分为前一半为 chosen，后一半为 rejected
    batch_size = ref_probs.shape[0]  # 假设 batch_size 是偶数，前半是 chosen，后半是 rejected

    chosen_ref_probs = ref_probs[:batch_size // 2]     # (batch_size // 2,)
    reject_ref_probs = ref_probs[batch_size // 2:]     # (batch_size // 2,)
    chosen_probs = probs[:batch_size // 2]             # (batch_size // 2,)
    reject_probs = probs[batch_size // 2:]             # (batch_size // 2,)

    # Step 4: log-ratio 比较（策略模型 vs 参考模型）
    pi_logratios = chosen_probs - reject_probs         # (batch_size // 2,)
    ref_logratios = chosen_ref_probs - reject_ref_probs  # (batch_size // 2,)

    # Step 5: DPO 损失计算，鼓励 chosen 比 rejected 的分数更高
    logits = pi_logratios - ref_logratios              # (batch_size // 2,)
    loss = -F.logsigmoid(beta * logits)                # (batch_size // 2,)

    return loss.mean()  # 标量，.mean()等价于DPO loss数学公式中的期望符号E
```

在Step 3中，之所以取batch的前后一半分别作为chosen和rejected，是因为在MiniMind的train函数中，为了并行执行训练，对chosen和rejected做了拼接（在数据加载器中做了padding，因此可以拼接），相应的代码在下一节展示。

# 四、开始训练DPO

训练DPO的代码在SFT训练代码的基础上，将交叉熵损失换成了DPO loss，如下：
```python
for step, batch in enumerate(train_loader):

    # x_chosen: (batch_size, seq_len)
    x_chosen = batch['x_chosen'].to(args.device)

    # x_rejected: (batch_size, seq_len)
    x_rejected = batch['x_rejected'].to(args.device)

    # 标签 token ids（通常是 x 向右 shift 一位）
    # y_chosen: (batch_size, seq_len)
    y_chosen = batch['y_chosen'].to(args.device)

    # y_rejected: (batch_size, seq_len)
    y_rejected = batch['y_rejected'].to(args.device)

    # mask_chosen: (batch_size, seq_len)，mask 表示哪些位置计算 loss（只在 assistant 回复部分）
    mask_chosen = batch['mask_chosen'].to(args.device)

    # mask_rejected: (batch_size, seq_len)
    mask_rejected = batch['mask_rejected'].to(args.device)

    # 拼接成整体 batch（大小为 2B）
    # x: (2 * batch_size, seq_len)
    x = torch.cat([x_chosen, x_rejected], dim=0)
    y = torch.cat([y_chosen, y_rejected], dim=0)
    mask = torch.cat([mask_chosen, mask_rejected], dim=0)

    # 设置学习率
    lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    with ctx:  # mixed precision/autocast 上下文
        with torch.no_grad():  # 冻结参考模型（ref_model）参数
            # ref_logits: (2 * batch_size, seq_len, vocab_size)
            ref_outputs = ref_model(x)
            ref_logits = ref_outputs.logits

        # 参考模型的 log prob，对应标签 token 的概率
        # ref_probs: (2 * batch_size, seq_len)
        ref_probs = logits_to_probs(ref_logits, y)
        ref_probs = ref_probs * mask  # 掩盖非 assistant 区域

        # 当前模型 logits
        # logits: (2 * batch_size, seq_len, vocab_size)
        outputs = model(x)
        logits = outputs.logits

        # 当前模型的 token-level log prob
        # probs: (2 * batch_size, seq_len)
        probs = logits_to_probs(logits, y)
        probs = probs * mask

        # 计算 DPO 损失（内部比较 probs[:batch_size] 与 probs[batch_size:]）
        loss = dpo_loss(ref_probs, probs, mask, beta=0.1)

        # 梯度累积处理
        loss = loss / args.accumulation_steps

    # 反向传播（混合精度）
    scaler.scale(loss).backward()

    # 累积完成后更新参数
    if (step + 1) % args.accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

上述代码中有一个函数`logits_to_probs`，可以将输入的logits(shape为[2 x batch_size, seq_len, vocab_size])转换成输出的对数概率probs（shape为[2 x batch_size, seq_len]），其定义如下：
```python
def logits_to_probs(logits, labels):
    # logits: Tensor of shape (batch_size, seq_len, vocab_size)
    # labels: Tensor of shape (batch_size, seq_len)

    # Step 1: 计算每个 token 的 log-softmax 概率
    log_probs = F.log_softmax(logits, dim=2)  
    # log_probs: (batch_size, seq_len, vocab_size)

    # Step 2: 收集 labels 对应的 log 概率
    # labels.unsqueeze(2): (batch_size, seq_len, 1)
    # torch.gather(..., dim=2): 从 log_probs 的第3维（vocab_size）中选择对应 label 的概率
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2))  
    # probs: (batch_size, seq_len, 1)

    probs = probs.squeeze(-1)  
    # probs: (batch_size, seq_len)  => 每个 token 的 log-probability

    return probs
```

输入的`logits`表示模型在该位置预测下一个token是词表中某个词的原始分数，shape为[batch_size, seq_len, vocab_size]。

第一步，将`logits`使用log_softmax转换为对数概率`log_probs`，即`log_probs`表示模型在该位置预测下一个token是词表中某个词的对数概率，shape不变。

第二步，通过torch.gather，从`log_probs`中查询输入的真实标签`labels`中每个token对应位置的log概率，shape为[batch_size, seq_len]，这是每个位置上真实标签的模型预测对数概率，也就是DPO loss的输入。
