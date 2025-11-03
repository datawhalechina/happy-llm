
# 一、LoRA的核心思想
LoRA，全称 **Low-Rank Adaptation of Large Language Models**，是一种在 **大模型中进行高效微调** 的方法，目标是 **只训练极少数参数** 就能让模型适应新任务，避免重新训练整个大模型，从而可以在没有充足GPU显存的情况下快速在自己的数据集上对大模型做微调。

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*XdoGFaSME49GrqfBPGJqdQ.png)

在Transformer、ViT、GPT等模型中，很多计算都包含线性层：
$$y = W x$$

$$
W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
$$

LoRA 的做法是：**不直接更新大模型参数W**，而是在其旁边**插入一个低秩矩阵BA**，作为可训练的残差项：

$$y = W x + BAx$$

其中：

$$ A \in \mathbb{R}^{r \times d_{\text{in}}} $$

$$ B \in \mathbb{R}^{d_{\text{out}} \times r} $$

$$ r \ll d_{\text{in}}, d_{\text{out}} $$

原先微调需要更新整个$W$，其参数量为

$$\text{Param}(W) = d_{\text{out}} \times d_{\text{in}}$，使用LoRA后，$B A$的参数量仅为$\text{Param}_{\text{LoRA}} = r \times d_{\text{in}} + d_{\text{out}} \times r = r (d_{\text{in}} + d_{\text{out}})$$

使用PyTorch实现LoRA类，如下：
```python
# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```
# 二、如何将LoRA注入到现有的LLM中？
下面的代码实现了这一功能：
```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():

        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 如果是 nn.Linear 且为方阵，则插入 LoRA 模块
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)  # 给 module 加一个 lora 成员变量
            original_forward = module.forward  # 保存原始 forward 方法

            # 构造新 forward：原始输出 + LoRA 输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora  # 替换 forward 方法
```

举个简单模型的例子：
```python
# 测试模型
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024)  # 方阵线性层

    @property
    def device(self):
        return next(self.parameters()).device  # 返回模型参数所在设备

    def forward(self, x):
        return self.linear(x)


model = TestModel()
print(model)
```
打印原始模型的结构：
```
TestModel(
  (linear): Linear(in_features=1024, out_features=1024, bias=True)
)
```
这表明TestModel有一个成员变量linear，是一个标准的nn.Linear层

注入LoRA：
```python
apply_lora(model)
print(model)
```

打印注入LoRA后的model：
```
TestModel(
  (linear): Linear(
    in_features=1024, out_features=1024, bias=True
    (lora): LoRA(
      (A): Linear(in_features=1024, out_features=16, bias=False)
      (B): Linear(in_features=16, out_features=1024, bias=False)
    )
  )
)
```
可以看到，lora层已经成功注入。

lora模块被注入到了nn.Linear中，成为nn.Linear这个module的一个成员变量。

我们可以打印模型的每一层：
```python
for name, module in model.named_modules():
    print(f"{name}: {module.__class__.__name__}")
```
```
: TestModel
linear: Linear
linear.lora: LoRA
linear.lora.A: Linear
linear.lora.B: Linear
```
# 三、LoRA权重的加载与保存
因为训练时只更新LoRA的参数，因此在保存和加载模型权重时，只需要处理更新的这部分LoRA参数。

首先，注入lora的model为：
```
TestModel(
  (linear): Linear(
    in_features=1024, out_features=1024, bias=True
    (lora): LoRA(
      (A): Linear(in_features=1024, out_features=16, bias=False)
      (B): Linear(in_features=16, out_features=1024, bias=False)
    )
  )
)
```
递归遍历打印模型中的所有模块：
```python
for name, module in model.named_modules():
    print(name,':',module)
```
如下：
```
: TestModel(
  (linear): Linear(
    in_features=1024, out_features=1024, bias=True
    (lora): LoRA(
      (A): Linear(in_features=1024, out_features=16, bias=False)
      (B): Linear(in_features=16, out_features=1024, bias=False)
    )
  )
)
linear : Linear(
  in_features=1024, out_features=1024, bias=True
  (lora): LoRA(
    (A): Linear(in_features=1024, out_features=16, bias=False)
    (B): Linear(in_features=16, out_features=1024, bias=False)
  )
)
linear.lora : LoRA(
  (A): Linear(in_features=1024, out_features=16, bias=False)
  (B): Linear(in_features=16, out_features=1024, bias=False)
)
linear.lora.A : Linear(in_features=1024, out_features=16, bias=False)
linear.lora.B : Linear(in_features=16, out_features=1024, bias=False)
```

可以看到，总共5个子模块。

我们只关心拥有`lora`属性的模块：
```python
# for name, module in model.named_modules():
#     attrs = [attr for attr in dir(module) if not attr.startswith('__')]
#     if 'lora' in attrs:
#         print(name,"------",module)
for name, module in model.named_modules():
    if hasattr(module, 'lora'):
        print(name,"------",module)
```
输出：
```
linear ------ Linear(
  in_features=1024, out_features=1024, bias=True
  (lora): LoRA(
    (A): Linear(in_features=1024, out_features=16, bias=False)
    (B): Linear(in_features=16, out_features=1024, bias=False)
  )
)
```
可以看到，只有第二个子模块`linear`具有`lora`属性，在模型训练时，也只有这一层的参数会被更新。

因此我们只需要保存`linear.lora`层的权重即可：
```python
def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            for k, v in module.lora.state_dict().items():
                state_dict[f"{name}.lora.{k}"] = v
    torch.save(state_dict, path)
    print(f"[LoRA] Saved {len(state_dict)} params to: {path}")

save_lora(model, "lora.pth")
```

加载保存的"lora.pth"，并解析其结构：
```python
lora = torch.load("lora.pth")
for k, v in lora.items():
    print(k, v.shape)
```

```
linear.lora.A.weight torch.Size([16, 1024])
linear.lora.B.weight torch.Size([1024, 16])
```
相应地，在加载训练好的模型权重时，也只是加载lora层的权重：
```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # replace用于隐掉{name}.lora.，因为load的执行者是module.lora.，不去掉会重复
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 调试信息
            for k, v in lora_state.items():
                print(k,'----',v.shape)
            print(module.lora)
            module.lora.load_state_dict(lora_state)
```

```python
load_lora(model, "lora.pth")
```
加载时的调试输出信息：
```
A.weight ---- torch.Size([16, 1024])
B.weight ---- torch.Size([1024, 16])
LoRA(
  (A): Linear(in_features=1024, out_features=16, bias=False)
  (B): Linear(in_features=16, out_features=1024, bias=False)
)
```

# 四、训练LoRA
这里将lora注入到MIniMind模型后，直接复用SFT的数据加载器和训练函数，相应的代码和SFT保持一致。

来看一下注入lora前后的训练参数量变化：
```python
# 初始化模型和分词器
model, tokenizer = init_model(lm_config)

# 注入 LoRA 模块（低秩适配器）
apply_lora(model)

# 计算总参数量
total_params = sum(p.numel() for p in model.parameters())

# 计算所有带有 "lora" 名字的参数量（即 LoRA 层的参数）
lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)

# 只在主进程中打印参数信息（DDP 分布式时）
if not ddp or dist.get_rank() == 0:
    print(f"LLM 总参数量: {total_params}")
    print(f"LoRA 参数量: {lora_params_count}")
    print(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

# 冻结除 LoRA 外的所有参数，只训练 LoRA 层
for name, param in model.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False

# 收集 LoRA 可训练参数
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        lora_params.append(param)

# 构建优化器，仅优化 LoRA 参数
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

# 构建数据集，这里复用SFT的数据加载器
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

# 如果使用分布式训练（DDP），使用 DistributedSampler 划分数据
train_sampler = DistributedSampler(train_ds) if ddp else None

# 构建数据加载器
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    pin_memory=True,
    drop_last=False,
    shuffle=False,  # 如果用 DDP，不能设置 shuffle
    num_workers=args.num_workers,
    sampler=train_sampler
)

# 使用自动混合精度（AMP），加速训练、节省显存，仅当使用 float16 或 bfloat16 时启用
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

# 每个 epoch 的迭代次数
iter_per_epoch = len(train_loader)

# 开始训练多个 epoch
for epoch in range(args.epochs):
    train_epoch(epoch, wandb)  # 执行单轮训练，wandb 可用于记录训练日志
```
输出：
```
LLM 总参数量: 26092032
LoRA 参数量: 262144
LoRA 参数占比: 1.00%
```
