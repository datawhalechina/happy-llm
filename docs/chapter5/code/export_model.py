import torch
import warnings
from transformers import AutoTokenizer
from k_model import Transformer, ModelConfig

warnings.filterwarnings('ignore', category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_model(tokenizer_path, model_config, model_ckpt_path, save_directory):
    # 注册自定义类和配置
    ModelConfig.register_for_auto_class()
    Transformer.register_for_auto_class("AutoModelForCausalLM")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token_id is not None:
        model_config.pad_token_id = tokenizer.pad_token_id

    # 初始化模型
    model = Transformer(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型权重
    state_dict = torch.load(model_ckpt_path, map_location=device)
    # 移除可能存在的多余前缀
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # 加载权重到模型
    model.load_state_dict(state_dict, strict=False)
    print(f'模型参数: {count_parameters(model)/1e6:.2f}M = {count_parameters(model)/1e9:.2f}B')

    # 保存完整模型和tokenizer
    model.save_pretrained(save_directory, safe_serialization=False)
    tokenizer.save_pretrained(save_directory)
    print(f'模型和tokenizer已保存至: {save_directory}')


if __name__ == '__main__':
    # 示例用法
    config = ModelConfig(
        dim=1024,
        n_layers=18,
    )

    export_model(
        tokenizer_path='./tokenizer_k/',
        model_config=config,
        model_ckpt_path='./BeelGroup_sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth',
        save_directory="k-model-215M"
    )
