"""Pretrain the scratch decoder-only LLM.

Run after you implement the tokenizer, dataset, model, and training loop.
"""

from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scratch_llm.config import ModelConfig, TrainConfig
from scratch_llm.data import CausalLMDataset
from scratch_llm.model import ScratchLLM
from scratch_llm.tokenizer import load_tokenizer
from scratch_llm.training import train_one_epoch
from scratch_llm.utils import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse pretraining CLI arguments."""

    parser = argparse.ArgumentParser(description="Pretrain scratch_llm")
    parser.add_argument("--data-path", required=True, help="JSONL pretraining data path")
    parser.add_argument("--tokenizer-dir", default="scratch_llm_runs/tokenizer", help="Tokenizer directory")
    parser.add_argument("--output-dir", default="scratch_llm_runs/checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Context length")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--dim", type=int, default=512, help="Model hidden size")
    parser.add_argument("--n-layers", type=int, default=8, help="Number of decoder blocks")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="Number of KV heads")
    parser.add_argument("--vocab-size", type=int, default=6144, help="Tokenizer vocabulary size")
    return parser.parse_args()


def main() -> None:
    """Build model, dataset, optimizer, and run pretraining."""

    args = parse_args()
    seed_everything(42)

    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
    )
    train_config = TrainConfig(
        data_path=args.data_path,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )

    tokenizer = load_tokenizer(train_config.tokenizer_dir)
    dataset = CausalLMDataset(
        train_config.data_path,
        tokenizer,
        max_length=train_config.max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    model = ScratchLLM(model_config).to(train_config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )

    total_steps = args.epochs * max(len(dataloader), 1)
    step = 0
    for epoch in range(train_config.epochs):
        step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            config=train_config,
            epoch=epoch,
            total_steps=total_steps,
            start_step=step,
        )


if __name__ == "__main__":
    main()
