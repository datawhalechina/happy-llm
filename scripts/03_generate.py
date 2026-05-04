"""Generate text from a scratch checkpoint.

Run after you implement checkpoint loading and generation.
"""

from __future__ import annotations

import argparse

import torch

from scratch_llm.config import GenerationConfig, ModelConfig
from scratch_llm.inference import generate
from scratch_llm.model import ScratchLLM
from scratch_llm.tokenizer import load_tokenizer
from scratch_llm.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse generation CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate with scratch_llm")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer-dir", default="scratch_llm_runs/tokenizer", help="Tokenizer directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="0 means greedy decoding")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling")
    parser.add_argument("--vocab-size", type=int, default=6144, help="Vocabulary size")
    parser.add_argument("--dim", type=int, default=512, help="Model hidden size")
    parser.add_argument("--n-layers", type=int, default=8, help="Number of decoder blocks")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    return parser.parse_args()


def main() -> None:
    """Load model and tokenizer, then print decoded generation."""

    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    model = ScratchLLM(model_config).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )
    new_ids = generate(model, input_ids, gen_config)
    text = tokenizer.decode(torch.cat([input_ids, new_ids], dim=1)[0], skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()
