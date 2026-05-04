"""Train the scratch tokenizer.

Run after you implement scratch_llm/tokenizer.py.
"""

from __future__ import annotations

import argparse

from scratch_llm.config import TokenizerConfig
from scratch_llm.tokenizer import train_bpe_tokenizer, validate_tokenizer, load_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse tokenizer CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a scratch ByteLevel BPE tokenizer")
    parser.add_argument("--data-path", required=True, help="JSONL file containing raw text")
    parser.add_argument("--output-dir", default="scratch_llm_runs/tokenizer", help="Tokenizer output directory")
    parser.add_argument("--vocab-size", type=int, default=6144, help="Target vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum BPE pair frequency")
    parser.add_argument("--text-key", default="text", help="JSONL field containing raw text")
    parser.add_argument("--sample-text", default="Hello, LLM.", help="Text used for tokenizer validation")
    return parser.parse_args()


def main() -> None:
    """Train and validate the tokenizer."""

    args = parse_args()
    config = TokenizerConfig(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        text_key=args.text_key,
    )
    train_bpe_tokenizer(args.data_path, args.output_dir, config)
    tokenizer = load_tokenizer(args.output_dir)
    report = validate_tokenizer(tokenizer, args.sample_text)
    print(report)


if __name__ == "__main__":
    main()
