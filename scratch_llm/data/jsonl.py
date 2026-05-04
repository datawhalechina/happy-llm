"""JSONL reading contracts for pretraining and SFT data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


def iter_jsonl_records(path: str | Path, encoding: str = "utf-8") -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a JSONL file.

    Args:
        path: File path to a .jsonl dataset.
        encoding: Text encoding used by the file.

    Returns:
        An iterator of dictionaries, one per line.

    TODO:
        Open the file, parse each non-empty line with json.loads, and include a
        useful line number in errors.
    """

    raise NotImplementedError("Implement JSONL record parsing in scratch_llm/data/jsonl.py")


def iter_jsonl_texts(
    path: str | Path,
    text_key: str = "text",
    encoding: str = "utf-8",
) -> Iterator[str]:
    """Yield text fields from JSONL records.

    Args:
        path: File path to a .jsonl dataset.
        text_key: Key containing the raw text for tokenizer/pretraining data.
        encoding: Text encoding used by the file.

    Returns:
        An iterator of text strings.
    """

    raise NotImplementedError("Implement text extraction using iter_jsonl_records")


def count_jsonl_lines(path: str | Path) -> int:
    """Count dataset rows without loading the whole file into memory.

    Args:
        path: File path to a .jsonl dataset.

    Returns:
        Number of newline-delimited records.
    """

    raise NotImplementedError("Implement line counting with a binary file scan")
