"""Reproducibility helpers."""

from __future__ import annotations


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch random number generators.

    Args:
        seed: Integer random seed.
        deterministic: If True, request deterministic torch algorithms where
            available.
    """

    raise NotImplementedError("Implement random, numpy, torch, and CUDA seeding")
