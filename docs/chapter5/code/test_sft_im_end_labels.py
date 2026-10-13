"""Regression: pad/eos share an id, so SFT labels must use ignore_index -100.

Run (from this directory, with numpy):
    python test_sft_im_end_labels.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

# Allow `from dataset import SFTDataset` when torch is installed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


IM_END_ID = 4  # tokenizer_k: pad_token == eos_token == <|im_end|>


def shift_labels(input_id, loss_mask):
    """The dataset contract: next-token labels + -100 on unsupervised positions."""
    input_id = np.array(input_id)
    Y = np.array(input_id[1:]).astype(np.int64)
    loss_mask = np.array(loss_mask[1:]).astype(np.int64)
    return np.where(loss_mask == 1, Y, -100).astype(np.int64), loss_mask


class FakeEncoding(dict):
    @property
    def data(self):
        return self


class FakeTokenizer:
    """Minimal chat tokenizer: pad and eos share id 4, matching tokenizer_k."""

    bos_token = "<|im_start|>"
    eos_token = "<|im_end|>"
    pad_token = "<|im_end|>"
    eos_token_id = IM_END_ID
    pad_token_id = IM_END_ID

    _table = {
        "<|im_start|>": 1,
        "system": 10,
        "user": 11,
        "assistant": 12,
        "\n": 2,
        "<|im_end|>": IM_END_ID,
        "hi": 20,
        "hello": 21,
        " ": 3,
    }

    def _encode(self, text: str):
        ids = []
        i = 0
        keys = sorted(self._table, key=len, reverse=True)
        while i < len(text):
            for key in keys:
                if text.startswith(key, i):
                    ids.append(self._table[key])
                    i += len(key)
                    break
            else:
                ids.append(30 + (ord(text[i]) % 50))
                i += 1
        return ids

    def __call__(self, text, **kwargs):
        return FakeEncoding(input_ids=self._encode(text))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class LabelContractTests(unittest.TestCase):
    def test_im_end_kept_prompt_and_pad_are_minus_100(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        text = tokenizer.apply_chat_template(messages)
        input_id = tokenizer(text)["input_ids"]
        max_length = len(input_id) + 3
        padding_len = max_length - len(input_id)
        padded = input_id + [tokenizer.pad_token_id] * padding_len

        # Same search as SFTDataset.generate_loss_mask
        mask = [0] * len(padded)
        a_sequence = tokenizer("<|im_start|>assistant\n")["input_ids"]
        n = len(padded)
        i = 0
        while i <= n - len(a_sequence):
            if padded[i : i + len(a_sequence)] == a_sequence:
                j = None
                for idx in range(i + len(a_sequence), n):
                    if padded[idx] == tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None:
                    for pos in range(i + len(a_sequence), j + 1):
                        mask[pos] = 1
                i += len(a_sequence)
            else:
                i += 1

        Y, shifted_mask = shift_labels(padded, mask)

        self.assertIn(IM_END_ID, Y.tolist(), "assistant <|im_end|> must remain a real label")
        im_end_positions = [k for k, y in enumerate(Y) if y == IM_END_ID]
        self.assertTrue(im_end_positions)
        for k in im_end_positions:
            self.assertEqual(shifted_mask[k], 1)

        self.assertTrue(np.any(Y == -100), "prompt / padding must use ignore_index -100")
        for k, (y, m) in enumerate(zip(Y, shifted_mask)):
            if m == 0:
                self.assertEqual(y, -100)
            else:
                self.assertNotEqual(y, -100)

    def test_pad_as_ignore_index_zeros_im_end_loss(self):
        """The failure mode in Transformer.forward before labels use -100."""
        # Old labels: padding is the same id as <|im_end|>, so ignore_index=pad
        # also drops the real stop token.
        targets_old = np.array([20, 21, IM_END_ID, IM_END_ID])
        ignored_old = targets_old == IM_END_ID
        self.assertTrue(ignored_old[2], "pad ignore_index drops the stop token")
        self.assertTrue(ignored_old[3])

        targets_hf = np.array([20, 21, IM_END_ID, -100])
        ignored_hf = targets_hf == -100
        self.assertFalse(ignored_hf[2], "im_end must stay in the loss")
        self.assertTrue(ignored_hf[3])

        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return

        logits = torch.zeros(4, 8)
        logits[2, IM_END_ID] = 5.0
        loss_pad_ignore = F.cross_entropy(
            logits, torch.tensor(targets_old), ignore_index=IM_END_ID, reduction="none"
        )
        self.assertEqual(loss_pad_ignore[2].item(), 0.0)
        loss_hf = F.cross_entropy(
            logits, torch.tensor(targets_hf), ignore_index=-100, reduction="none"
        )
        self.assertGreater(loss_hf[2].item(), 0.0)
        self.assertEqual(loss_hf[3].item(), 0.0)


class SFTDatasetIntegrationTests(unittest.TestCase):
    def test_dataset_returns_minus_100_and_keeps_im_end(self):
        try:
            import torch  # noqa: F401
            from dataset import SFTDataset
        except ImportError:
            self.skipTest("torch/pandas not installed")

        tokenizer = FakeTokenizer()
        sample = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sft.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
            ds = SFTDataset(path, tokenizer, max_length=32)
            X, Y, loss_mask = ds[0]

        self.assertTrue(torch.any(Y == -100))
        self.assertTrue(torch.any((Y == IM_END_ID) & (loss_mask == 1)))
        self.assertFalse(torch.any((Y == -100) & (loss_mask == 1)))


if __name__ == "__main__":
    unittest.main()
