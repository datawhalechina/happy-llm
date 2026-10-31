import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "docs" / "chapter5" / "code" / "dataset.py"


class Encoding(dict):
    @property
    def data(self):
        return self


class FakeTokenizer:
    pad_token_id = 4
    eos_token_id = 4

    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, text):
        if text == "<|im_start|>assistant\n":
            return {"input_ids": [1, 2, 3]}
        return Encoding({"input_ids": list(self.token_ids)})

    def apply_chat_template(self, sample, tokenize=False, add_generation_prompt=False):
        return sample["kind"]


def load_dataset_class():
    spec = importlib.util.spec_from_file_location("happy_llm_dataset", DATASET_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SFTDataset


class SFTLossMaskRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_class = load_dataset_class()

    def dataset_item(self, token_ids):
        tokenizer = FakeTokenizer(token_ids)
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", encoding="utf-8", delete=False
        )
        try:
            handle.write(json.dumps({"kind": "fixture"}) + "\n")
            handle.flush()
            handle.close()
            dataset = self.dataset_class(handle.name, tokenizer, max_length=8)
            return dataset[0]
        finally:
            if not handle.closed:
                handle.close()
            os.unlink(handle.name)

    def test_truncated_assistant_does_not_supervise_padding_eos(self):
        _, _, loss_mask = self.dataset_item([9, 9, 1, 2, 3, 5, 6])
        self.assertEqual(loss_mask.sum().item(), 0)

    def test_complete_assistant_keeps_real_eos_supervision(self):
        _, _, loss_mask = self.dataset_item([9, 9, 1, 2, 3, 5, 6, 4])
        self.assertEqual(loss_mask.sum().item(), 3)


if __name__ == "__main__":
    unittest.main()
