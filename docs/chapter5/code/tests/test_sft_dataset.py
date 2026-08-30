import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dataset import SFTDataset
from k_model import ModelConfig, Transformer


CODE_DIR = Path(__file__).resolve().parents[1]


class SFTDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(CODE_DIR / "tokenizer_k")

    def setUp(self):
        self.data_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            encoding="utf-8",
            delete=False,
        )
        json.dump(
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，有什么可以帮你？"},
                {"role": "user", "content": "再见"},
                {"role": "assistant", "content": "再见！"},
            ],
            self.data_file,
            ensure_ascii=False,
        )
        self.data_file.write("\n")
        self.data_file.close()

    def tearDown(self):
        Path(self.data_file.name).unlink()

    def test_im_end_label_is_not_ignored(self):
        self.assertEqual(
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        dataset = SFTDataset(
            self.data_file.name,
            self.tokenizer,
            max_length=64,
        )
        inputs, labels, loss_mask = dataset[0]

        active_labels = labels[loss_mask.bool()]
        self.assertEqual(
            active_labels.eq(self.tokenizer.eos_token_id).sum().item(),
            2,
        )
        self.assertTrue(torch.all(labels[loss_mask == 0] == -100))

        config = ModelConfig(
            dim=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=len(self.tokenizer),
            hidden_dim=32,
            multiple_of=8,
            max_seq_len=64,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        model = Transformer(config).eval()
        losses = model(inputs.unsqueeze(0), labels.unsqueeze(0)).last_loss
        eos_positions = (labels == self.tokenizer.eos_token_id) & loss_mask.bool()

        self.assertTrue(torch.all(losses[eos_positions] > 0))
        self.assertTrue(torch.all(losses[labels == -100] == 0))


if __name__ == "__main__":
    unittest.main()
