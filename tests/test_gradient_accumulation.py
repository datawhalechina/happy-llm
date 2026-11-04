import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "docs" / "chapter5" / "code"
sys.path.insert(0, str(CODE))

from dataset import PretrainDataset, SFTDataset
from k_model import ModelConfig, Transformer


def load_training_script(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Experiment tracking is disabled. Import the real loop without requiring
    # the optional hosted logger or downloading pretrained weights.
    with patch.dict(sys.modules, {"swanlab": ModuleType("swanlab")}):
        spec.loader.exec_module(module)
    return module


class GradientAccumulationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(CODE / "tokenizer_k", local_files_only=True)
        cls.scripts = [load_training_script(name) for name in ("ddp_pretrain", "ddp_sft_full")]

    def check_training(self, batch_count, accumulation_steps, epochs, optimizer_class=torch.optim.SGD):
        for script in self.scripts:
            with self.subTest(script=script.__name__, batches=batch_count), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "train.jsonl"
                if script.__name__ == "ddp_sft_full":
                    sample = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
                    dataset_class = SFTDataset
                else:
                    sample = {"text": "a b c"}
                    dataset_class = PretrainDataset
                path.write_text((json.dumps(sample) + "\n") * batch_count, encoding="utf-8")
                loader = DataLoader(dataset_class(path, self.tokenizer, max_length=32), batch_size=1)
                torch.manual_seed(42)
                config = ModelConfig(dim=16, n_layers=1, n_heads=2, n_kv_heads=1,
                                     multiple_of=8, max_seq_len=32, vocab_size=len(self.tokenizer))
                config.pad_token_id = self.tokenizer.pad_token_id
                script.model = Transformer(config)
                reference = deepcopy(script.model)
                script.optimizer = optimizer_class(script.model.parameters(), lr=0.01)
                expected_optimizer = optimizer_class(reference.parameters(), lr=0.01)
                script.train_loader = loader
                script.iter_per_epoch = len(loader)
                script.ctx = nullcontext()
                script.scaler = torch.amp.GradScaler("cpu", enabled=False)
                script.args = SimpleNamespace(device="cpu", epochs=epochs, accumulation_steps=accumulation_steps,
                                              grad_clip=1000.0, log_interval=1000, save_interval=1000,
                                              use_swanlab=False)
                script.get_lr = lambda *args: 0.01
                script.Logger = lambda message: None
                reference_batches = list(loader)
                actual_steps = []
                hook = script.optimizer.register_step_post_hook(lambda *args: actual_steps.append(1))
                try:
                    for epoch in range(epochs):
                        script.train_epoch(epoch)
                        for start in range(0, len(reference_batches), accumulation_steps):
                            group = reference_batches[start:start + accumulation_steps]
                            expected_optimizer.zero_grad(set_to_none=True)
                            for inputs, targets, mask in group:
                                losses = reference(inputs, targets).last_loss
                                loss = (losses * mask.view(-1)).sum() / mask.sum()
                                (loss / len(group)).backward()
                            torch.nn.utils.clip_grad_norm_(reference.parameters(), script.args.grad_clip)
                            expected_optimizer.step()
                        expected_steps = (epoch + 1) * ((batch_count + accumulation_steps - 1) // accumulation_steps)
                        self.assertEqual(len(actual_steps), expected_steps)
                        self.assertTrue(all(parameter.grad is None for parameter in script.model.parameters()))
                        for actual, expected in zip(script.model.parameters(), reference.parameters()):
                            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                finally:
                    hook.remove()

    def test_short_epoch_updates_once(self):
        self.check_training(batch_count=3, accumulation_steps=4, epochs=1)

    def test_partial_group_does_not_leak_into_next_epoch(self):
        self.check_training(batch_count=6, accumulation_steps=4, epochs=2)

    def test_complete_groups_keep_existing_behavior(self):
        self.check_training(batch_count=4, accumulation_steps=2, epochs=1)

    def test_no_accumulation_keeps_existing_behavior(self):
        self.check_training(batch_count=3, accumulation_steps=1, epochs=1)

    def test_adamw_partial_groups_match_reference(self):
        self.check_training(batch_count=6, accumulation_steps=4, epochs=2, optimizer_class=torch.optim.AdamW)


if __name__ == "__main__":
    unittest.main()
