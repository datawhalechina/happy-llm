import ast
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "docs" / "chapter5" / "code" / "ddp_sft_full.py"
GUIDE = ROOT / "docs" / "chapter5" / "第五章 动手搭建大模型.md"


def _load_masked_loss():
    tree = ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "masked_loss")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAIN_SCRIPT), "exec"), namespace)
    return namespace["masked_loss"]


class ZeroMaskLossRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.masked_loss = staticmethod(_load_masked_loss())

    def test_zero_supervision_masks_stay_finite_and_keep_graph(self):
        for sample_name in ("user-only", "truncated-assistant"):
            with self.subTest(sample_name=sample_name):
                token_loss = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
                zero_mask = torch.zeros(3, dtype=torch.long)
                baseline = torch.sum(token_loss * zero_mask) / zero_mask.sum()
                self.assertTrue(torch.isnan(baseline))
                loss = self.masked_loss(token_loss, zero_mask)
                self.assertEqual(loss.item(), 0.0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(token_loss.grad).all())
                self.assertTrue(torch.equal(token_loss.grad, torch.zeros_like(token_loss.grad)))

    def test_normal_assistant_mask_keeps_mean_loss(self):
        token_loss = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        loss = self.masked_loss(token_loss, torch.tensor([1, 0, 1]))
        self.assertAlmostEqual(loss.item(), 2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(token_loss.grad).all())

    def test_optimizer_step_does_not_receive_nan_from_empty_batch(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        token_loss = parameter * torch.tensor([1.0, 2.0, 3.0])
        loss = self.masked_loss(token_loss, torch.zeros(3, dtype=torch.long))
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())

    def test_training_paths_use_the_guard(self):
        script = TRAIN_SCRIPT.read_text(encoding="utf-8")
        guide = GUIDE.read_text(encoding="utf-8")
        self.assertIn("loss = masked_loss(loss, loss_mask)", script)
        self.assertNotIn("torch.sum(loss * loss_mask) / loss_mask.sum()", script)
        self.assertEqual(guide.count("loss = masked_loss(loss, loss_mask)"), 2)
        self.assertNotIn("torch.sum(loss * loss_mask) / loss_mask.sum()", guide)


if __name__ == "__main__":
    unittest.main()
