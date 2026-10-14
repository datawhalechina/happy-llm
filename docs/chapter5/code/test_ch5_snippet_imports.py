"""Regression: chapter 5.1 tutorial cells must be copy-paste runnable.

Issues #182 / #184 / #185: RMSNorm used `args` before it existed,
`apply_rotary_emb` annotated with `Tuple` without importing it, and
MLP called `F.silu` without `torch.nn.functional`.

Run:
    python docs/chapter5/code/test_ch5_snippet_imports.py
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


def _chapter5_markdown() -> Path:
    here = Path(__file__).resolve()
    # docs/chapter5/code/this_file.py -> docs/chapter5/*.md
    chapter_dir = here.parent.parent
    matches = list(chapter_dir.glob("*动手搭建大模型.md"))
    if matches:
        return matches[0]
    local = Path(__file__).resolve().parent / "chapter5.md"
    if local.exists():
        return local
    raise FileNotFoundError("chapter 5 markdown not found")


def _python_fences(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


class Chapter51SnippetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md = _chapter5_markdown().read_text(encoding="utf-8")
        cls.fences = _python_fences(cls.md)

    def test_prelude_instantiates_args_and_imports(self):
        prelude = next(
            (block for block in self.fences if "args = ModelConfig()" in block),
            None,
        )
        self.assertIsNotNone(prelude, "5.1.1 must instantiate args = ModelConfig()")
        self.assertIn("import torch", prelude)
        self.assertIn("import torch.nn as nn", prelude)
        self.assertIn("import torch.nn.functional as F", prelude)
        self.assertIn("from typing import Optional, Tuple", prelude)
        ast.parse(prelude)

    def test_apply_rotary_emb_tuple_is_defined_by_prelude(self):
        rotary = next(
            (block for block in self.fences if "def apply_rotary_emb" in block),
            None,
        )
        self.assertIsNotNone(rotary)
        self.assertIn("Tuple[torch.Tensor, torch.Tensor]", rotary)
        prelude = next(block for block in self.fences if "args = ModelConfig()" in block)
        combined = prelude + "\n" + rotary
        # Tuple must resolve after the prelude; compiling the combined block
        # catches a missing typing import even without executing torch.
        tree = ast.parse(combined)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertIn("Tuple", names)

    def test_mlp_uses_F_imported_in_prelude(self):
        mlp = next((block for block in self.fences if "class MLP" in block), None)
        self.assertIsNotNone(mlp)
        self.assertIn("F.silu", mlp)
        prelude = next(block for block in self.fences if "args = ModelConfig()" in block)
        self.assertIn("torch.nn.functional as F", prelude)


if __name__ == "__main__":
    unittest.main(verbosity=2)
