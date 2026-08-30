# -*- coding: utf-8 -*-
import pathlib
import sys
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch


CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from amp_utils import configure_amp


class ConfigureAmpTest(unittest.TestCase):
    def test_cpu_disables_autocast_and_scaler_for_every_dtype(self):
        for dtype in ("float32", "bfloat16", "float16"):
            with self.subTest(dtype=dtype), patch(
                "amp_utils.torch.amp.autocast"
            ) as autocast, patch(
                "amp_utils.torch.amp.GradScaler"
            ) as grad_scaler:
                ctx, _ = configure_amp("cpu", dtype)

                self.assertIsInstance(ctx, nullcontext)
                autocast.assert_not_called()
                grad_scaler.assert_called_once_with("cuda", enabled=False)

    def test_cuda_uses_requested_low_precision_dtype(self):
        cases = (
            ("bfloat16", torch.bfloat16, False),
            ("float16", torch.float16, True),
        )
        for dtype, torch_dtype, scaler_enabled in cases:
            with self.subTest(dtype=dtype), patch(
                "amp_utils.torch.amp.autocast"
            ) as autocast, patch(
                "amp_utils.torch.amp.GradScaler"
            ) as grad_scaler:
                configure_amp("cuda", dtype)

                autocast.assert_called_once_with(
                    device_type="cuda",
                    dtype=torch_dtype,
                )
                grad_scaler.assert_called_once_with(
                    "cuda",
                    enabled=scaler_enabled,
                )

    def test_cuda_float32_disables_autocast_and_scaler(self):
        with patch("amp_utils.torch.amp.autocast") as autocast, patch(
            "amp_utils.torch.amp.GradScaler"
        ) as grad_scaler:
            ctx, _ = configure_amp("cuda", "float32")

            self.assertIsInstance(ctx, nullcontext)
            autocast.assert_not_called()
            grad_scaler.assert_called_once_with("cuda", enabled=False)

    def test_rejects_unknown_dtype(self):
        with self.assertRaisesRegex(ValueError, "不支持的数据类型"):
            configure_amp("cuda", "float64")


if __name__ == "__main__":
    unittest.main()
