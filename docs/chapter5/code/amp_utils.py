# -*- coding: utf-8 -*-
from contextlib import nullcontext

import torch


TORCH_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def configure_amp(device_type, dtype):
    """根据设备和精度配置 autocast 上下文与梯度缩放器。"""
    if dtype not in TORCH_DTYPES:
        supported = ", ".join(TORCH_DTYPES)
        raise ValueError(f"不支持的数据类型 {dtype!r}，可选值为：{supported}")

    if device_type == "cuda" and dtype != "float32":
        ctx = torch.amp.autocast(device_type="cuda", dtype=TORCH_DTYPES[dtype])
    else:
        ctx = nullcontext()

    # BF16 与 FP32 具有相同的指数位宽，通常不需要梯度缩放。
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device_type == "cuda" and dtype == "float16"),
    )
    return ctx, scaler
