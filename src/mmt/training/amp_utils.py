from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn


def get_amp_config(
    model: nn.Module,
    enable: bool = True,
) -> Tuple[torch.device, bool, Optional[torch.dtype]]:
    """
    Decide AMP settings based on the model's current device.

    Returns
    -------
    device : torch.device
        Device inferred from the model (first parameter or model.device).
    amp_enabled : bool
        True if AMP should be used (CUDA + enable=True), else False.
    amp_dtype : Optional[torch.dtype]
        torch.bfloat16 if supported, else torch.float16 when enabled,
        otherwise None.

    Notes
    -----
    - We only enable AMP on CUDA devices.
    - On CPU/MPS, this returns (device, False, None) and you should treat
      the context as a no-op.
    """
    # Infer device from model (prefer a custom .device attribute if present)
    dev = getattr(model, "device", next(model.parameters()).device)

    if dev.type == "cuda" and enable:
        # Prefer bf16 when available, else fp16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dev, True, dtype

    # No AMP on non-CUDA or when disabled
    return dev, False, None


def amp_ctx_for_model(
    model: nn.Module,
    enable: bool = True,
):
    """
    Autocast context manager based on the model's current device.

    Usage
    -----
        with amp_ctx_for_model(model, enable=True):
            outputs = model(**batch)

    Behaviour
    ---------
    - CUDA:
        * if enable=True:
            - use torch.bfloat16 if supported, else torch.float16
            - returns a torch.amp.autocast context
        * if enable=False:
            - returns a nullcontext (no AMP)
    - CPU / MPS:
        * always returns a nullcontext (no AMP)
    """
    dev, amp_enabled, amp_dtype = get_amp_config(model, enable=enable)

    if dev.type == "cuda" and amp_enabled and amp_dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)

    # Fallback: do nothing (no AMP)
    return nullcontext()
