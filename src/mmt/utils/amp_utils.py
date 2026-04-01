"""
Automatic Mixed Precision (AMP) utilities.

Provides:
- `get_amp_config(...)` to choose AMP enablement and dtype based on model device.
- `amp_ctx_for_model(...)` to return an autocast context on CUDA, else a no-op context.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import nullcontext
from typing import cast, Any
from contextlib import contextmanager

import torch
import torch.nn as nn

IS_CUDA_AVAILABLE = torch.cuda.is_available()
if IS_CUDA_AVAILABLE:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch._C import _SDPBackend


# ----------------------------------------------------------------------------------------------------------------------
def get_amp_config(model: nn.Module, enable: bool = True) -> tuple[torch.device, bool, torch.dtype | None]:
    """
    Decide AMP settings based on the model's current device.

    Returns
    -------
    tuple[torch.device, bool, torch.dtype | None]
        Returns (device, amp_enabled, amp_dtype), with each element as follows:
            device : torch.device
                Device inferred from the model (first parameter or model.device).
            amp_enabled : bool
                True if AMP should be used (CUDA + enable=True), else False.
            amp_dtype : torch.dtype | None
                torch.bfloat16 if supported, else torch.float16 when enabled, otherwise None.

    Notes
    -----
    - We only enable AMP on CUDA devices.
    - On CPU/MPS, this returns (device, False, None) and you should treat the context as a no-op.

    """

    # Infer device from model (prefer a custom .device attribute if present)
    dev = getattr(model, "device", next(model.parameters()).device)

    if (dev.type == "cuda") and enable:
        # Prefer bf16 when available, else fp16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dev, True, dtype

    # No AMP on non-CUDA or when disabled
    return dev, False, None


# ----------------------------------------------------------------------------------------------------------------------
def amp_ctx_for_model(model: nn.Module, enable: bool = True) -> Any:
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

    Returns
    -------
    Any
        Either a torch.amp.autocast (if possible) or contextlib.nullcontext instance (fallback).

    """

    dev, amp_enabled, amp_dtype = get_amp_config(model=model, enable=enable)

    if (dev.type == "cuda") and amp_enabled and (amp_dtype is not None):
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)

    # Fallback: do nothing (no AMP)
    return nullcontext()


# ----------------------------------------------------------------------------------------------------------------------
@contextmanager
def sdpa_math_only_ctx() -> Generator:
    """
    Context manager that forces PyTorch's Scaled Dot-Product Attention (SDPA) to use the "math" backend on CUDA
    (disables FlashAttention and memory-efficient SDPA).

    Why:
      - Some CUDA setups can produce NaN/Inf (typically in gradients, sometimes forward) when using Flash or
        memory-efficient SDPA kernels with bf16/AMP and certain masks/shapes.
      - Forcing the math backend is slower but tends to be the most numerically stable.

    Usage:
      Wrap the top-level train/eval call (entrypoints) so it applies to training, validation, and evaluation in a
      single place.

        with sdpa_math_only_ctx():
            train_finetune(...)   # or train_pretrain(...), run_eval(...)

    On CPU (or when CUDA is unavailable), this is a no-op.

    """

    if IS_CUDA_AVAILABLE:
        backend = cast(_SDPBackend, SDPBackend.MATH)
        with sdpa_kernel(backend):
            yield
    else:
        yield
