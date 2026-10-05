"""Opt-in Hopper numerical modes; native remains the package default.

The resolved choice is an autograd input, not a backward-time global lookup.
A package default persists across notebook cells and execution contexts. Scoped
overrides let the pinned Transformers adapter select a mode even though it
filters custom FlashAttention keywords.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import FakeTensor

MODES = ("native", "sbs", "sbs_gproj")
KEY_PIPELINES = ("single_stage", "two_stage")
_PACKAGE_DEFAULT = ("native", "two_stage")
_OVERRIDE = ContextVar("fa4_numerics_override", default=None)


@lru_cache
def _cuda_arch(device):
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def resolve_numerics(mode=None, key_pipeline=None):
    default_mode, default_pipeline = _OVERRIDE.get() or _PACKAGE_DEFAULT
    mode = default_mode if mode is None else mode
    key_pipeline = default_pipeline if key_pipeline is None else key_pipeline
    if mode not in MODES:
        raise ValueError(f"Unknown FA4 numerical mode {mode!r}; expected {MODES}")
    if key_pipeline not in KEY_PIPELINES:
        raise ValueError(f"Unknown FA4 key pipeline {key_pipeline!r}; expected {KEY_PIPELINES}")
    return mode, key_pipeline


def get_default_numerics():
    return _PACKAGE_DEFAULT


def set_default_numerics(mode, *, key_pipeline="two_stage"):
    """Set the package default across notebook cells; return the previous pair."""
    global _PACKAGE_DEFAULT
    resolved = resolve_numerics(mode, key_pipeline)
    previous = _PACKAGE_DEFAULT
    _PACKAGE_DEFAULT = resolved
    return previous


@contextmanager
def numerical_mode(mode, *, key_pipeline=None):
    token = _OVERRIDE.set(resolve_numerics(mode, key_pipeline))
    try:
        yield
    finally:
        _OVERRIDE.reset(token)


def validate_numerics(mode, q, k, v, *, arch, causal, softmax_scale,
                      varlen=False, num_splits=1, pack_gqa=None,
                      deterministic=False, window_size=(None, None), **options):
    """Fail before autograd construction for unsupported opt-in requests."""
    if mode == "native":
        return
    failures = []
    if varlen:
        failures.append("padding/packed/varlen sequences")
    if (arch != 90 or q is not None and q.is_cuda
            and not isinstance(q, FakeTensor) and _cuda_arch(q.device) != 90):
        failures.append("architecture other than Hopper SM90")
    if q is None or k is None or v is None:
        failures.append("MLA/shared-KV routes")
    else:
        n = q.shape[1] if q.ndim == 4 else 0
        if (q.ndim != 4 or n <= 0 or tuple(q.shape) != (1, n, 16, 256)
                or tuple(k.shape) != (1, n, 4, 256)
                or tuple(v.shape) != (1, n, 4, 256)):
            failures.append("geometry other than batch-one positive equal-length d256 16:4 GQA")
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            failures.append("attention dtype other than BF16")
        if any(t.device != q.device for t in (k, v)):
            failures.append("mixed devices")
    if not causal:
        failures.append("noncausal attention")
    if softmax_scale is not None and softmax_scale != 1 / 16:
        failures.append("scale other than 1/16")
    if num_splits != 1:
        failures.append("split forward/automatic splitting")
    if pack_gqa is False:
        failures.append("disabled forward GQA head packing")
    if not isinstance(deterministic, bool):
        failures.append("non-boolean deterministic flag")
    if tuple(window_size) != (None, None):
        failures.append("local/window attention")
    for name, value in options.items():
        disabled = (value is None or value is False
                    or isinstance(value, (int, float)) and value == 0
                    or isinstance(value, (tuple, list)) and len(value) == 0)
        if not disabled:
            failures.append(name)
    if failures:
        raise ValueError(f"FA4 {mode} does not support: " + "; ".join(failures))
