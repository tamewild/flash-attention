"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flash-attn-4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]

from .numerics import get_default_numerics, set_default_numerics, numerical_mode

__all__ += ["get_default_numerics", "set_default_numerics", "numerical_mode"]

# A rewritten installation needs a fresh kernel, including Transformers discovery
# and compiled source caches. The setup audit checks this import-time stamp.
import hashlib as _hashlib
from pathlib import Path as _Path
_manifest_path = _Path(__file__).with_name("_pcss_patch_manifest.json")
_pcss_manifest_stamp = (
    _hashlib.sha256(_manifest_path.read_bytes()).hexdigest() if _manifest_path.exists() else None
)
