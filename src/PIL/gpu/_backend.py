"""
PIL.gpu._backend — Backend lifecycle management.

Handles lazy initialization and shutdown of the GPU compute backend.
"""

from __future__ import annotations

import atexit

try:
    from PIL import _imaging_gpu as _core
except ImportError:
    _core = None

_initialized = False


def _ensure_backend() -> None:
    """Initialize the GPU backend on first use (lazy)."""
    global _initialized
    if _initialized:
        return
    if _core is None:
        msg = (
            "PIL._imaging_gpu is not available.  "
            "Pillow was built without GPU support (OpenCL/CUDA not found)."
        )
        raise RuntimeError(msg)
    _core.backend_init(0)  # 0 = auto-select
    _initialized = True
    atexit.register(_shutdown)


def _shutdown() -> None:
    """Release GPU resources."""
    global _initialized
    if _initialized and _core is not None:
        _core.backend_shutdown()
        _initialized = False


def get_backend_name() -> str | None:
    """Return the active backend name ('OpenCL' or 'CUDA'), or None."""
    _ensure_backend()
    return _core.get_backend_name()


def get_device_name() -> str | None:
    """Return the GPU device name, or None."""
    _ensure_backend()
    return _core.get_device_name()


# Constants re-exported from C module for convenience
BACKEND_NONE = 0
BACKEND_OPENCL = 1
BACKEND_CUDA = 2
