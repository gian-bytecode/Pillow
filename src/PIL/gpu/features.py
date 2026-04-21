"""PIL.gpu.features — Re-export of PIL.features plus GPU feature checks."""

from __future__ import annotations

from PIL.features import *  # noqa: F403

from ._backend import get_backend_name, get_device_name

__all__ = ["check_gpu", "get_gpu_backend", "get_gpu_device"]


def check_gpu() -> bool:
    """Return True if GPU acceleration is available."""
    try:
        from PIL.gpu import is_available

        return is_available()
    except Exception:
        return False


def get_gpu_backend():
    """Return the active GPU backend name, or None."""
    try:
        return get_backend_name()
    except Exception:
        return None


def get_gpu_device():
    """Return the GPU device name, or None."""
    try:
        return get_device_name()
    except Exception:
        return None
