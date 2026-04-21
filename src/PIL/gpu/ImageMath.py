"""
PIL.gpu.ImageMath — GPU-side mathematical image operations.

Delegates to PIL.ImageMath with CPU fallback.
"""

from __future__ import annotations

from PIL import ImageMath as _CPUMath

from . import Image as _GpuImage


def eval(expression: str, **kw: object) -> object:
    """Evaluate an image expression.

    Works like :func:`PIL.ImageMath.eval`, but accepts GPU images.
    Images are downloaded to CPU, evaluated, and the result is uploaded.
    """
    cpu_kw: dict[str, object] = {}
    for k, v in kw.items():
        if isinstance(v, _GpuImage.Image):
            cpu_kw[k] = v.to_cpu()
        else:
            cpu_kw[k] = v

    result = _CPUMath.eval(expression, **cpu_kw)  # type: ignore[attr-defined]

    if hasattr(result, "im"):
        # PIL ImageMath returns an _Operand; extract the PIL Image
        from PIL import Image as PILImage

        if isinstance(result, PILImage.Image):
            return _GpuImage.Image.from_cpu(result)
        # _Operand case
        cpu_result = result.im
        return _GpuImage.Image.from_cpu(cpu_result)
    return result


def unsafe_eval(expression: str, **kw: object) -> object:
    """Evaluate expression (unsafe mode). See :func:`PIL.ImageMath.unsafe_eval`."""
    cpu_kw2: dict[str, object] = {}
    for k, v in kw.items():
        if isinstance(v, _GpuImage.Image):
            cpu_kw2[k] = v.to_cpu()
        else:
            cpu_kw2[k] = v

    result = _CPUMath.unsafe_eval(expression, **cpu_kw2)

    if hasattr(result, "im"):
        from PIL import Image as PILImage

        if isinstance(result, PILImage.Image):
            return _GpuImage.Image.from_cpu(result)
        cpu_result = result.im
        return _GpuImage.Image.from_cpu(cpu_result)
    return result
