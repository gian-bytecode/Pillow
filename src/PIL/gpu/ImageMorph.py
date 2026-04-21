"""
PIL.gpu.ImageMorph — GPU-side morphological operations.

Delegates to PIL.ImageMorph with CPU fallback.
"""

from __future__ import annotations

from PIL import ImageMorph as _CPUMorph

from . import Image as _GpuImage


class LutBuilder(_CPUMorph.LutBuilder):
    """LUT builder — same as PIL.ImageMorph.LutBuilder."""
    pass


class MorphOp:
    """Morphological operations on GPU images.

    Downloads to CPU, applies the operation, and re-uploads.
    """

    def __init__(self, lut=None, op_name=None, patterns=None):
        self._cpu_op = _CPUMorph.MorphOp(
            lut=lut, op_name=op_name, patterns=patterns
        )

    def apply(self, image: "_GpuImage.Image"):
        """Apply the morphological operation. Returns (count, image)."""
        cpu = image.to_cpu()
        count, result = self._cpu_op.apply(cpu)
        return count, _GpuImage.Image.from_cpu(result)

    def match(self, image: "_GpuImage.Image"):
        """Get list of matching pixel coordinates."""
        cpu = image.to_cpu()
        return self._cpu_op.match(cpu)

    def get_on_pixels(self, image: "_GpuImage.Image"):
        """Get list of non-zero pixel coordinates."""
        cpu = image.to_cpu()
        return self._cpu_op.get_on_pixels(cpu)
