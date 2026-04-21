"""
PIL.gpu.ImageSequence — Iterate over image frames.

Provides the same Iterator interface as PIL.ImageSequence, but wraps
frames as GPU images.
"""

from __future__ import annotations

from . import Image as _GpuImage


class Iterator:
    """Iterate over frames of a multi-frame image.

    Accepts either a GPU Image (single-frame) or a file path.
    For multi-frame files (GIF, TIFF, etc.), frames are loaded from
    the CPU Image and uploaded one at a time.
    """

    def __init__(self, im: _GpuImage.Image | str | object) -> None:
        if isinstance(im, _GpuImage.Image):
            # Single GPU image — wrap as single-frame sequence
            self._frames = [im]
        elif isinstance(im, str):
            # File path — load all frames from CPU
            from PIL import Image as PILImage

            cpu = PILImage.open(im)
            self._frames = []
            try:
                while True:
                    cpu.seek(cpu.tell())
                    self._frames.append(_GpuImage.Image.from_cpu(cpu.copy()))
                    cpu.seek(cpu.tell() + 1)
            except EOFError:
                pass
        else:
            # Assume it's a PIL Image with seek()
            self._frames = []
            try:
                while True:
                    im.seek(im.tell())  # type: ignore[attr-defined]
                    self._frames.append(_GpuImage.Image.from_cpu(im.copy()))  # type: ignore[attr-defined]
                    im.seek(im.tell() + 1)  # type: ignore[attr-defined]
            except EOFError:
                pass
        self._pos = 0

    def __iter__(self) -> Iterator:
        self._pos = 0
        return self

    def __next__(self) -> _GpuImage.Image:
        if self._pos >= len(self._frames):
            raise StopIteration
        frame = self._frames[self._pos]
        self._pos += 1
        return frame
