"""
PIL.gpu.ImageDraw — GPU-side drawing operations.

Provides a Draw object mirroring PIL.ImageDraw.  For operations that
don't have a native GPU kernel, we download to CPU, draw, and re-upload.
"""

from __future__ import annotations

from PIL import ImageDraw as _CPUDraw

from . import Image as _GpuImage


class Draw:
    """Draw object wrapping a GPU Image.

    Drawing is done by downloading to CPU, drawing with PIL.ImageDraw,
    and re-uploading.  Future versions may add GPU-native kernels.
    """

    def __init__(self, im: _GpuImage.Image, mode: str | None = None):
        self._gpu_im = im
        self._mode = mode

    def _cpu_draw(self):
        """Create a CPU Image + Draw pair."""
        cpu = self._gpu_im.to_cpu()
        draw = _CPUDraw.Draw(cpu, mode=self._mode)
        return cpu, draw

    def _update(self, cpu):
        """Re-upload modified CPU image."""
        new_gpu = _GpuImage.Image.from_cpu(cpu)
        self._gpu_im._im = new_gpu._im

    def line(self, xy, fill=None, width=0, joint=None):
        cpu, draw = self._cpu_draw()
        draw.line(xy, fill=fill, width=width, joint=joint)
        self._update(cpu)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        cpu, draw = self._cpu_draw()
        draw.rectangle(xy, fill=fill, outline=outline, width=width)
        self._update(cpu)

    def ellipse(self, xy, fill=None, outline=None, width=1):
        cpu, draw = self._cpu_draw()
        draw.ellipse(xy, fill=fill, outline=outline, width=width)
        self._update(cpu)

    def polygon(self, xy, fill=None, outline=None, width=1):
        cpu, draw = self._cpu_draw()
        draw.polygon(xy, fill=fill, outline=outline, width=width)
        self._update(cpu)

    def arc(self, xy, start, end, fill=None, width=1):
        cpu, draw = self._cpu_draw()
        draw.arc(xy, start, end, fill=fill, width=width)
        self._update(cpu)

    def chord(self, xy, start, end, fill=None, outline=None, width=1):
        cpu, draw = self._cpu_draw()
        draw.chord(xy, start, end, fill=fill, outline=outline, width=width)
        self._update(cpu)

    def pieslice(self, xy, start, end, fill=None, outline=None, width=1):
        cpu, draw = self._cpu_draw()
        draw.pieslice(xy, start, end, fill=fill, outline=outline, width=width)
        self._update(cpu)

    def point(self, xy, fill=None):
        cpu, draw = self._cpu_draw()
        draw.point(xy, fill=fill)
        self._update(cpu)

    def text(
        self,
        xy,
        text,
        fill=None,
        font=None,
        anchor=None,
        spacing=4,
        align="left",
        direction=None,
        features=None,
        language=None,
        stroke_width=0,
        stroke_fill=None,
        embedded_color=False,
    ):
        cpu, draw = self._cpu_draw()
        draw.text(
            xy,
            text,
            fill=fill,
            font=font,
            anchor=anchor,
            spacing=spacing,
            align=align,
            direction=direction,
            features=features,
            language=language,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            embedded_color=embedded_color,
        )
        self._update(cpu)

    def multiline_text(
        self,
        xy,
        text,
        fill=None,
        font=None,
        anchor=None,
        spacing=4,
        align="left",
        direction=None,
        features=None,
        language=None,
        stroke_width=0,
        stroke_fill=None,
        embedded_color=False,
    ):
        cpu, draw = self._cpu_draw()
        draw.multiline_text(
            xy,
            text,
            fill=fill,
            font=font,
            anchor=anchor,
            spacing=spacing,
            align=align,
            direction=direction,
            features=features,
            language=language,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            embedded_color=embedded_color,
        )
        self._update(cpu)

    def textbbox(
        self,
        xy,
        text,
        font=None,
        anchor=None,
        spacing=4,
        align="left",
        direction=None,
        features=None,
        language=None,
        stroke_width=0,
        embedded_color=False,
    ):
        cpu, draw = self._cpu_draw()
        return draw.textbbox(
            xy,
            text,
            font=font,
            anchor=anchor,
            spacing=spacing,
            align=align,
            direction=direction,
            features=features,
            language=language,
            stroke_width=stroke_width,
            embedded_color=embedded_color,
        )

    def textlength(
        self,
        text,
        font=None,
        direction=None,
        features=None,
        language=None,
        embedded_color=False,
    ):
        cpu, draw = self._cpu_draw()
        return draw.textlength(
            text,
            font=font,
            direction=direction,
            features=features,
            language=language,
            embedded_color=embedded_color,
        )
