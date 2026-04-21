"""
PIL.gpu.Image — GPU-resident image class mirroring PIL.Image.

Operations keep data in VRAM.  Transfer to/from CPU is explicit via
``from_cpu()`` / ``to_cpu()`` or implicit in ``open()`` / ``save()``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image as _CPUImage

from . import _core
from ._backend import _ensure_backend

# Re-export enums so ``from PIL.gpu.Image import Resampling`` works
Resampling = _CPUImage.Resampling
Transpose = _CPUImage.Transpose


class Image:
    """A GPU-resident image.  Wraps a C ``ImagingGPU`` handle."""

    __slots__ = ("_im",)

    def __init__(self, gpu_core_image: Any) -> None:
        self._im: Any = gpu_core_image

    # ------------------------------------------------------------------ #
    # Factories                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def new(
        mode: str, size: tuple[int, int], color: int | tuple[int, ...] = 0
    ) -> Image:
        """Create a new GPU image filled with *color*."""
        _ensure_backend()
        if isinstance(color, int):
            color = (color,) * 4
        elif isinstance(color, (tuple, list)):
            color = tuple(color) + (0,) * (4 - len(color))
        c = _core.fill(mode, size, color[:4])
        return Image(c)

    @staticmethod
    def from_cpu(cpu_image: _CPUImage.Image) -> Image:
        """Upload a :class:`PIL.Image.Image` to the GPU."""
        _ensure_backend()
        cpu_image.load()
        # Use raw bytes transfer (avoids struct layout mismatches with
        # binary-installed Pillow).
        # Pillow stores RGB as RGBX internally (4 bytes/pixel), matching
        # GPU internal storage.  Use the band rawmode for the transfer.
        mode = cpu_image.mode
        rawmode = mode
        if mode == "RGB":
            rawmode = "RGBX"
        elif mode == "LA":
            rawmode = "LA"
        elif mode == "YCbCr":
            rawmode = "YCbCrX"

        raw = cpu_image.tobytes("raw", rawmode)
        gpu_im = _core.from_bytes(mode, cpu_image.size, raw)
        return Image(gpu_im)

    @staticmethod
    def open(
        fp: str | object,
        mode: str = "r",
        formats: list[str] | tuple[str, ...] | None = None,
    ) -> Image:
        """Open an image file and upload it to the GPU.

        Accepts the same arguments as :func:`PIL.Image.open`.
        """
        cpu = _CPUImage.open(fp, mode="r", formats=formats)  # type: ignore[arg-type]
        cpu.load()  # ensure pixels are decoded
        return Image.from_cpu(cpu)

    # ------------------------------------------------------------------ #
    # Transfer back to CPU                                               #
    # ------------------------------------------------------------------ #

    def to_cpu(self) -> _CPUImage.Image:
        """Download GPU image to a new :class:`PIL.Image.Image`."""
        data = self._im.to_bytes()
        mode = self.mode
        rawmode = mode
        if mode == "RGB":
            rawmode = "RGBX"
        elif mode == "YCbCr":
            rawmode = "YCbCrX"
        return _CPUImage.frombytes(mode, self.size, data, "raw", rawmode)

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def mode(self) -> str:
        return self._im.mode

    @property
    def size(self) -> tuple[int, int]:
        return self._im.size

    @property
    def width(self) -> int:
        return self._im.size[0]

    @property
    def height(self) -> int:
        return self._im.size[1]

    @property
    def bands(self) -> int:
        return self._im.bands

    @property
    def backend(self) -> str:
        return self._im.backend

    # ------------------------------------------------------------------ #
    # Image operations (return new GPU Image)                            #
    # ------------------------------------------------------------------ #

    def copy(self) -> Image:
        return Image(self._im.copy())

    def convert(self, mode: str) -> Image:
        if mode == self.mode:
            return self.copy()
        return Image(self._im.convert(mode))

    def resize(
        self,
        size: tuple[int, int],
        resample: int = Resampling.BILINEAR,
        box: tuple[float, float, float, float] | None = None,
    ) -> Image:
        if box is None:
            box = (0.0, 0.0, float(self.width), float(self.height))
        return Image(self._im.resize(size, resample, box))

    def transpose(self, method: int) -> Image:
        return Image(self._im.transpose(method))

    def transform(
        self,
        size: tuple[int, int],
        method: int,
        data: tuple[float, ...] | list[float] | None = None,
        resample: int = Resampling.NEAREST,
        fill: int = 1,
    ) -> Image:
        if data is None:
            data = (1, 0, 0, 0, 1, 0, 0, 0)
        coeffs = tuple(float(x) for x in data) + (0.0,) * (8 - len(data))
        return Image(self._im.transform(size, method, coeffs, resample, fill))

    def filter(self, kernel: object) -> Image:
        """Apply an image filter.

        *kernel* can be a :class:`PIL.ImageFilter.Kernel`,
        a :class:`~PIL.gpu.ImageFilter.RankFilter`,
        a :class:`~PIL.gpu.ImageFilter.ModeFilter`,
        or any GPU filter with a ``filter(image)`` method.
        """
        # Convolution kernels (Kernel instances and BuiltinFilter classes)
        # have .filterargs = ((kx, ky), scale, offset, sequence)
        if hasattr(kernel, "filterargs"):
            fa = kernel.filterargs
            kx, ky = fa[0]
            scale_val = float(fa[1])
            offset_val = float(fa[2])
            seq = fa[3]
            return Image(self._im.filter(kx, ky, seq, scale_val, offset_val))

        # Dispatch filter objects with .filter() method
        # (RankFilter, ModeFilter, GaussianBlur, BoxBlur, UnsharpMask, etc.)
        if hasattr(kernel, "filter") and callable(kernel.filter):
            return kernel.filter(self)

        msg = f"unsupported filter type: {type(kernel)}"
        raise TypeError(msg)

    def point(
        self,
        lut: list[int] | bytes | Callable[[int], int],
        mode: str | None = None,
    ) -> Image:
        """Apply a point transform.

        If *lut* is a callable, it's called for each value 0-255.
        If *lut* is a sequence, it's used directly as a lookup table.
        """
        if callable(lut):
            lut = [lut(i) for i in range(256 * self.bands)]
        lut_bytes = bytes(int(v) & 0xFF for v in lut)
        return Image(self._im.point_lut(lut_bytes, self.bands))

    def point_transform(self, scale: float, offset: float) -> Image:
        """Apply ``pixel = pixel * scale + offset`` to all channels."""
        return Image(self._im.point_transform(scale, offset))

    # ------------------------------------------------------------------ #
    # Blur / sharpen                                                     #
    # ------------------------------------------------------------------ #

    def gaussian_blur(self, radius: float, passes: int = 3) -> Image:
        return Image(self._im.gaussian_blur(float(radius), float(radius), passes))

    def box_blur(self, radius: float, n: int = 1) -> Image:
        return Image(self._im.box_blur(float(radius), float(radius), n))

    def unsharp_mask(self, radius: float, percent: int, threshold: int) -> Image:
        return Image(self._im.unsharp_mask(float(radius), percent, threshold))

    # ------------------------------------------------------------------ #
    # Channel operations                                                 #
    # ------------------------------------------------------------------ #

    def getchannel(self, channel: int | str) -> Image:
        if isinstance(channel, str):
            bands = _CPUImage.getmodebands(self.mode)
            channel = list("RGBA"[:bands]).index(channel)
        return Image(self._im.getband(channel))

    def putchannel(self, channel_im: Image, channel: int) -> Image:
        return Image(self._im.putband(channel_im._im, channel))

    def split(self) -> tuple[Image, ...]:
        """Split into individual band images."""
        parts = self._im.split()
        return tuple(Image(p) for p in parts)

    # ------------------------------------------------------------------ #
    # Crop / Expand / Paste / Offset                                     #
    # ------------------------------------------------------------------ #

    def crop(self, box: tuple[int, int, int, int]) -> Image:
        """Extract a rectangular region."""
        return Image(self._im.crop(box))

    def paste(
        self,
        im: Image,
        box: tuple[int, int] | tuple[int, int, int, int] | int | None = None,
        mask: Image | None = None,
    ) -> None:
        """Paste another image onto this one (in-place)."""
        dx, dy = 0, 0
        if box is not None:
            if isinstance(box, (tuple, list)) and len(box) >= 2:
                dx, dy = box[0], box[1]
            else:
                dx, dy = box, 0
        mask_im = mask._im if mask is not None else None
        self._im.paste(im._im, (dx, dy), mask_im)

    def expand(self, border: int, fill: int | tuple[int, ...] = 0) -> Image:
        """Add a border around the image."""
        if isinstance(fill, int):
            fill = (fill,) * 4
        return Image(self._im.expand(border, border, *fill[:4]))

    def offset(self, xoffset: int, yoffset: int = 0) -> Image:
        """Circular-shift the image."""
        return Image(self._im.offset(xoffset, yoffset))

    # ------------------------------------------------------------------ #
    # Statistics / Analysis                                              #
    # ------------------------------------------------------------------ #

    def getbbox(self, alpha_only: bool = False) -> tuple[int, int, int, int] | None:
        """Return bounding box of non-zero pixels."""
        return self._im.getbbox(int(alpha_only))

    def getextrema(self) -> tuple[tuple[int, int], ...] | tuple[int, int]:
        """Return min/max values per band."""
        return self._im.getextrema()

    # ------------------------------------------------------------------ #
    # Effects                                                            #
    # ------------------------------------------------------------------ #

    def effect_spread(self, distance: int) -> Image:
        """Randomly displace pixels."""
        return Image(self._im.effect_spread(distance))

    def reduce(self, factor: int | tuple[int, int]) -> Image:
        """Reduce image by integer factor (box averaging).
        factor can be an int or (factor_x, factor_y) tuple."""
        if isinstance(factor, int):
            fx, fy = factor, factor
        else:
            fx, fy = factor
        return Image(self._im.reduce(fx, fy))

    def rank_filter(self, size: int, rank: int) -> Image:
        """Rank filter: select the rank-th value in a size x size window.
        rank=0 for min, rank=size*size//2 for median, rank=size*size-1 for max."""
        return Image(self._im.rank_filter(size, rank))

    def mode_filter(self, size: int = 3) -> Image:
        """Mode filter: most frequent pixel value in size x size window (L only)."""
        return Image(self._im.mode_filter(size))

    # ------------------------------------------------------------------ #
    # Image processing (posterize, solarize, negative, equalize)         #
    # ------------------------------------------------------------------ #

    def negative(self) -> Image:
        """Negate image (invert all channels)."""
        return Image(self._im.negative())

    def posterize(self, bits: int) -> Image:
        """Reduce number of bits per channel."""
        return Image(self._im.posterize(bits))

    def solarize(self, threshold: int = 128) -> Image:
        """Solarize: invert values above threshold."""
        return Image(self._im.solarize(threshold))

    def _equalize_with_lut(self, lut: bytes) -> Image:
        """Apply a pre-computed equalization LUT."""
        return Image(self._im.equalize(lut))

    # ------------------------------------------------------------------ #
    # Compositing                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def blend(im1: Image, im2: Image, alpha: float) -> Image:
        _ensure_backend()
        return Image(_core.blend(im1._im, im2._im, alpha))

    @staticmethod
    def alpha_composite(im1: Image, im2: Image) -> Image:
        _ensure_backend()
        return Image(_core.alpha_composite(im1._im, im2._im))

    # ------------------------------------------------------------------ #
    # Chop (binary pixel ops)                                            #
    # ------------------------------------------------------------------ #

    def _chop(
        self, other: Image, op: int, scale: float = 1.0, offset: int = 0
    ) -> Image:
        return Image(self._im.chop(other._im, op, scale, offset))

    # ------------------------------------------------------------------ #
    # Statistics                                                         #
    # ------------------------------------------------------------------ #

    def histogram(self) -> list[int]:
        """Return a list of pixel counts, one for each pixel value."""
        return self._im.histogram()

    # ------------------------------------------------------------------ #
    # Save (convenience — downloads to CPU then saves)                   #
    # ------------------------------------------------------------------ #

    def save(
        self, fp: str | object, format: str | None = None, **params: object
    ) -> None:
        """Download to CPU and save via :meth:`PIL.Image.Image.save`."""
        self.to_cpu().save(fp, format=format, **params)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Representation                                                     #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"<PIL.gpu.Image mode={self.mode} size={self.width}x{self.height} "
            f"backend={self.backend}>"
        )

    def __del__(self) -> None:
        # _im handles its own C-level cleanup via tp_dealloc
        pass


# -------------------------------------------------------------------- #
# Module-level helpers (matching PIL.Image API)                        #
# -------------------------------------------------------------------- #


def new(mode: str, size: tuple[int, int], color: int | tuple[int, ...] = 0) -> Image:
    return Image.new(mode, size, color)


def open(
    fp: str | object,
    mode: str = "r",
    formats: list[str] | tuple[str, ...] | None = None,
) -> Image:
    return Image.open(fp, mode=mode, formats=formats)


def from_cpu(cpu_image: _CPUImage.Image) -> Image:
    return Image.from_cpu(cpu_image)


def blend(im1: Image, im2: Image, alpha: float) -> Image:
    return Image.blend(im1, im2, alpha)


def alpha_composite(im1: Image, im2: Image) -> Image:
    return Image.alpha_composite(im1, im2)


def merge(mode: str, bands: list[Image] | tuple[Image, ...]) -> Image:
    """Merge single-band GPU images into a multi-band image."""
    _ensure_backend()
    band_ims = [b._im for b in bands]
    return Image(_core.merge(mode, band_ims))


def linear_gradient(mode: str, size: tuple[int, int], direction: int = 0) -> Image:
    """Create a linear gradient GPU image."""
    _ensure_backend()
    return Image(_core.linear_gradient(mode, size, direction))


def radial_gradient(mode: str, size: tuple[int, int]) -> Image:
    """Create a radial gradient GPU image."""
    _ensure_backend()
    return Image(_core.radial_gradient(mode, size))
