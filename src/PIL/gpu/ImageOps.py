"""
PIL.gpu.ImageOps — GPU-accelerated image operations.

Mirrors PIL.ImageOps where possible.
"""

from __future__ import annotations

from PIL import _imaging_gpu as _core

from .Image import Image, Resampling


def flip(image: Image) -> Image:
    """Flip top to bottom."""
    return image.transpose(_core.FLIP_TOP_BOTTOM)


def mirror(image: Image) -> Image:
    """Flip left to right."""
    return image.transpose(_core.FLIP_LEFT_RIGHT)


def grayscale(image: Image) -> Image:
    """Convert to grayscale."""
    return image.convert("L")


def invert(image: Image) -> Image:
    """Invert pixel values."""
    return image.point_transform(-1.0, 255.0)


def scale(image: Image, factor: float, resample: int = Resampling.BILINEAR) -> Image:
    """Scale by a factor."""
    w, h = image.size
    new_size = (max(1, int(w * factor)), max(1, int(h * factor)))
    return image.resize(new_size, resample)


def fit(
    image: Image,
    size: tuple[int, int],
    method: int = Resampling.BILINEAR,
    centering: tuple[float, float] = (0.5, 0.5),
) -> Image:
    """Resize and crop to fit exact *size*, preserving aspect ratio."""
    im_w, im_h = image.size
    out_w, out_h = size
    im_ratio = im_w / im_h
    out_ratio = out_w / out_h

    if im_ratio > out_ratio:
        # Image is wider — crop width
        new_h = out_h
        new_w = int(im_w * out_h / im_h + 0.5)
    else:
        new_w = out_w
        new_h = int(im_h * out_w / im_w + 0.5)

    image = image.resize((new_w, new_h), method)

    # Crop to target
    cx, cy = centering
    left = int((new_w - out_w) * cx + 0.5)
    top = int((new_h - out_h) * cy + 0.5)
    box = (float(left), float(top), float(left + out_w), float(top + out_h))
    return image.resize(size, Resampling.NEAREST, box)


def contain(
    image: Image, size: tuple[int, int], method: int = Resampling.BILINEAR
) -> Image:
    """Resize to fit within *size*, preserving aspect ratio."""
    im_w, im_h = image.size
    out_w, out_h = size
    ratio = min(out_w / im_w, out_h / im_h)
    new_w = max(1, int(im_w * ratio + 0.5))
    new_h = max(1, int(im_h * ratio + 0.5))
    return image.resize((new_w, new_h), method)


def cover(
    image: Image, size: tuple[int, int], method: int = Resampling.BILINEAR
) -> Image:
    """Resize to cover *size*, preserving aspect ratio (may be larger)."""
    im_w, im_h = image.size
    out_w, out_h = size
    ratio = max(out_w / im_w, out_h / im_h)
    new_w = max(1, int(im_w * ratio + 0.5))
    new_h = max(1, int(im_h * ratio + 0.5))
    return image.resize((new_w, new_h), method)


def pad(
    image: Image,
    size: tuple[int, int],
    method: int = Resampling.BILINEAR,
    color=0,
    centering: tuple[float, float] = (0.5, 0.5),
) -> Image:
    """Resize to fit and pad to exact *size*."""
    resized = contain(image, size, method)
    out = Image.new(image.mode, size, color)
    rw, rh = resized.size
    ox = int((size[0] - rw) * centering[0])
    oy = int((size[1] - rh) * centering[1])
    out.paste(resized, (ox, oy))
    return out


def crop(image: Image, border: int = 0) -> Image:
    """Remove a border from the image."""
    w, h = image.size
    return image.crop((border, border, w - border, h - border))


def expand(image: Image, border: int = 0, fill=0) -> Image:
    """Add a border around the image."""
    return image.expand(border, fill)


def autocontrast(
    image: Image, cutoff=0, ignore=None, mask=None, preserve_tone=False
) -> Image:
    """Normalize image contrast by histogram stretching."""
    h = image.histogram()
    bands = image.bands
    band_size = len(h) // bands if bands > 0 else 256

    lut = []
    for b in range(bands):
        bh = h[b * band_size : (b + 1) * band_size]
        n = sum(bh)
        if isinstance(cutoff, (tuple, list)):
            cut_low = n * cutoff[0] // 100
            cut_high = n * cutoff[1] // 100
        else:
            cut_low = n * cutoff // 100
            cut_high = cut_low

        lo, acc = 0, 0
        for i in range(256):
            if ignore is not None and i in (
                ignore if isinstance(ignore, (list, tuple)) else [ignore]
            ):
                continue
            acc += bh[i]
            if acc > cut_low:
                lo = i
                break

        hi, acc = 255, 0
        for i in range(255, -1, -1):
            if ignore is not None and i in (
                ignore if isinstance(ignore, (list, tuple)) else [ignore]
            ):
                continue
            acc += bh[i]
            if acc > cut_high:
                hi = i
                break

        if hi <= lo:
            lut.extend(list(range(256)))
        else:
            scale_v = 255.0 / (hi - lo)
            offset_v = -lo * scale_v
            lut.extend(
                max(0, min(255, int(i * scale_v + offset_v + 0.5))) for i in range(256)
            )

    return image.point(lut)


def colorize(
    image: Image, black, white, mid=None, blackpoint=0, whitepoint=255, midpoint=127
) -> Image:
    """Colorize a grayscale image (CPU fallback)."""
    from PIL import ImageOps as _CPUOps

    cpu = image.to_cpu()
    if cpu.mode != "L":
        cpu = cpu.convert("L")
    result = _CPUOps.colorize(cpu, black, white, mid, blackpoint, whitepoint, midpoint)
    return Image.from_cpu(result)


def equalize(image: Image, mask=None) -> Image:
    """Equalize the image histogram."""
    h = image.histogram()
    bands = image.bands
    band_size = len(h) // bands if bands > 0 else 256

    lut = []
    for b in range(bands):
        bh = h[b * band_size : (b + 1) * band_size]
        n = sum(bh)
        if n == 0:
            lut.extend(list(range(256)))
            continue
        cdf = []
        acc = 0
        for v in bh:
            acc += v
            cdf.append(acc)
        cdf_min = next((c for c in cdf if c > 0), 0)
        denom = n - cdf_min
        if denom <= 0:
            lut.extend(list(range(256)))
            continue
        lut.extend(
            max(0, min(255, int((cdf[i] - cdf_min) * 255 / denom + 0.5)))
            for i in range(256)
        )

    lut_bytes = bytes(lut)
    return image._equalize_with_lut(lut_bytes)


def posterize(image: Image, bits: int) -> Image:
    """Reduce number of bits per channel."""
    return image.posterize(bits)


def solarize(image: Image, threshold: int = 128) -> Image:
    """Solarize: invert values above threshold."""
    return image.solarize(threshold)


def exif_transpose(image: Image) -> Image:
    """Transpose based on EXIF orientation (no-op for GPU images)."""
    return image


def deform(image: Image, deformer, resample=Resampling.BILINEAR) -> Image:
    """Deform image using a deformer object (CPU fallback)."""
    from PIL import ImageOps as _CPUOps

    cpu = image.to_cpu()
    result = _CPUOps.deform(cpu, deformer, resample=resample)
    return Image.from_cpu(result)
