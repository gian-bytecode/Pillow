"""
Tests for PIL.gpu — GPU-accelerated imaging with OpenCL/CUDA.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from PIL import Image

# Skip entire module if GPU support is not available
_imaging_gpu = pytest.importorskip("_imaging_gpu", reason="GPU extension not available")


@pytest.fixture(scope="module", autouse=True)
def gpu_backend() -> Generator[None, None, None]:
    """Initialize and shut down the GPU backend once per test module."""
    _imaging_gpu.backend_init(0)
    yield
    _imaging_gpu.backend_shutdown()


# ------------------------------------------------------------------ #
# Low-level _imaging_gpu tests                                       #
# ------------------------------------------------------------------ #


class TestLowLevelGPU:
    """Tests for the raw _imaging_gpu C extension."""

    def test_backend_info(self) -> None:
        name = _imaging_gpu.get_backend_name()
        device = _imaging_gpu.get_device_name()
        assert name in ("OpenCL", "CUDA")
        assert len(device) > 0

    def test_is_available(self) -> None:
        assert _imaging_gpu.is_available()

    def test_new(self) -> None:
        im = _imaging_gpu.new("RGB", (100, 100))
        assert im.mode == "RGB"
        assert im.size == (100, 100)
        assert im.bands == 3

    def test_fill(self) -> None:
        im = _imaging_gpu.fill("RGBA", (50, 50), (255, 128, 0, 255))
        assert im.mode == "RGBA"
        assert im.size == (50, 50)
        assert im.bands == 4

    def test_from_bytes_to_bytes_roundtrip(self) -> None:
        # RGB internally stored as 4 bytes/pixel (RGBX)
        data = b"\x64\x96\xc8\x00" * 100  # 10x10 RGB
        im = _imaging_gpu.from_bytes("RGB", (10, 10), data)
        result = im.to_bytes()
        assert result == data

    def test_from_bytes_L_roundtrip(self) -> None:
        data = bytes(range(256)) * 4  # 32x32 L
        im = _imaging_gpu.from_bytes("L", (32, 32), data)
        result = im.to_bytes()
        assert result == data

    def test_from_bytes_RGBA_roundtrip(self) -> None:
        data = b"\xff\x80\x40\xc0" * 400  # 20x20 RGBA
        im = _imaging_gpu.from_bytes("RGBA", (20, 20), data)
        result = im.to_bytes()
        assert result == data

    def test_copy(self) -> None:
        im = _imaging_gpu.fill("RGBA", (50, 50), (10, 20, 30, 40))
        c = im.copy()
        assert c.mode == im.mode
        assert c.size == im.size
        assert c.to_bytes() == im.to_bytes()

    def test_convert_RGBA_to_L(self) -> None:
        im = _imaging_gpu.fill("RGBA", (50, 50), (255, 255, 255, 255))
        gray = im.convert("L")
        assert gray.mode == "L"
        assert gray.size == (50, 50)

    def test_resize(self) -> None:
        im = _imaging_gpu.fill("RGB", (100, 100), (128, 64, 32, 0))
        small = im.resize((50, 50))
        assert small.size == (50, 50)
        assert small.mode == "RGB"

    def test_transpose_flip_lr(self) -> None:
        im = _imaging_gpu.fill("RGB", (100, 50), (128, 64, 32, 0))
        flipped = im.transpose(_imaging_gpu.FLIP_LEFT_RIGHT)
        assert flipped.size == (100, 50)

    def test_transpose_flip_tb(self) -> None:
        im = _imaging_gpu.fill("RGB", (100, 50), (128, 64, 32, 0))
        flipped = im.transpose(_imaging_gpu.FLIP_TOP_BOTTOM)
        assert flipped.size == (100, 50)

    def test_gaussian_blur(self) -> None:
        im = _imaging_gpu.fill("RGB", (100, 100), (200, 100, 50, 0))
        blurred = im.gaussian_blur(3.0, 3.0, 3)
        assert blurred.mode == im.mode
        assert blurred.size == im.size

    def test_box_blur(self) -> None:
        im = _imaging_gpu.fill("RGB", (100, 100), (200, 100, 50, 0))
        blurred = im.box_blur(2.0, 2.0, 1)
        assert blurred.mode == im.mode
        assert blurred.size == im.size

    def test_histogram(self) -> None:
        im = _imaging_gpu.fill("RGBA", (10, 10), (128, 64, 32, 255))
        hist = im.histogram()
        assert len(hist) == 256 * 4  # RGBA → 4*256 bins
        assert sum(hist) == 10 * 10 * 4  # total samples

    def test_blend(self) -> None:
        a = _imaging_gpu.fill("RGBA", (50, 50), (255, 0, 0, 255))
        b = _imaging_gpu.fill("RGBA", (50, 50), (0, 255, 0, 255))
        blended = _imaging_gpu.blend(a, b, 0.5)
        assert blended.mode == "RGBA"
        assert blended.size == (50, 50)

    def test_alpha_composite(self) -> None:
        bg = _imaging_gpu.fill("RGBA", (50, 50), (255, 0, 0, 255))
        fg = _imaging_gpu.fill("RGBA", (50, 50), (0, 0, 255, 128))
        result = _imaging_gpu.alpha_composite(bg, fg)
        assert result.mode == "RGBA"
        assert result.size == (50, 50)


# ------------------------------------------------------------------ #
# High-level PIL.gpu.Image tests                                     #
# ------------------------------------------------------------------ #


class TestGpuImage:
    """Tests for the PIL.gpu.Image Python API."""

    def test_from_cpu_to_cpu_RGB(self) -> None:
        cpu = Image.new("RGB", (100, 100), (100, 150, 200))
        from PIL.gpu.Image import Image as GpuImage

        gpu = GpuImage.from_cpu(cpu)
        result = gpu.to_cpu()
        assert result.mode == "RGB"
        assert result.size == (100, 100)
        assert result.getpixel((50, 50)) == (100, 150, 200)

    def test_from_cpu_to_cpu_RGBA(self) -> None:
        cpu = Image.new("RGBA", (80, 60), (128, 64, 32, 200))
        from PIL.gpu.Image import Image as GpuImage

        gpu = GpuImage.from_cpu(cpu)
        result = gpu.to_cpu()
        assert result.getpixel((40, 30)) == (128, 64, 32, 200)

    def test_from_cpu_to_cpu_L(self) -> None:
        cpu = Image.new("L", (50, 50), 42)
        from PIL.gpu.Image import Image as GpuImage

        gpu = GpuImage.from_cpu(cpu)
        result = gpu.to_cpu()
        assert result.getpixel((25, 25)) == 42

    def test_gaussian_blur(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (200, 200), (100, 150, 200))
        gpu = GpuImage.from_cpu(cpu)
        blurred = gpu.gaussian_blur(5)
        result = blurred.to_cpu()
        assert result.mode == "RGB"
        assert result.size == (200, 200)
        # Uniform image blurred should stay the same
        assert result.getpixel((100, 100)) == (100, 150, 200)

    def test_resize(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (200, 200), (50, 100, 150))
        gpu = GpuImage.from_cpu(cpu)
        resized = gpu.resize((100, 50))
        result = resized.to_cpu()
        assert result.size == (100, 50)

    def test_convert_to_grayscale(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        gpu = GpuImage.from_cpu(cpu)
        gray = gpu.convert("L")
        result = gray.to_cpu()
        assert result.mode == "L"
        assert result.getpixel((50, 50)) == 255

    def test_transpose(self) -> None:
        from PIL.gpu.Image import Image as GpuImage
        from PIL.gpu.Image import Transpose

        cpu = Image.new("RGB", (100, 50), (200, 100, 50))
        gpu = GpuImage.from_cpu(cpu)
        flipped = gpu.transpose(Transpose.FLIP_TOP_BOTTOM)
        result = flipped.to_cpu()
        assert result.size == (100, 50)

    def test_copy(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (1, 2, 3))
        gpu = GpuImage.from_cpu(cpu)
        c = gpu.copy()
        result = c.to_cpu()
        assert result.getpixel((50, 50)) == (1, 2, 3)

    def test_properties(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        gpu = GpuImage.new("RGBA", (320, 240))
        assert gpu.mode == "RGBA"
        assert gpu.size == (320, 240)
        assert gpu.width == 320
        assert gpu.height == 240
        assert gpu.bands == 4


# ------------------------------------------------------------------ #
# PIL.gpu.ImageOps tests                                             #
# ------------------------------------------------------------------ #


class TestGpuImageOps:
    """Tests for PIL.gpu.ImageOps."""

    def test_invert(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (100, 150, 200))
        gpu = GpuImage.from_cpu(cpu)
        inverted = ImageOps.invert(gpu)
        result = inverted.to_cpu()
        assert result.mode == "RGB"
        assert result.size == (50, 50)

    def test_grayscale(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (255, 255, 255))
        gpu = GpuImage.from_cpu(cpu)
        gray = ImageOps.grayscale(gpu)
        assert gray.mode == "L"

    def test_flip(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (100, 100, 100))
        gpu = GpuImage.from_cpu(cpu)
        flipped = ImageOps.flip(gpu)
        assert flipped.size == (50, 50)

    def test_mirror(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (100, 100, 100))
        gpu = GpuImage.from_cpu(cpu)
        mirrored = ImageOps.mirror(gpu)
        assert mirrored.size == (50, 50)


# ------------------------------------------------------------------ #
# PIL.gpu.ImageEnhance tests                                         #
# ------------------------------------------------------------------ #


class TestGpuImageEnhance:
    """Tests for PIL.gpu.ImageEnhance."""

    def test_brightness(self) -> None:
        from PIL.gpu import ImageEnhance
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (100, 100, 100))
        gpu = GpuImage.from_cpu(cpu)
        enhancer = ImageEnhance.Brightness(gpu)
        result = enhancer.enhance(1.0)  # no change
        assert result.size == (100, 100)

    def test_contrast(self) -> None:
        from PIL.gpu import ImageEnhance
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (100, 100, 100))
        gpu = GpuImage.from_cpu(cpu)
        enhancer = ImageEnhance.Contrast(gpu)
        result = enhancer.enhance(1.5)
        assert result.size == (100, 100)


# ------------------------------------------------------------------ #
# New feature tests                                                  #
# ------------------------------------------------------------------ #


class TestGpuNewFeatures:
    """Tests for newly added GPU operations."""

    def test_crop(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        cropped = gpu.crop((10, 10, 50, 50))
        assert cropped.size == (40, 40)
        assert cropped.mode == "RGB"

    def test_expand(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        expanded = gpu.expand(10, 0)
        assert expanded.size == (120, 120)

    def test_offset(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        shifted = gpu.offset(10, 20)
        assert shifted.size == (100, 100)

    def test_negative(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (100, 150, 200))
        gpu = GpuImage.from_cpu(cpu)
        neg = gpu.negative()
        result = neg.to_cpu()
        px = result.getpixel((0, 0))
        assert px == (155, 105, 55)

    def test_posterize(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (123, 200, 50))
        gpu = GpuImage.from_cpu(cpu)
        post = gpu.posterize(4)
        assert post.size == (100, 100)

    def test_solarize(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (200, 50, 100))
        gpu = GpuImage.from_cpu(cpu)
        sol = gpu.solarize(128)
        assert sol.size == (100, 100)
        result = sol.to_cpu()
        px = result.getpixel((0, 0))
        # 200 > 128, so R should be inverted: 255-200=55
        # 50 < 128, stays: 50
        # 100 < 128, stays: 100
        assert px == (55, 50, 100)

    def test_split_native(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (100, 150, 200))
        gpu = GpuImage.from_cpu(cpu)
        bands = gpu.split()
        assert len(bands) == 3
        for b in bands:
            assert b.mode == "L"
            assert b.size == (50, 50)

    def test_getbbox(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        bbox = gpu.getbbox()
        # All transparent -> bbox is None
        assert bbox is None

    def test_getextrema(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("L", (50, 50), 128)
        gpu = GpuImage.from_cpu(cpu)
        ext = gpu.getextrema()
        assert ext == (128, 128)

    def test_effect_spread(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        spread = gpu.effect_spread(5)
        assert spread.size == (100, 100)

    def test_point_lut(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("L", (50, 50), 100)
        gpu = GpuImage.from_cpu(cpu)
        # Invert via LUT
        inv_lut = [255 - i for i in range(256)]
        result = gpu.point(inv_lut)
        cpu_result = result.to_cpu()
        assert cpu_result.getpixel((0, 0)) == 155

    def test_paste(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        bg = GpuImage.new("RGB", (100, 100), (0, 0, 0))
        fg = GpuImage.new("RGB", (20, 20), (255, 0, 0))
        bg.paste(fg, (10, 10))
        result = bg.to_cpu()
        px = result.getpixel((15, 15))
        assert isinstance(px, tuple)
        assert abs(px[0] - 255) <= 1  # allow rounding tolerance
        assert px[1] == 0
        assert px[2] == 0


class TestGpuImageOpsNew:
    """Tests for new ImageOps functions."""

    def test_posterize(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (123, 200, 50))
        gpu = GpuImage.from_cpu(cpu)
        result = ImageOps.posterize(gpu, 4)
        assert result.size == (50, 50)

    def test_solarize(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (200, 50, 100))
        gpu = GpuImage.from_cpu(cpu)
        result = ImageOps.solarize(gpu, 128)
        assert result.size == (50, 50)

    def test_equalize(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("L", (50, 50), 128)
        gpu = GpuImage.from_cpu(cpu)
        result = ImageOps.equalize(gpu)
        assert result.size == (50, 50)

    def test_autocontrast(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("L", (50, 50), 128)
        gpu = GpuImage.from_cpu(cpu)
        result = ImageOps.autocontrast(gpu)
        assert result.size == (50, 50)

    def test_expand(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        result = ImageOps.expand(gpu, 10)
        assert result.size == (70, 70)

    def test_crop(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        result = ImageOps.crop(gpu, 10)
        assert result.size == (80, 80)

    def test_pad(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 50), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        result = ImageOps.pad(gpu, (100, 100))
        assert result.size == (100, 100)

    def test_cover(self) -> None:
        from PIL.gpu import ImageOps
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 50), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        result = ImageOps.cover(gpu, (200, 200))
        assert result.width >= 200
        assert result.height >= 200


class TestGpuImageChopsNew:
    """Tests for new ImageChops functions."""

    def test_offset(self) -> None:
        from PIL.gpu import ImageChops
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (100, 100), (255, 0, 0))
        gpu = GpuImage.from_cpu(cpu)
        result = ImageChops.offset(gpu, 10, 20)
        assert result.size == (100, 100)

    def test_constant(self) -> None:
        from PIL.gpu import ImageChops
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("L", (50, 50), 0)
        gpu = GpuImage.from_cpu(cpu)
        result = ImageChops.constant(gpu, 128)
        assert result.size == (50, 50)

    def test_duplicate(self) -> None:
        from PIL.gpu import ImageChops
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (50, 50), (100, 100, 100))
        gpu = GpuImage.from_cpu(cpu)
        dup = ImageChops.duplicate(gpu)
        assert dup.size == gpu.size


class TestGpuNewModules:
    """Tests for new GPU modules."""

    def test_image_draw_rectangle(self) -> None:
        from PIL.gpu import ImageDraw
        from PIL.gpu.Image import Image as GpuImage

        gpu = GpuImage.new("RGB", (100, 100), (0, 0, 0))
        draw = ImageDraw.Draw(gpu)
        draw.rectangle([10, 10, 50, 50], fill=(255, 0, 0))
        result = gpu.to_cpu()
        assert result.getpixel((30, 30)) == (255, 0, 0)

    def test_image_sequence(self) -> None:
        from PIL.gpu import ImageSequence
        from PIL.gpu.Image import Image as GpuImage

        gpu = GpuImage.new("RGB", (50, 50), (128, 128, 128))
        frames = list(ImageSequence.Iterator(gpu))
        assert len(frames) == 1

    def test_image_transform_constants(self) -> None:
        from PIL.gpu import ImageTransform

        assert ImageTransform.AFFINE is not None
        assert ImageTransform.PERSPECTIVE is not None

    def test_image_palette_import(self) -> None:
        from PIL.gpu import ImagePalette

        assert hasattr(ImagePalette, "ImagePalette")


class TestGpuNewKernels:
    """Tests for newly added GPU-native kernels.

    Covers: reduce, rank_filter, mode_filter, YCbCr.
    """

    def test_reduce_2x(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        # 100x100 white image -> reduce by 2 -> 50x50
        cpu = Image.new("RGB", (100, 100), (200, 100, 50))
        gpu = GpuImage.from_cpu(cpu)
        reduced = gpu.reduce(2)
        result = reduced.to_cpu()
        assert result.size == (50, 50)
        r, g, b = result.getpixel((25, 25))  # type: ignore[misc]
        assert abs(r - 200) <= 1
        assert abs(g - 100) <= 1
        assert abs(b - 50) <= 1

    def test_reduce_3x(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("L", (99, 99), 128)
        gpu = GpuImage.from_cpu(cpu)
        reduced = gpu.reduce(3)
        result = reduced.to_cpu()
        assert result.size == (33, 33)
        assert abs(result.getpixel((16, 16)) - 128) <= 1  # type: ignore[operator]

    def test_reduce_asymmetric(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("RGB", (120, 80), (10, 20, 30))
        gpu = GpuImage.from_cpu(cpu)
        reduced = gpu.reduce((2, 4))
        result = reduced.to_cpu()
        assert result.size == (60, 20)

    def test_rank_filter_median(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        # Create image with salt-and-pepper noise on gray background
        cpu = Image.new("L", (64, 64), 128)
        pixels = cpu.load()
        assert pixels is not None
        pixels[32, 32] = 255  # salt
        pixels[33, 33] = 0  # pepper
        gpu = GpuImage.from_cpu(cpu)
        # Median filter (3x3 window, rank=4 = center of 9 values)
        filtered = gpu.rank_filter(3, 4)
        result = filtered.to_cpu()
        # Noise should be removed
        assert result.getpixel((32, 32)) == 128
        assert result.getpixel((33, 33)) == 128

    def test_rank_filter_min(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("L", (32, 32), 100)
        pixels = cpu.load()
        assert pixels is not None
        pixels[16, 16] = 50
        gpu = GpuImage.from_cpu(cpu)
        # Min filter: rank=0
        filtered = gpu.rank_filter(3, 0)
        result = filtered.to_cpu()
        # Neighbors of (16,16) should see min=50
        assert result.getpixel((15, 16)) == 50
        assert result.getpixel((17, 16)) == 50

    def test_rank_filter_max(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        cpu = Image.new("L", (32, 32), 100)
        pixels = cpu.load()
        assert pixels is not None
        pixels[16, 16] = 200
        gpu = GpuImage.from_cpu(cpu)
        # Max filter: rank=8 (3*3 - 1)
        filtered = gpu.rank_filter(3, 8)
        result = filtered.to_cpu()
        # Neighbors should see max=200
        assert result.getpixel((15, 16)) == 200

    def test_mode_filter(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        # Create L image where most values are 100, a few are 200
        cpu = Image.new("L", (32, 32), 100)
        pixels = cpu.load()
        assert pixels is not None
        pixels[16, 16] = 200
        gpu = GpuImage.from_cpu(cpu)
        filtered = gpu.mode_filter(3)
        result = filtered.to_cpu()
        # The mode should be 100 everywhere (200 is outnumbered)
        assert result.getpixel((16, 16)) == 100

    def test_convert_ycbcr_to_rgb(self) -> None:
        from PIL.gpu.Image import Image as GpuImage

        # Create a YCbCr image and convert to RGB on GPU
        cpu_rgb = Image.new("RGB", (32, 32), (255, 0, 0))
        cpu_ycbcr = cpu_rgb.convert("YCbCr")
        gpu = GpuImage.from_cpu(cpu_ycbcr)
        gpu_rgb = gpu.convert("RGB")
        result = gpu_rgb.to_cpu()
        r, g, b = result.getpixel((16, 16))  # type: ignore[misc]
        # Allow some rounding difference
        assert abs(r - 255) <= 2
        assert abs(g - 0) <= 2
        assert abs(b - 0) <= 2
