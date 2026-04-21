"""
Benchmark: PIL.gpu (OpenCL/CUDA) vs PIL (CPU)

Compares performance of common image operations on GPU vs CPU.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

from PIL import Image, ImageFilter


def _bench(
    label: str, func: Callable[[], object], warmup: int = 2, runs: int = 10
) -> float:
    """Run *func* with warmup and return median time in ms."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times)
    return med


def main() -> None:
    import _imaging_gpu as g  # type: ignore[import-not-found]

    from PIL.gpu import Image as GpuImage

    g.backend_init(0)
    print(f"GPU Backend: {g.get_backend_name()}")
    print(f"GPU Device:  {g.get_device_name()}")
    print()

    sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]

    for w, h in sizes:
        print(f"=== {w}x{h} ===")

        # Prepare CPU and GPU images
        cpu_img = Image.new("RGB", (w, h), (100, 150, 200))
        gpu_img = GpuImage.from_cpu(cpu_img)

        # --- Gaussian Blur ---
        cpu_blur = _bench(
            "CPU GaussianBlur(5)",
            lambda: cpu_img.filter(ImageFilter.GaussianBlur(5)),
        )
        gpu_blur = _bench(
            "GPU gaussian_blur(5)",
            lambda: gpu_img.gaussian_blur(5),
        )
        speedup = cpu_blur / gpu_blur if gpu_blur > 0 else float("inf")
        print(
            f"  GaussianBlur(5): "
            f"CPU {cpu_blur:7.2f} ms  GPU {gpu_blur:7.2f} ms  -> {speedup:.1f}x"
        )

        # --- Box Blur ---
        cpu_box = _bench(
            "CPU BoxBlur(5)",
            lambda: cpu_img.filter(ImageFilter.BoxBlur(5)),
        )
        gpu_box = _bench(
            "GPU box_blur(5)",
            lambda: gpu_img.box_blur(5),
        )
        speedup = cpu_box / gpu_box if gpu_box > 0 else float("inf")
        print(
            f"  BoxBlur(5):      "
            f"CPU {cpu_box:7.2f} ms  GPU {gpu_box:7.2f} ms  -> {speedup:.1f}x"
        )

        # --- Resize (down to 1/4) ---
        new_size = (w // 4, h // 4)
        cpu_resize = _bench(
            "CPU resize",
            lambda: cpu_img.resize(new_size),
        )
        gpu_resize = _bench(
            "GPU resize",
            lambda: gpu_img.resize(new_size),
        )
        speedup = cpu_resize / gpu_resize if gpu_resize > 0 else float("inf")
        print(
            f"  Resize(1/4):     "
            f"CPU {cpu_resize:7.2f} ms  GPU {gpu_resize:7.2f} ms  -> {speedup:.1f}x"
        )

        # --- Convert RGB -> L ---
        cpu_conv = _bench(
            "CPU convert L",
            lambda: cpu_img.convert("L"),
        )
        gpu_conv = _bench(
            "GPU convert L",
            lambda: gpu_img.convert("L"),
        )
        speedup = cpu_conv / gpu_conv if gpu_conv > 0 else float("inf")
        print(
            f"  Convert(L):      "
            f"CPU {cpu_conv:7.2f} ms  GPU {gpu_conv:7.2f} ms  -> {speedup:.1f}x"
        )

        # --- Transpose (flip LR) ---
        cpu_flip = _bench(
            "CPU transpose FLIP_LR",
            lambda: cpu_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        )
        gpu_flip = _bench(
            "GPU transpose FLIP_LR",
            lambda: gpu_img.transpose(0),  # FLIP_LEFT_RIGHT = 0
        )
        speedup = cpu_flip / gpu_flip if gpu_flip > 0 else float("inf")
        print(
            f"  Flip LR:         "
            f"CPU {cpu_flip:7.2f} ms  GPU {gpu_flip:7.2f} ms  -> {speedup:.1f}x"
        )

        # --- Full pipeline: upload + blur + resize + download ---
        def cpu_pipeline() -> object:
            return cpu_img.filter(ImageFilter.GaussianBlur(5)).resize(new_size)

        def gpu_pipeline() -> object:
            g_im = GpuImage.from_cpu(cpu_img)
            g_blur = g_im.gaussian_blur(5)
            g_small = g_blur.resize(new_size)
            return g_small.to_cpu()

        cpu_pipe = _bench("CPU pipeline", cpu_pipeline)
        gpu_pipe = _bench("GPU pipeline", gpu_pipeline)
        speedup = cpu_pipe / gpu_pipe if gpu_pipe > 0 else float("inf")
        print(
            f"  Pipeline(blur+resize+xfer): "
            f"CPU {cpu_pipe:7.2f} ms  GPU {gpu_pipe:7.2f} ms  -> {speedup:.1f}x"
        )

        print()

    g.backend_shutdown()


if __name__ == "__main__":
    main()
