"""
Comprehensive GPU vs CPU Benchmark Suite for Pillow GPU Backend.

Tests all GPU-accelerated operations across image sizes 128, 256, 512, 1024, 2048.
Measures:
  - CPU-only time (PIL)
  - GPU compute time (excluding transfer)
  - GPU total time (including CPU->GPU->CPU transfer)
"""

from __future__ import annotations

import json
import statistics
import sys
import time

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from PIL.gpu import Image as GpuImageModule
from PIL.gpu import ImageEnhance as GpuImageEnhance
from PIL.gpu import ImageOps as GpuImageOps

# GPU imports
from PIL.gpu.Image import Image as GpuImage

SIZES = [128, 256, 512, 1024, 2048]
WARMUP = 2
REPEATS = 10


def timeit(func, warmup=WARMUP, repeats=REPEATS):
    """Run func warmup+repeats times, return median time in ms."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.median(times)


def make_images(size, mode="RGB"):
    """Create CPU and GPU test images."""
    cpu = Image.new(
        mode,
        (size, size),
        (
            (128, 64, 192)
            if mode == "RGB"
            else (128, 64, 192, 255) if mode == "RGBA" else 128
        ),
    )
    # Add some variation
    pixels = cpu.load()
    for y in range(0, size, 4):
        for x in range(0, size, 4):
            if mode == "L":
                pixels[x, y] = (x * 7 + y * 13) % 256
            elif mode == "RGBA":
                pixels[x, y] = ((x * 7) % 256, (y * 13) % 256, ((x + y) * 5) % 256, 200)
            else:
                pixels[x, y] = ((x * 7) % 256, (y * 13) % 256, ((x + y) * 5) % 256)
    return cpu


def make_gpu(cpu):
    return GpuImage.from_cpu(cpu)


def benchmark_transfer(sizes):
    """Benchmark CPU->GPU and GPU->CPU transfer times."""
    results = {}
    for s in sizes:
        cpu = make_images(s)
        # CPU -> GPU
        t_up = timeit(lambda: GpuImage.from_cpu(cpu))
        gpu = GpuImage.from_cpu(cpu)
        # GPU -> CPU
        t_down = timeit(lambda: gpu.to_cpu())
        # Round-trip
        t_rt = timeit(lambda: GpuImage.from_cpu(cpu).to_cpu())
        results[s] = {"upload_ms": t_up, "download_ms": t_down, "roundtrip_ms": t_rt}
        print(
            f"  Transfer {s}x{s}: "
            f"up={t_up:.3f}ms  down={t_down:.3f}ms  rt={t_rt:.3f}ms"
        )
    return results


def run_benchmarks():
    all_results = {}

    # ============================================================
    print("=" * 60)
    print("TRANSFER BENCHMARKS")
    print("=" * 60)
    all_results["transfer"] = benchmark_transfer(SIZES)

    # ============================================================
    # Define all benchmarks as (name, cpu_func, gpu_func, mode)
    # cpu_func(cpu_img) -> result
    # gpu_func(gpu_img) -> result  (compute only, no transfer)
    # ============================================================

    benchmarks = []

    # --- Basic operations ---
    benchmarks.append(("copy", lambda c: c.copy(), lambda g: g.copy(), "RGB"))

    benchmarks.append(
        ("convert_RGB_to_L", lambda c: c.convert("L"), lambda g: g.convert("L"), "RGB")
    )

    benchmarks.append(
        (
            "convert_L_to_RGB",
            lambda c: c.convert("RGB"),
            lambda g: g.convert("RGB"),
            "L",
        )
    )

    benchmarks.append(
        (
            "convert_RGB_to_RGBA",
            lambda c: c.convert("RGBA"),
            lambda g: g.convert("RGBA"),
            "RGB",
        )
    )

    # --- Resize ---
    benchmarks.append(
        (
            "resize_half_nearest",
            lambda c: c.resize((c.width // 2, c.height // 2), Image.Resampling.NEAREST),
            lambda g: g.resize((g.width // 2, g.height // 2), 0),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "resize_half_bilinear",
            lambda c: c.resize(
                (c.width // 2, c.height // 2), Image.Resampling.BILINEAR
            ),
            lambda g: g.resize((g.width // 2, g.height // 2), 2),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "resize_double_bilinear",
            lambda c: c.resize((c.width * 2, c.height * 2), Image.Resampling.BILINEAR),
            lambda g: g.resize((g.width * 2, g.height * 2), 2),
            "RGB",
        )
    )

    # --- Transpose ---
    benchmarks.append(
        (
            "transpose_flip_lr",
            lambda c: c.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            lambda g: g.transpose(0),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "transpose_rotate_90",
            lambda c: c.transpose(Image.Transpose.ROTATE_90),
            lambda g: g.transpose(2),
            "RGB",
        )
    )

    # --- Blur ---
    benchmarks.append(
        (
            "gaussian_blur_r2",
            lambda c: c.filter(ImageFilter.GaussianBlur(2)),
            lambda g: g.gaussian_blur(2.0),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "gaussian_blur_r5",
            lambda c: c.filter(ImageFilter.GaussianBlur(5)),
            lambda g: g.gaussian_blur(5.0),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "box_blur_r2",
            lambda c: c.filter(ImageFilter.BoxBlur(2)),
            lambda g: g.box_blur(2.0),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "box_blur_r5",
            lambda c: c.filter(ImageFilter.BoxBlur(5)),
            lambda g: g.box_blur(5.0),
            "RGB",
        )
    )

    # --- Unsharp mask ---
    benchmarks.append(
        (
            "unsharp_mask",
            lambda c: c.filter(ImageFilter.UnsharpMask(2, 150, 3)),
            lambda g: g.unsharp_mask(2.0, 150, 3),
            "RGB",
        )
    )

    # --- Convolution (3x3 SHARPEN) ---
    benchmarks.append(
        (
            "convolve_sharpen",
            lambda c: c.filter(ImageFilter.SHARPEN),
            lambda g: g.filter(ImageFilter.SHARPEN),
            "RGB",
        )
    )

    # --- Point ops ---
    _lut = bytes([(255 - i) for i in range(256)] * 3)
    benchmarks.append(
        (
            "point_lut_invert",
            lambda c: c.point(lambda x: 255 - x),
            lambda g: g.point(lambda x: 255 - x),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "point_transform",
            lambda c: c.point(lambda x: min(255, int(x * 1.5 + 10))),
            lambda g: g.point_transform(1.5, 10.0),
            "RGB",
        )
    )

    # --- Negative / Posterize / Solarize ---
    benchmarks.append(
        ("negative", lambda c: ImageOps.invert(c), lambda g: g.negative(), "RGB")
    )

    benchmarks.append(
        (
            "posterize_4",
            lambda c: ImageOps.posterize(c, 4),
            lambda g: g.posterize(4),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "solarize_128",
            lambda c: ImageOps.solarize(c, 128),
            lambda g: g.solarize(128),
            "RGB",
        )
    )

    # --- Blend / Alpha composite ---
    # These need two images, handled specially
    benchmarks.append(("blend_0.5", "BLEND", "BLEND", "RGB"))

    benchmarks.append(("alpha_composite", "ALPHA_COMP", "ALPHA_COMP", "RGBA"))

    # --- Histogram ---
    benchmarks.append(
        ("histogram", lambda c: c.histogram(), lambda g: g.histogram(), "L")
    )

    benchmarks.append(
        ("histogram_RGB", lambda c: c.histogram(), lambda g: g.histogram(), "RGB")
    )

    # --- Getbbox / Getextrema ---
    benchmarks.append(("getbbox", lambda c: c.getbbox(), lambda g: g.getbbox(), "L"))

    benchmarks.append(
        ("getextrema", lambda c: c.getextrema(), lambda g: g.getextrema(), "L")
    )

    # --- Crop / Expand / Offset ---
    def _crop_quarter_cpu(c):
        x0, y0 = c.width // 4, c.height // 4
        return c.crop((x0, y0, 3 * c.width // 4, 3 * c.height // 4))

    def _crop_quarter_gpu(g):
        x0, y0 = g.width // 4, g.height // 4
        return g.crop((x0, y0, 3 * g.width // 4, 3 * g.height // 4))

    benchmarks.append(("crop_quarter", _crop_quarter_cpu, _crop_quarter_gpu, "RGB"))

    benchmarks.append(
        (
            "expand_10px",
            lambda c: ImageOps.expand(c, 10, fill=(0, 0, 0)),
            lambda g: g.expand(10, fill=0),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "offset",
            lambda c: ImageChops.offset(c, 50, 50),
            lambda g: g.offset(50, 50),
            "RGB",
        )
    )

    # --- Split / GetChannel ---
    benchmarks.append(("split", lambda c: c.split(), lambda g: g.split(), "RGB"))

    benchmarks.append(
        ("getchannel_0", lambda c: c.getchannel(0), lambda g: g.getchannel(0), "RGB")
    )

    # --- Effect spread ---
    benchmarks.append(
        (
            "effect_spread_5",
            lambda c: c.effect_spread(5),
            lambda g: g.effect_spread(5),
            "RGB",
        )
    )

    # --- Reduce ---
    benchmarks.append(
        ("reduce_2x", lambda c: c.reduce(2), lambda g: g.reduce(2), "RGB")
    )

    benchmarks.append(
        ("reduce_4x", lambda c: c.reduce(4), lambda g: g.reduce(4), "RGB")
    )

    # --- Rank filter ---
    benchmarks.append(
        (
            "rank_filter_median_3",
            lambda c: c.filter(ImageFilter.MedianFilter(3)),
            lambda g: g.rank_filter(3, 4),
            "L",
        )
    )

    benchmarks.append(
        (
            "rank_filter_median_5",
            lambda c: c.filter(ImageFilter.MedianFilter(5)),
            lambda g: g.rank_filter(5, 12),
            "L",
        )
    )

    # --- Mode filter ---
    benchmarks.append(
        (
            "mode_filter_3",
            lambda c: c.filter(ImageFilter.ModeFilter(3)),
            lambda g: g.mode_filter(3),
            "L",
        )
    )

    # --- ImageOps ---
    benchmarks.append(
        (
            "imageops_autocontrast",
            lambda c: ImageOps.autocontrast(c),
            lambda g: GpuImageOps.autocontrast(g),
            "RGB",
        )
    )

    benchmarks.append(
        (
            "imageops_equalize",
            lambda c: ImageOps.equalize(c),
            lambda g: GpuImageOps.equalize(g),
            "RGB",
        )
    )

    # --- ImageEnhance ---
    benchmarks.append(("enhance_brightness_1.5", "BRIGHTNESS", "BRIGHTNESS", "RGB"))

    benchmarks.append(("enhance_contrast_1.5", "CONTRAST", "CONTRAST", "RGB"))

    benchmarks.append(("enhance_color_0.5", "COLOR", "COLOR", "RGB"))

    benchmarks.append(("enhance_sharpness_2.0", "SHARPNESS", "SHARPNESS", "RGB"))

    # --- Gradients ---
    benchmarks.append(("linear_gradient", "LINGRAD", "LINGRAD", "L"))

    benchmarks.append(("radial_gradient", "RADGRAD", "RADGRAD", "L"))

    # --- Paste ---
    benchmarks.append(("paste", "PASTE", "PASTE", "RGBA"))

    # --- Transform (affine) ---
    benchmarks.append(("transform_affine", "AFFINE", "AFFINE", "RGB"))

    # ============================================================
    print("\n" + "=" * 60)
    print("OPERATION BENCHMARKS")
    print("=" * 60)

    for bname, cpu_fn, gpu_fn, mode in benchmarks:
        print(f"\n--- {bname} ({mode}) ---")
        all_results[bname] = {}

        for s in SIZES:
            cpu_img = make_images(s, mode)
            gpu_img = make_gpu(cpu_img)

            # Handle special cases
            if cpu_fn == "BLEND":
                cpu_img2 = make_images(s, mode)
                gpu_img2 = make_gpu(cpu_img2)
                t_cpu = timeit(lambda: Image.blend(cpu_img, cpu_img2, 0.5))
                t_gpu = timeit(lambda: GpuImage.blend(gpu_img, gpu_img2, 0.5))
            elif cpu_fn == "ALPHA_COMP":
                cpu_img2 = make_images(s, "RGBA")
                gpu_img2 = make_gpu(cpu_img2)
                t_cpu = timeit(lambda: Image.alpha_composite(cpu_img, cpu_img2))
                t_gpu = timeit(lambda: GpuImage.alpha_composite(gpu_img, gpu_img2))
            elif cpu_fn == "BRIGHTNESS":
                ce = ImageEnhance.Brightness(cpu_img)
                ge = GpuImageEnhance.Brightness(gpu_img)
                t_cpu = timeit(lambda: ce.enhance(1.5))
                t_gpu = timeit(lambda: ge.enhance(1.5))
            elif cpu_fn == "CONTRAST":
                ce = ImageEnhance.Contrast(cpu_img)
                ge = GpuImageEnhance.Contrast(gpu_img)
                t_cpu = timeit(lambda: ce.enhance(1.5))
                t_gpu = timeit(lambda: ge.enhance(1.5))
            elif cpu_fn == "COLOR":
                ce = ImageEnhance.Color(cpu_img)
                ge = GpuImageEnhance.Color(gpu_img)
                t_cpu = timeit(lambda: ce.enhance(0.5))
                t_gpu = timeit(lambda: ge.enhance(0.5))
            elif cpu_fn == "SHARPNESS":
                ce = ImageEnhance.Sharpness(cpu_img)
                ge = GpuImageEnhance.Sharpness(gpu_img)
                t_cpu = timeit(lambda: ce.enhance(2.0))
                t_gpu = timeit(lambda: ge.enhance(2.0))
            elif cpu_fn == "LINGRAD":
                t_cpu = timeit(lambda: Image.linear_gradient("L"))
                t_gpu = timeit(lambda: GpuImageModule.linear_gradient("L", (s, s)))
            elif cpu_fn == "RADGRAD":
                t_cpu = timeit(lambda: Image.radial_gradient("L"))
                t_gpu = timeit(lambda: GpuImageModule.radial_gradient("L", (s, s)))
            elif cpu_fn == "PASTE":
                cpu_src = make_images(s // 2, "RGBA")
                gpu_src = make_gpu(cpu_src)

                def cpu_paste():
                    c = cpu_img.copy()
                    c.paste(cpu_src, (0, 0))

                def gpu_paste():
                    g = gpu_img.copy()
                    g.paste(gpu_src, (0, 0))

                t_cpu = timeit(cpu_paste)
                t_gpu = timeit(gpu_paste)
            elif cpu_fn == "AFFINE":
                # Identity-ish affine (6 coefficients)
                coeffs = (1.1, 0.1, 0, 0.1, 0.9, 0)
                t_cpu = timeit(
                    lambda: cpu_img.transform(
                        cpu_img.size,
                        Image.Transform.AFFINE,
                        coeffs,
                        Image.Resampling.BILINEAR,
                    )
                )
                t_gpu = timeit(lambda: gpu_img.transform(gpu_img.size, 0, coeffs, 2, 1))
            else:
                t_cpu = timeit(lambda: cpu_fn(cpu_img))
                t_gpu = timeit(lambda: gpu_fn(gpu_img))

            speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
            all_results[bname][s] = {
                "cpu_ms": round(t_cpu, 3),
                "gpu_ms": round(t_gpu, 3),
                "speedup": round(speedup, 2),
            }
            print(
                f"  {s:5d}x{s:<5d} "
                f" CPU: {t_cpu:8.3f}ms  GPU: {t_gpu:8.3f}ms  speedup: {speedup:.2f}x"
            )

    return all_results


def generate_report(results):
    """Generate a Markdown report from benchmark results."""
    lines = []
    lines.append("# Pillow GPU Backend — Comprehensive Benchmark Report\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Python:** {sys.version.split()[0]}\n")
    lines.append(f"**Platform:** {sys.platform}\n")
    lines.append(f"**Image sizes:** {', '.join(f'{s}x{s}' for s in SIZES)}\n")
    lines.append(
        f"**Timing:** median of {REPEATS} runs after {WARMUP} warmup iterations\n"
    )
    lines.append("")

    # Backend info
    try:
        from PIL._imaging_gpu import backend_info

        info = backend_info()
        lines.append(f"**GPU Backend:** {info.get('backend', 'unknown')}\n")
        lines.append(f"**GPU Device:** {info.get('device', 'unknown')}\n")
    except Exception:
        pass
    lines.append("")

    # Transfer table
    lines.append("## 1. Transfer Overhead (CPU ↔ GPU)\n")
    lines.append("| Size | Upload (ms) | Download (ms) | Round-trip (ms) |")
    lines.append("|------|------------|--------------|----------------|")
    if "transfer" in results:
        for s in SIZES:
            d = results["transfer"][s]
            up = d["upload_ms"]
            dn = d["download_ms"]
            rt = d["roundtrip_ms"]
            lines.append(f"| {s}×{s} | {up:.3f} | {dn:.3f} | {rt:.3f} |")
    lines.append("")

    # Group operations by category
    categories = {
        "Basic Operations": [
            "copy",
            "convert_RGB_to_L",
            "convert_L_to_RGB",
            "convert_RGB_to_RGBA",
        ],
        "Resize": [
            "resize_half_nearest",
            "resize_half_bilinear",
            "resize_double_bilinear",
        ],
        "Transpose / Transform": [
            "transpose_flip_lr",
            "transpose_rotate_90",
            "transform_affine",
        ],
        "Blur": [
            "gaussian_blur_r2",
            "gaussian_blur_r5",
            "box_blur_r2",
            "box_blur_r5",
            "unsharp_mask",
        ],
        "Convolution": ["convolve_sharpen"],
        "Point Operations": [
            "point_lut_invert",
            "point_transform",
            "negative",
            "posterize_4",
            "solarize_128",
        ],
        "Compositing": ["blend_0.5", "alpha_composite", "paste"],
        "Statistics / Analysis": [
            "histogram",
            "histogram_RGB",
            "getbbox",
            "getextrema",
        ],
        "Geometry": ["crop_quarter", "expand_10px", "offset"],
        "Channel Operations": ["split", "getchannel_0"],
        "Rank / Mode Filters": [
            "rank_filter_median_3",
            "rank_filter_median_5",
            "mode_filter_3",
        ],
        "Reduction": ["reduce_2x", "reduce_4x"],
        "Effects": ["effect_spread_5"],
        "ImageOps": ["imageops_autocontrast", "imageops_equalize"],
        "ImageEnhance": [
            "enhance_brightness_1.5",
            "enhance_contrast_1.5",
            "enhance_color_0.5",
            "enhance_sharpness_2.0",
        ],
        "Gradients": ["linear_gradient", "radial_gradient"],
    }

    section_num = 2
    for cat_name, ops in categories.items():
        valid_ops = [o for o in ops if o in results]
        if not valid_ops:
            continue
        lines.append(f"## {section_num}. {cat_name}\n")

        # Build table
        header = "| Operation |"
        for s in SIZES:
            header += f" {s}² CPU | {s}² GPU | {s}² × |"
        lines.append(header)

        sep = "|-----------|"
        for _ in SIZES:
            sep += "-------:|-------:|------:|"
        lines.append(sep)

        for op in valid_ops:
            row = f"| {op} |"
            for s in SIZES:
                d = results[op].get(s, {})
                cpu_ms = d.get("cpu_ms", 0)
                gpu_ms = d.get("gpu_ms", 0)
                speedup = d.get("speedup", 0)
                marker = "**" if speedup >= 2.0 else ""
                row += f" {cpu_ms:.2f} | {gpu_ms:.2f} |"
                row += f" {marker}{speedup:.1f}x{marker} |"
            lines.append(row)

        lines.append("")
        section_num += 1

    # Summary: crossover analysis
    lines.append(f"## {section_num}. Crossover Analysis\n")
    lines.append(
        "The crossover point is the smallest image size where "
        "GPU compute (excluding transfer) is faster than CPU.\n"
    )
    lines.append("| Operation | Crossover Size | Notes |")
    lines.append("|-----------|---------------|-------|")

    for cat_name, ops in categories.items():
        for op in ops:
            if op not in results or op == "transfer":
                continue
            crossover = "Never (CPU faster)"
            note = ""
            for s in SIZES:
                d = results[op].get(s, {})
                if d.get("speedup", 0) > 1.0:
                    crossover = f"{s}×{s}"
                    sp = d.get("speedup", 0)
                    note = f"{sp:.1f}x speedup"
                    break
            # Check if always faster
            all_faster = all(
                results[op].get(s, {}).get("speedup", 0) > 1.0 for s in SIZES
            )
            if all_faster:
                note += " (GPU always faster)"
            lines.append(f"| {op} | {crossover} | {note} |")

    lines.append("")

    # Legend
    lines.append(f"## {section_num + 1}. Legend\n")
    lines.append("- **CPU (ms)**: Median execution time using CPU-only PIL")
    lines.append(
        "- **GPU (ms)**: Median execution time using GPU compute only "
        "(data already on GPU)"
    )
    lines.append(
        "- **×**: Speedup factor (CPU time / GPU time). "
        "Values > 1.0 mean GPU is faster"
    )
    lines.append("- **Bold** speedup values indicate ≥ 2× improvement")
    lines.append(
        "- Transfer overhead (Section 1) is NOT included in GPU times. "
        "For end-to-end comparison, add the round-trip transfer time to GPU times"
    )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Pillow GPU Benchmark Suite")
    print("=" * 60)

    results = run_benchmarks()

    # Save raw JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate report
    report = generate_report(results)
    with open("GPU_BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("  Raw data: benchmark_results.json")
    print("  Report:   GPU_BENCHMARK_REPORT.md")
