GPU Acceleration
================

.. note::

   The GPU backend is a **first implementation** — fully functional and already
   well-optimised for real-world workloads, but not yet a stable API.
   Method signatures and internal structures **may change** in future releases as
   the design converges with upstream discussions (see :issue:`1546`).
   Contributions, bug reports, and architecture feedback are very welcome.

Pillow includes an optional GPU-accelerated backend that offloads image
processing to the graphics card via **OpenCL** (any vendor) or
**CUDA** (NVIDIA). The CPU code-path is never touched: ``from PIL import Image``
continues to work exactly as before. The GPU API is accessed through the
parallel ``PIL.gpu`` namespace.

Platform support
----------------

+------------------+----------+------+------------------------------------------+
| Platform         | OpenCL   | CUDA | Notes                                    |
+==================+==========+======+==========================================+
| **Linux**        | ✓        | ✓    | Recommended. Any GPU with an ICD driver  |
|                  |          |      | works (AMD, Intel, NVIDIA, etc.).        |
+------------------+----------+------+------------------------------------------+
| **Windows**      | ✓        | ✓    | Intel oneAPI, AMD ROCm-OCL, or NVIDIA    |
|                  |          |      | GPU Toolkit. Tested on Intel Iris Xe.    |
+------------------+----------+------+------------------------------------------+
| **macOS**        | ⚠        | ✗    | Apple ships a system OpenCL 1.2 runtime  |
|                  |          |      | (deprecated since 10.14, still present). |
|                  |          |      | NVIDIA CUDA is not available on macOS.   |
+------------------+----------+------+------------------------------------------+

Building with GPU support
--------------------------

OpenCL (any platform)
~~~~~~~~~~~~~~~~~~~~~

OpenCL is a vendor-neutral standard.  The runtime (``libOpenCL.so`` /
``OpenCL.dll``) ships with every modern GPU driver — **no vendor SDK is
required**.  For building you only need the standard Khronos headers and
the ICD loader.

.. tab:: Linux (apt)

   .. code-block:: shell

      # Headers + ICD loader (vendor-agnostic)
      sudo apt install opencl-headers ocl-icd-opencl-dev

      pip install --upgrade .

.. tab:: Linux (dnf/yum)

   .. code-block:: shell

      sudo dnf install opencl-headers ocl-icd-devel
      pip install --upgrade .

.. tab:: Windows

   ``OpenCL.dll`` is already present in ``System32`` with any modern Intel,
   AMD, or NVIDIA driver.  You only need the Khronos OpenCL headers for
   compilation.  Install the `Khronos OpenCL SDK
   <https://github.com/KhronosGroup/OpenCL-SDK/releases>`_ (or any package
   that provides ``CL/cl.h``), then::

      pip install --upgrade .

.. tab:: macOS

   .. code-block:: shell

      # OpenCL framework + headers ship with Xcode Command Line Tools
      xcode-select --install
      pip install --upgrade .

CUDA (NVIDIA only, Linux or Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA is an NVIDIA-specific technology.  Install the
`CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_
(version 11.0 or newer), which provides ``cuda.h``, ``libcuda``, and
``libnvrtc``:

.. code-block:: shell

   pip install --upgrade .

The build system detects ``cuda.h``, ``libcuda``, and ``libnvrtc``
automatically.  To point it at a non-standard installation directory::

   CUDA_ROOT=/path/to/cuda pip install --upgrade .

Verifying the build
~~~~~~~~~~~~~~~~~~~

After installation, confirm which backends were compiled::

   python -c "
   from PIL import features
   print('OpenCL:', features.check('opencl'))
   print('CUDA  :', features.check('cuda'))
   "

If both return ``False``, OpenCL/CUDA headers were not found during the build.
Re-read the instructions above and rebuild with ``pip install --no-build-isolation --upgrade .``.

Quick-start
-----------

The GPU API mirrors the standard :py:mod:`PIL.Image` API as closely as
possible, so existing PIL code is easy to migrate::

   from PIL.gpu import Image as GpuImage
   from PIL.gpu import ImageEnhance, ImageOps

   # Upload once
   with open("photo.jpg", "rb") as f:
       cpu_img = __import__("PIL.Image", fromlist=["open"]).open(f)
   gpu_img = GpuImage.from_cpu(cpu_img)

   # Process entirely on GPU — no CPU round-trips
   gpu_img = gpu_img.resize((1920, 1080), resample=2)          # bilinear
   gpu_img = ImageEnhance.Contrast(gpu_img).enhance(1.4)
   gpu_img = gpu_img.filter(__import__("PIL.ImageFilter",
       fromlist=["SHARPEN"]).SHARPEN)

   # Download only when you need pixels back
   result = gpu_img.to_cpu()
   result.save("output.jpg")

GPU images and CPU images are never mixed automatically — an explicit
:py:meth:`~PIL.gpu.Image.Image.from_cpu` / :py:meth:`~PIL.gpu.Image.Image.to_cpu`
call is always required.

Supported operations
--------------------

The table below lists every operation that has a native GPU kernel.
"CPU fallback" means the operation downloads to RAM, runs on CPU, and
re-uploads (transparent but slower for large images).

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Status
     - Notes
   * - ``copy``
     - GPU
     - Per-pixel parallel memcopy
   * - ``convert`` (RGB↔L, RGB↔RGBA, RGB↔YCbCr, …)
     - GPU
     - Parallel colour-space transform
   * - ``resize``
     - GPU
     - Nearest, bilinear, bicubic, Lanczos
   * - ``transpose`` / ``rotate``
     - GPU
     - All 7 :py:class:`~PIL.Image.Transpose` variants
   * - ``transform`` (affine & perspective)
     - GPU
     - 6-coefficient affine; 8-coefficient perspective
   * - ``gaussian_blur``
     - GPU
     - Separable per-pixel kernel
   * - ``box_blur``
     - GPU
     - Per-pixel parallel (horizontal + vertical pass)
   * - ``unsharp_mask``
     - GPU
     - Gaussian blur + difference
   * - ``filter`` (convolution kernels)
     - GPU
     - SHARPEN, SMOOTH, EDGE_ENHANCE, EMBOSS, etc.
   * - ``filter`` (rank filters)
     - GPU
     - MedianFilter, MinFilter, MaxFilter
   * - ``filter`` (ModeFilter)
     - GPU
     - Histogram-based mode per neighbourhood
   * - ``point`` / ``point_transform``
     - GPU
     - LUT (256-entry) and scale/offset
   * - ``negative`` / ``posterize`` / ``solarize``
     - GPU
     - Per-pixel parallel
   * - ``histogram``
     - GPU
     - Parallel reduce with shared-memory atomics
   * - ``getbbox`` / ``getextrema``
     - GPU
     - Parallel reduce
   * - ``crop`` / ``expand`` / ``offset``
     - GPU
     - Per-pixel copy with coordinate mapping
   * - ``split`` / ``getchannel``
     - GPU
     - Per-pixel channel extraction
   * - ``paste`` / ``alpha_composite`` / ``blend``
     - GPU
     - Per-pixel with optional alpha
   * - ``effect_spread``
     - GPU
     - Per-pixel stochastic scatter
   * - ``reduce``
     - GPU
     - Box-averaging downsample
   * - ``ImageEnhance.Brightness/Contrast/Color/Sharpness``
     - GPU
     - All four enhancers have native kernels
   * - ``ImageOps.autocontrast`` / ``equalize``
     - GPU
     - Histogram-based, fully on GPU
   * - ``linear_gradient`` / ``radial_gradient``
     - GPU
     - Generation entirely on GPU (no CPU source needed)
   * - ``ImageDraw``, ``ImageMath``, ``ImageMorph``
     - CPU fallback
     - Planned for a future iteration

Performance
-----------

GPU acceleration is most effective when the image is large enough that the
parallel work outweighs the kernel-launch overhead (~0.1–0.5 ms on typical
hardware).  General guidelines:

**Operations that scale very well** (embarrassingly parallel, all pixels
independent):

- Affine / perspective transform
- Resize (bilinear, bicubic)
- Alpha composite, blend, paste
- ImageEnhance (brightness, contrast, colour, sharpness)
- Convolution kernels (SHARPEN, SMOOTH, EMBOSS, …)
- Point operations (LUT, scale/offset, negative, posterize, solarize)
- Histogram (parallel reduce)
- effect_spread

**Operations that scale moderately** (parallel but memory-bandwidth bound
or multi-pass):

- Box blur, gaussian blur, unsharp mask
- Mode filter / rank filter
- Convert, copy, transpose, offset, crop, expand

**Operations where the GPU advantage is smaller** (output is tiny relative
to input, or the algorithm has low arithmetic intensity):

- ``getbbox``, ``getextrema`` (reduction to a single value)
- ``reduce`` (downsamples aggressively, output is small)
- Large-radius gaussian blur (dominated by memory bandwidth)

**Transfer overhead**: uploading/downloading an image to/from VRAM takes
time proportional to the pixel count.  The GPU delivers the best net
speedup when multiple operations are chained *without* round-tripping
through CPU memory — see `Keeping data on the GPU`_ below.

To measure concrete timings on your own hardware, run the benchmark suite
included in the repository::

   python Tests/benchmark_gpu.py

Keeping data on the GPU
-----------------------

The GPU delivers the biggest gains when a pipeline performs **multiple
operations without round-tripping to CPU**::

   # Efficient — three GPU kernels, one upload, one download
   img = GpuImage.from_cpu(cpu_img)              # upload once
   img = img.resize((w, h), resample=2)
   img = ImageEnhance.Brightness(img).enhance(1.2)
   img = img.filter(ImageFilter.SHARPEN)
   result = img.to_cpu()                         # download once

   # Less efficient — three separate uploads/downloads
   cpu_img = cpu_img.resize((w, h), ...)         # CPU
   cpu_img = ImageEnhance.Brightness(cpu_img).enhance(1.2)   # CPU
   cpu_img = cpu_img.filter(ImageFilter.SHARPEN)              # CPU

Backend selection
-----------------

At import time Pillow tries CUDA first (if compiled in), then OpenCL.
You can check and override::

   from PIL.gpu._backend import get_backend_name, set_preferred_backend

   print(get_backend_name())          # "cuda", "opencl", or "cpu"
   set_preferred_backend("opencl")    # force OpenCL even if CUDA is available

If neither CUDA nor OpenCL was compiled in, all ``PIL.gpu`` operations
fall back to the CPU transparently.

Contributing
------------

The GPU backend lives under ``src/libImaging/gpu/`` (C kernels) and
``src/PIL/gpu/`` (Python API). New kernel contributions should:

1. Add both an **OpenCL** (``GpuOpenCL.c``) and a **CUDA** (``GpuCuda.c``)
   implementation, or clearly document why one is not provided.
2. Include a test in ``Tests/test_gpu.py`` that compares GPU output to the
   reference CPU output within a small tolerance.
3. Update the operation table in this document.

For architecture discussions and roadmap, refer to :issue:`1546`.
