"""
PIL.gpu.ImagePalette — Re-export from PIL.

Palette handling is a CPU-only concept (paletted images are converted
to full-color before GPU upload), so this module simply re-exports
PIL.ImagePalette.
"""

from PIL.ImagePalette import *  # noqa: F401,F403
from PIL.ImagePalette import ImagePalette  # noqa: F401
