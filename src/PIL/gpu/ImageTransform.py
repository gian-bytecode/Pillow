"""
PIL.gpu.ImageTransform — Transform classes for GPU images.

Mirrors PIL.ImageTransform constants and classes.
"""

from __future__ import annotations

from PIL.Image import Transform

# Transform method constants (matching PIL)
AFFINE = Transform.AFFINE
EXTENT = Transform.EXTENT
PERSPECTIVE = Transform.PERSPECTIVE
QUAD = Transform.QUAD
MESH = Transform.MESH


class AffineTransform:
    method = AFFINE

    def __init__(self, data):
        self.data = data


class ExtentTransform:
    method = EXTENT

    def __init__(self, data):
        self.data = data


class PerspectiveTransform:
    method = PERSPECTIVE

    def __init__(self, data):
        self.data = data


class QuadTransform:
    method = QUAD

    def __init__(self, data):
        self.data = data
