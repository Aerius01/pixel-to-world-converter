"""Pixel to World Converter - Convert pixel tracks to GPS coordinates using drone sensor data."""

from .converter import (
    convert,
    compute_homographies,
    PixelToWorldError,
    ValidationError,
    VideoError,
    DataLoadError
)
from .pipeline import convert_pixel_to_world, build_frame_homographies
from .projection import project_bbox

__version__ = "1.0.0"
__all__ = [
    'convert',
    'compute_homographies',
    'convert_pixel_to_world',
    'build_frame_homographies',
    'project_bbox',
    'PixelToWorldError',
    'ValidationError',
    'VideoError',
    'DataLoadError',
]
