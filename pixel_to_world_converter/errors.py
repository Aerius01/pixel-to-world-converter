"""Exceptions raised by the pixel-to-world conversion."""


class PixelToWorldError(Exception):
    """Base exception for all pixel-to-world conversion errors."""
    pass


class ValidationError(PixelToWorldError):
    """Raised when input file validation fails (missing files, permissions, etc.)."""
    pass


class VideoError(PixelToWorldError):
    """Raised when video file cannot be opened or read."""
    pass


class DataLoadError(PixelToWorldError):
    """Raised when CSV data is missing required columns or contains invalid values."""
    pass
