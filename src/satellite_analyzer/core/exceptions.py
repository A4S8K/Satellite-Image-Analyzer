"""Domain-specific errors for Satellite Image Analyzer."""


class SatelliteAnalyzerError(Exception):
    """Base class for application errors shown to users."""


class InvalidImageError(SatelliteAnalyzerError):
    """Raised when an image path is invalid or its contents cannot be decoded."""


class UnsupportedFormatError(SatelliteAnalyzerError):
    """Raised when an image extension is not supported."""


class ImageProcessingError(SatelliteAnalyzerError):
    """Raised when a preprocessing or output operation cannot complete."""
