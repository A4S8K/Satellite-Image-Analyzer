"""Pixel value normalization utilities."""

import numpy as np

from satellite_analyzer.core.exceptions import ImageProcessingError


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Return a float32 image scaled to the inclusive range 0.0 through 1.0."""

    if not np.issubdtype(image.dtype, np.number):
        raise ImageProcessingError("Only numeric image arrays can be normalized.")

    values = image.astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
    else:
        maximum = float(np.nanmax(values)) if values.size else 0.0
        scale = 255.0 if maximum > 1.0 else 1.0

    if scale <= 0:
        raise ImageProcessingError(
            "Image data type does not have a positive normalization range."
        )
    return np.clip(values / scale, 0.0, 1.0)
