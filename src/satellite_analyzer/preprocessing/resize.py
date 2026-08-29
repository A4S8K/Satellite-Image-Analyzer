"""Image resizing utilities."""

import cv2
import numpy as np

from satellite_analyzer.core.exceptions import ImageProcessingError


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to an explicit width and height using area interpolation."""

    if width <= 0 or height <= 0:
        raise ImageProcessingError("Resize width and height must be positive integers.")
    if image.ndim not in {2, 3}:
        raise ImageProcessingError("Only 2D or 3D image arrays can be resized.")
    try:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    except cv2.error as error:
        raise ImageProcessingError(f"Could not resize image: {error}") from error
