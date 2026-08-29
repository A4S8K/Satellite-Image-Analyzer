"""Conservative noise reduction filters."""

import cv2
import numpy as np

from satellite_analyzer.core.exceptions import ImageProcessingError


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """Apply a small Gaussian blur that reduces noise while retaining structure."""

    if image.ndim not in {2, 3}:
        raise ImageProcessingError("Only 2D or 3D image arrays can be denoised.")
    try:
        return cv2.GaussianBlur(image, (3, 3), sigmaX=0)
    except cv2.error as error:
        raise ImageProcessingError(f"Could not reduce image noise: {error}") from error
