"""Tests for standalone operations and the preprocessing pipeline."""

from pathlib import Path

import numpy as np

from satellite_analyzer.core.config import PreprocessingSettings
from satellite_analyzer.io.image_loader import ImageLoader
from satellite_analyzer.preprocessing.filters import reduce_noise
from satellite_analyzer.preprocessing.normalization import normalize_image
from satellite_analyzer.preprocessing.pipeline import PreprocessingPipeline
from satellite_analyzer.preprocessing.resize import resize_image


def test_resize_image(rgb_array: np.ndarray) -> None:
    """Resize returns the requested dimensions."""

    resized = resize_image(rgb_array, width=32, height=16)

    assert resized.shape == (16, 32, 3)


def test_normalize_image(rgb_array: np.ndarray) -> None:
    """uint8 pixels are converted into float values from zero through one."""

    normalized = normalize_image(rgb_array)

    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized[0, 0, 0] == 0.0


def test_reduce_noise(rgb_array: np.ndarray) -> None:
    """Noise reduction retains image shape and data type."""

    denoised = reduce_noise(rgb_array)

    assert denoised.shape == rgb_array.shape
    assert denoised.dtype == rgb_array.dtype


def test_pipeline_execution(png_path: Path) -> None:
    """The pipeline executes resize, normalization, and noise reduction in order."""

    image = ImageLoader().load(png_path)
    settings = PreprocessingSettings(
        target_width=20,
        target_height=10,
        normalize=True,
        denoise=True,
    )
    processed = PreprocessingPipeline(settings).process(image)

    assert processed.array.shape == (10, 20, 3)
    assert processed.array.dtype == np.float32
    assert processed.array.min() >= 0.0
    assert processed.array.max() <= 1.0
