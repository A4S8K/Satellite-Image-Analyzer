"""Reusable synthetic imagery fixtures."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin


@pytest.fixture()
def rgb_array() -> np.ndarray:
    """Return a deterministic 64 by 64 RGB array."""

    rows, columns = np.indices((64, 64))
    return np.stack((rows * 4, columns * 4, (rows + columns) * 2), axis=-1).astype(
        np.uint8
    )


@pytest.fixture()
def png_path(tmp_path: Path, rgb_array: np.ndarray) -> Path:
    """Create a small PNG image."""

    path = tmp_path / "sample.png"
    Image.fromarray(rgb_array).save(path)
    return path


@pytest.fixture()
def jpg_path(tmp_path: Path, rgb_array: np.ndarray) -> Path:
    """Create a small JPEG image."""

    path = tmp_path / "sample.jpg"
    Image.fromarray(rgb_array).save(path)
    return path


@pytest.fixture()
def tiff_path(tmp_path: Path, rgb_array: np.ndarray) -> Path:
    """Create a small, georeferenced three-band TIFF image."""

    path = tmp_path / "sample.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rgb_array.shape[0],
        width=rgb_array.shape[1],
        count=3,
        dtype=rgb_array.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 64, 1, 1),
    ) as dataset:
        dataset.write(np.moveaxis(rgb_array, -1, 0))
    return path
