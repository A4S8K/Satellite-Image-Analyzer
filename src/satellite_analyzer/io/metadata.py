"""Metadata extraction for loaded image data."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel
import rasterio
from rasterio.errors import RasterioError

from satellite_analyzer.io.image_loader import ImageData


class ImageMetadata(BaseModel):
    """Portable metadata describing an image and optional raster details."""

    filename: str
    format: str
    width: int
    height: int
    channels: int
    dtype: str
    file_size: int
    raster: dict[str, Any] | None = None


def extract_metadata(image: ImageData) -> ImageMetadata:
    """Extract standard metadata and available TIFF/GeoTIFF raster metadata."""

    return ImageMetadata(
        filename=image.path.name,
        format=image.format,
        width=image.width,
        height=image.height,
        channels=image.channels,
        dtype=str(image.dtype),
        file_size=image.path.stat().st_size,
        raster=_extract_raster_metadata(image.path),
    )


def _extract_raster_metadata(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    try:
        with rasterio.open(path) as dataset:
            return {
                "driver": dataset.driver,
                "band_count": dataset.count,
                "crs": str(dataset.crs) if dataset.crs else None,
                "transform": list(dataset.transform) if dataset.transform else None,
                "bounds": list(dataset.bounds),
            }
    except (RasterioError, OSError, ValueError):
        return None
