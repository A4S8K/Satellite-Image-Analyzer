"""Tests for image metadata extraction."""

from pathlib import Path

from satellite_analyzer.io.image_loader import ImageLoader
from satellite_analyzer.io.metadata import extract_metadata


def test_extracts_standard_metadata(png_path: Path) -> None:
    """Standard properties are available for a loaded PNG."""

    metadata = extract_metadata(ImageLoader().load(png_path))

    assert metadata.filename == "sample.png"
    assert metadata.format == "PNG"
    assert metadata.width == 64
    assert metadata.height == 64
    assert metadata.channels == 3
    assert metadata.dtype == "uint8"
    assert metadata.file_size > 0
    assert metadata.raster is None


def test_extracts_tiff_raster_metadata(tiff_path: Path) -> None:
    """Available GeoTIFF properties are surfaced without coordinate conversion."""

    metadata = extract_metadata(ImageLoader().load(tiff_path))

    assert metadata.format == "GTIFF"
    assert metadata.raster is not None
    assert metadata.raster["band_count"] == 3
    assert metadata.raster["crs"] == "EPSG:4326"
