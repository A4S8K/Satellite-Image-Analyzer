"""Tests for supported image loading and validation."""

from pathlib import Path

import pytest

from satellite_analyzer.core.exceptions import InvalidImageError, UnsupportedFormatError
from satellite_analyzer.io.image_loader import ImageLoader


@pytest.mark.parametrize("fixture_name", ["png_path", "jpg_path", "tiff_path"])
def test_loads_supported_images(request: pytest.FixtureRequest, fixture_name: str) -> None:
    """PNG, JPEG, and TIFF files decode into a consistent representation."""

    image_path = request.getfixturevalue(fixture_name)
    image = ImageLoader().load(image_path)

    assert image.width == 64
    assert image.height == 64
    assert image.channels == 3
    assert str(image.dtype) == "uint8"


def test_rejects_missing_file(tmp_path: Path) -> None:
    """A non-existent path receives an actionable custom error."""

    with pytest.raises(InvalidImageError, match="does not exist"):
        ImageLoader().load(tmp_path / "missing.png")


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported extensions are rejected before attempting to decode."""

    path = tmp_path / "image.bmp"
    path.write_bytes(b"not an image")

    with pytest.raises(UnsupportedFormatError, match="Supported formats"):
        ImageLoader().load(path)


def test_rejects_corrupted_file(tmp_path: Path) -> None:
    """Unreadable bytes with a supported suffix receive a custom error."""

    path = tmp_path / "corrupted.png"
    path.write_bytes(b"this is not a PNG file")

    with pytest.raises(InvalidImageError, match="Could not decode"):
        ImageLoader().load(path)
