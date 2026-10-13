"""Regression tests for the visual-RGB boundary used by fire analysis."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from satellite_analyzer.core.exceptions import InvalidImageError
from satellite_analyzer.io.image_loader import ImageLoader
from satellite_analyzer.io.rgb_loader import RGBImageLoader


def _write_tiff(
    path: Path,
    pixels: np.ndarray,
    interpretation: tuple[ColorInterp, ...],
    *,
    nodata: int | None = None,
) -> None:
    """Write band-first synthetic data with an explicit band interpretation."""

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=pixels.shape[2],
        height=pixels.shape[1],
        count=pixels.shape[0],
        dtype=pixels.dtype,
        transform=from_origin(10, 20, 1, 1),
        nodata=nodata,
    ) as dataset:
        dataset.write(pixels)
        dataset.colorinterp = interpretation


def test_rgb_loader_keeps_original_pixels_and_metadata(png_path: Path) -> None:
    """The detection adapter must not rewrite the file or native I/O data."""

    before = png_path.read_bytes()
    native = ImageLoader().load(png_path)
    original = native.array.copy()

    loaded = RGBImageLoader().load(png_path)

    assert loaded.rgb.dtype == np.uint8
    assert loaded.rgb.shape == (64, 64, 3)
    np.testing.assert_array_equal(loaded.rgb, original)
    assert loaded.valid_mask.dtype == np.bool_
    assert loaded.valid_mask.all()
    assert loaded.metadata.width == native.width
    assert loaded.metadata.height == native.height
    assert loaded.metadata.channels == native.channels
    assert loaded.path == png_path
    assert png_path.read_bytes() == before
    np.testing.assert_array_equal(native.array, original)


def test_palette_colors_and_transparency_are_decoded(tmp_path: Path) -> None:
    """Palette indices are not brightness, and hidden colors are invalid."""

    path = tmp_path / "palette.png"
    image = Image.new("P", (3, 1))
    image.putpalette([255, 150, 10, 30, 60, 90] + [0] * (768 - 6))
    image.putdata([0, 1, 0])
    image.save(path, transparency=0)

    loaded = RGBImageLoader().load(path)

    np.testing.assert_array_equal(
        loaded.rgb, np.array([[[255, 150, 10], [30, 60, 90], [255, 150, 10]]])
    )
    np.testing.assert_array_equal(loaded.valid_mask, [[False, True, False]])
    assert loaded.metadata.channels == 1
    assert ImageLoader().load(path).array.ndim == 2


@pytest.mark.parametrize("mode", ["L", "LA"])
def test_grayscale_expands_without_changing_metadata(tmp_path: Path, mode: str) -> None:
    """Grayscale gets neutral RGB values while reported channels remain native."""

    path = tmp_path / f"grayscale_{mode}.png"
    gray = np.array([[10, 230]], dtype=np.uint8)
    pixels = gray if mode == "L" else np.dstack((gray, [[0, 255]]))
    Image.fromarray(pixels.astype(np.uint8), mode=mode).save(path)

    loaded = RGBImageLoader().load(path)

    np.testing.assert_array_equal(loaded.rgb, np.repeat(gray[..., None], 3, axis=2))
    expected_valid = [[True, True]] if mode == "L" else [[False, True]]
    np.testing.assert_array_equal(loaded.valid_mask, expected_valid)
    assert loaded.metadata.channels == (1 if mode == "L" else 2)


def test_rgba_retains_rgb_and_excludes_zero_alpha(tmp_path: Path) -> None:
    """Transparent hot-looking RGB values must not become valid candidates."""

    path = tmp_path / "transparent.png"
    pixels = np.array(
        [[[255, 150, 5, 0], [255, 150, 5, 255], [10, 20, 30, 128]]],
        dtype=np.uint8,
    )
    Image.fromarray(pixels).save(path)

    loaded = RGBImageLoader().load(path)

    np.testing.assert_array_equal(loaded.rgb, pixels[..., :3])
    np.testing.assert_array_equal(loaded.valid_mask, [[False, True, True]])
    assert loaded.metadata.channels == 4


@pytest.mark.parametrize("kind", ["uint16_png", "cmyk_jpeg", "float_tiff"])
def test_rejects_unsupported_visual_color_or_pixel_scale(
    tmp_path: Path, kind: str
) -> None:
    """Unsupported color spaces and numeric scales must not silently become RGB."""

    if kind == "uint16_png":
        path = tmp_path / "high_depth.png"
        Image.fromarray(np.full((3, 3), 4095, dtype=np.uint16)).save(path)
    elif kind == "cmyk_jpeg":
        path = tmp_path / "cmyk.jpg"
        Image.new("CMYK", (3, 3), (0, 120, 250, 0)).save(path)
    else:
        path = tmp_path / "float.tif"
        _write_tiff(
            path,
            np.full((3, 3, 3), 0.5, dtype=np.float32),
            (ColorInterp.red, ColorInterp.green, ColorInterp.blue),
        )

    with pytest.raises(InvalidImageError):
        RGBImageLoader().load(path)


def test_rejects_image_over_pixel_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader enforces its input-size limit before decoding pixel data."""

    path = tmp_path / "large.png"
    Image.new("RGB", (11, 11)).save(path)

    def fail_if_decoded(*args: object, **kwargs: object) -> None:
        pytest.fail("Oversized input reached pixel decoding")

    monkeypatch.setattr(Image.Image, "load", fail_if_decoded)

    with pytest.raises(InvalidImageError):
        RGBImageLoader(max_pixels=120).load(path)


@pytest.mark.parametrize("budget", [0, -1])
def test_rejects_nonpositive_pixel_budget(budget: int) -> None:
    """An invalid resource budget receives a clear configuration error."""

    with pytest.raises(ValueError):
        RGBImageLoader(max_pixels=budget)


def test_named_tiff_bands_are_reordered_to_rgb(tmp_path: Path) -> None:
    """RGB channel order comes from color interpretation, not band positions."""

    path = tmp_path / "reordered.tif"
    bands = np.array([[[150, 40]], [[5, 80]], [[255, 20]]], dtype=np.uint8)
    _write_tiff(path, bands, (ColorInterp.green, ColorInterp.blue, ColorInterp.red))

    loaded = RGBImageLoader().load(path)

    np.testing.assert_array_equal(loaded.rgb, [[[255, 150, 5], [20, 40, 80]]])
    assert loaded.valid_mask.all()
    assert loaded.metadata.raster is not None
    assert loaded.metadata.raster["band_count"] == 3


def test_tiff_nodata_and_grayscale_are_handled(tmp_path: Path) -> None:
    """Nodata remains invalid even when the grayscale raster expands to RGB."""

    path = tmp_path / "grayscale.tif"
    bands = np.array([[[0, 90, 220]]], dtype=np.uint8)
    _write_tiff(path, bands, (ColorInterp.gray,), nodata=0)

    loaded = RGBImageLoader().load(path)

    np.testing.assert_array_equal(loaded.valid_mask, [[False, True, True]])
    np.testing.assert_array_equal(loaded.rgb[..., 0], bands[0])
    np.testing.assert_array_equal(loaded.rgb[..., 0], loaded.rgb[..., 1])
    assert loaded.metadata.channels == 1


def test_tiff_alpha_masks_hidden_hot_pixels(tmp_path: Path) -> None:
    """Named TIFF alpha has the same validity semantics as PNG transparency."""

    path = tmp_path / "rgba.tif"
    pixels = np.array([[[255, 150, 5, 0], [255, 150, 5, 255]]], dtype=np.uint8)
    _write_tiff(
        path,
        np.moveaxis(pixels, -1, 0),
        (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha),
    )

    loaded = RGBImageLoader().load(path)

    np.testing.assert_array_equal(loaded.valid_mask, [[False, True]])
    np.testing.assert_array_equal(loaded.rgb, pixels[..., :3])


@pytest.mark.parametrize("bands", [2, 3, 5])
def test_rejects_tiff_without_rgb_band_interpretation(
    tmp_path: Path, bands: int
) -> None:
    """Unspecified multispectral bands cannot safely be interpreted as RGB."""

    path = tmp_path / f"ambiguous_{bands}.tif"
    _write_tiff(
        path,
        np.full((bands, 2, 3), 100, dtype=np.uint8),
        (ColorInterp.undefined,) * bands,
    )

    with pytest.raises(InvalidImageError):
        RGBImageLoader().load(path)


def test_corrupted_visual_image_gets_domain_error(tmp_path: Path) -> None:
    """A supported suffix does not make invalid bytes a valid image."""

    path = tmp_path / "corrupted.png"
    path.write_bytes(b"not image data")

    with pytest.raises(InvalidImageError):
        RGBImageLoader().load(path)
