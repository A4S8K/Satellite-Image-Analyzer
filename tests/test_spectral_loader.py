"""Small synthetic GeoTIFFs exercise calibration, masking, and spatial checks."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from satellite_analyzer.core.exceptions import SatelliteAnalyzerError
from satellite_analyzer.core.ndvi_config import NDVISettings
from satellite_analyzer.io.spectral_loader import SpectralBandLoader
from tests.test_radiometry import _landsat_mtl, _sentinel_xml

_GRID = from_origin(500_000, 5_500_000, 10, 10)


def _write(
    path: Path,
    pixels: np.ndarray,
    *,
    transform: Affine = _GRID,
    crs: str | None = "EPSG:32643",
    nodata: float | None = None,
    scales: tuple[float, ...] | None = None,
    offsets: tuple[float, ...] | None = None,
    descriptions: tuple[str, ...] | None = None,
    colorinterp: tuple[ColorInterp, ...] | None = None,
) -> Path:
    bands = pixels[np.newaxis] if pixels.ndim == 2 else pixels
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands.shape[2],
        height=bands.shape[1],
        count=bands.shape[0],
        dtype=bands.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as output:
        output.write(bands)
        if scales is not None:
            output.scales = scales
        if offsets is not None:
            output.offsets = offsets
        if descriptions is not None:
            output.descriptions = descriptions
        if colorinterp is not None:
            output.colorinterp = colorinterp
    return path


def _pair(tmp_path: Path, **kwargs: Any) -> tuple[Path, Path]:
    red = _write(tmp_path / "red.tif", np.full((4, 4), 0.2, np.float32), **kwargs)
    nir = _write(tmp_path / "nir.tif", np.full((4, 4), 0.6, np.float32), **kwargs)
    return red, nir


def test_separate_float_bands_preserve_grid_and_provenance(tmp_path: Path) -> None:
    red_path, nir_path = _pair(tmp_path)
    original = red_path.read_bytes()
    pair = SpectralBandLoader().load(red_path=red_path, nir_path=nir_path)
    assert pair.red.dtype == pair.nir.dtype == np.float32
    assert pair.red.shape == pair.nir.shape == pair.valid_mask.shape == (4, 4)
    assert pair.valid_mask.all()
    assert np.allclose(pair.red, 0.2) and np.allclose(pair.nir, 0.6)
    assert pair.crs == rasterio.CRS.from_epsg(32643)
    assert pair.transform == _GRID
    assert pair.sensor == "custom"
    assert not pair.alignment["applied"]
    assert pair.red_metadata["calibration"]["source"] == "inferred_float_reflectance"
    assert pair.red_metadata["path"] == str(red_path.resolve())
    json.dumps(pair.red_metadata, allow_nan=False)
    json.dumps(pair.nir_metadata, allow_nan=False)
    json.dumps(pair.alignment, allow_nan=False)
    assert red_path.read_bytes() == original
    assert not pair.red.flags.writeable
    assert not pair.nir.flags.writeable
    assert not pair.valid_mask.flags.writeable


def test_rotated_same_grid_is_supported(tmp_path: Path) -> None:
    grid = _GRID * Affine.rotation(15)
    red_path, nir_path = _pair(tmp_path, transform=grid)
    pair = SpectralBandLoader().load(red_path=red_path, nir_path=nir_path)
    assert pair.transform == grid
    assert not pair.alignment["applied"]


def test_nodata_nan_infinity_and_negative_reflectance_masked(tmp_path: Path) -> None:
    values = np.array([[0.2, -999], [np.nan, np.inf], [-0.1, 0.0]], np.float32)
    red = _write(tmp_path / "red.tif", values, nodata=-999)
    nir = _write(tmp_path / "nir.tif", np.full(values.shape, 0.6, np.float32))
    pair = SpectralBandLoader(NDVISettings(input_units="reflectance")).load(
        red_path=red, nir_path=nir
    )
    assert np.array_equal(
        pair.valid_mask, [[True, False], [False, False], [False, True]]
    )
    assert np.isnan(pair.red[~pair.valid_mask]).all()
    assert pair.red_metadata["nodata"] == "-999.0"
    assert pair.red_metadata["valid_pixels_before_calibration"] == 3
    assert pair.red_metadata["valid_pixels_after_calibration"] == 2


def test_all_invalid_float_input_remains_validless(tmp_path: Path) -> None:
    red = _write(tmp_path / "red.tif", np.full((3, 3), np.nan, np.float32))
    nir = _write(tmp_path / "nir.tif", np.full((3, 3), np.nan, np.float32))
    pair = SpectralBandLoader().load(red_path=red, nir_path=nir)
    assert not pair.valid_mask.any()
    assert np.isnan(pair.red).all() and np.isnan(pair.nir).all()


def test_tiff_scaling_applies_additive_offset_once(tmp_path: Path) -> None:
    red = _write(
        tmp_path / "red.tif",
        np.full((3, 3), 2000, np.uint16),
        scales=(0.0001,),
        offsets=(-0.1,),
    )
    nir = _write(
        tmp_path / "nir.tif",
        np.full((3, 3), 6000, np.uint16),
        scales=(0.0001,),
        offsets=(-0.1,),
    )
    pair = SpectralBandLoader().load(red_path=red, nir_path=nir)
    assert np.allclose(pair.red, 0.1)
    assert np.allclose(pair.nir, 0.5)
    assert np.allclose((pair.nir - pair.red) / (pair.nir + pair.red), 2 / 3)


def test_declared_reflectance_rejects_tiff_rescaling(tmp_path: Path) -> None:
    red, nir = _pair(tmp_path, scales=(0.0001,))
    with pytest.raises(SatelliteAnalyzerError, match="conflicts"):
        SpectralBandLoader(NDVISettings(input_units="reflectance")).load(
            red_path=red, nir_path=nir
        )


def test_unknown_integer_scale_is_not_assumed_from_sensor(tmp_path: Path) -> None:
    red = _write(tmp_path / "B04.tif", np.full((3, 3), 2000, np.uint16))
    nir = _write(tmp_path / "B08.tif", np.full((3, 3), 6000, np.uint16))
    with pytest.raises(SatelliteAnalyzerError, match="Unknown radiometric scale"):
        SpectralBandLoader(NDVISettings(sensor="sentinel-2")).load(
            red_path=red, nir_path=nir
        )


def test_explicit_sentinel_coefficients_mask_dn_zero(tmp_path: Path) -> None:
    red = _write(tmp_path / "B04.tif", np.array([[0, 2000], [500, 2000]], np.uint16))
    nir = _write(tmp_path / "B08.tif", np.full((2, 2), 6000, np.uint16))
    settings = NDVISettings(
        sensor="sentinel-2",
        red_scale=0.0001,
        red_offset=-0.1,
        nir_scale=0.0001,
        nir_offset=-0.1,
    )
    pair = SpectralBandLoader(settings).load(red_path=red, nir_path=nir)
    assert np.array_equal(pair.valid_mask, [[False, True], [False, True]])
    assert pair.red_metadata["calibration"]["zero_is_nodata"]


def test_multiband_mapping_is_explicit_and_preserves_spectral_names(
    tmp_path: Path,
) -> None:
    pixels = np.stack(
        [np.full((3, 3), 0.8, np.float32), np.full((3, 3), 0.2, np.float32)]
    )
    path = _write(tmp_path / "stack.tif", pixels, descriptions=("B08", "B04"))
    pair = SpectralBandLoader(NDVISettings(sensor="sentinel-2")).load(
        multiband_path=path, red_band=2, nir_band=1
    )
    assert np.allclose(pair.red, 0.2) and np.allclose(pair.nir, 0.8)
    assert pair.red_metadata["band_index"] == 2
    assert pair.nir_metadata["band_index"] == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"red_path": Path("red.tif")},
        {"multiband_path": Path("stack.tif")},
        {"multiband_path": Path("stack.tif"), "red_band": 1, "nir_band": 1},
        {"multiband_path": Path("stack.tif"), "red_band": 0, "nir_band": 2},
        {"multiband_path": Path("stack.tif"), "red_band": True, "nir_band": 2},
        {
            "multiband_path": Path("stack.tif"),
            "red_path": Path("red.tif"),
            "red_band": 1,
            "nir_band": 2,
        },
    ],
)
def test_invalid_input_modes_rejected(arguments: dict[str, Any]) -> None:
    with pytest.raises(SatelliteAnalyzerError):
        SpectralBandLoader().load(**arguments)


def test_mapping_past_band_count_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "stack.tif", np.ones((2, 3, 3), np.float32))
    with pytest.raises(SatelliteAnalyzerError, match="band count"):
        SpectralBandLoader().load(multiband_path=path, red_band=1, nir_band=3)


@pytest.mark.parametrize("hardlink", [False, True])
def test_separate_inputs_cannot_reference_one_physical_band(
    tmp_path: Path, hardlink: bool
) -> None:
    red = _write(tmp_path / "red.tif", np.full((3, 3), 0.2, np.float32))
    nir = red
    if hardlink:
        nir = tmp_path / "nir.tif"
        try:
            nir.hardlink_to(red)
        except OSError as error:
            pytest.skip(f"Filesystem does not support hardlinks: {error}")
    with pytest.raises(SatelliteAnalyzerError, match="distinct physical bands"):
        SpectralBandLoader().load(red_path=red, nir_path=nir)


def test_separate_multiband_and_rgb_nir_inputs_rejected(tmp_path: Path) -> None:
    rgb = _write(
        tmp_path / "rgb.tif",
        np.ones((3, 3, 3), np.uint8),
        colorinterp=(ColorInterp.red, ColorInterp.green, ColorInterp.blue),
    )
    with pytest.raises(SatelliteAnalyzerError, match="one band"):
        SpectralBandLoader().load(red_path=rgb, nir_path=rgb)
    with pytest.raises(SatelliteAnalyzerError, match="RGB channels"):
        SpectralBandLoader().load(multiband_path=rgb, red_band=1, nir_band=3)


@pytest.mark.parametrize("nir_name", ["B03", "B04", "B8A"])
def test_known_wrong_spectral_names_rejected(tmp_path: Path, nir_name: str) -> None:
    red = _write(tmp_path / "B04.tif", np.full((3, 3), 0.2, np.float32))
    nir = _write(tmp_path / f"scene_{nir_name}.tif", np.full((3, 3), 0.6, np.float32))
    with pytest.raises(SatelliteAnalyzerError, match="Known band name"):
        SpectralBandLoader(NDVISettings(sensor="sentinel-2")).load(
            red_path=red, nir_path=nir
        )


@pytest.mark.parametrize("metadata_type", ["sentinel", "landsat"])
def test_metadata_sensor_revalidates_custom_sensor_band_names(
    tmp_path: Path, metadata_type: str
) -> None:
    metadata = (
        _sentinel_xml(tmp_path)
        if metadata_type == "sentinel"
        else _landsat_mtl(tmp_path)
    )
    wrong_nir_name = "B05" if metadata_type == "sentinel" else "B08"
    red = _write(tmp_path / "B04.tif", np.full((3, 3), 10_000, np.uint16))
    nir = _write(tmp_path / f"{wrong_nir_name}.tif", np.full((3, 3), 20_000, np.uint16))
    with pytest.raises(SatelliteAnalyzerError, match="Known band name"):
        SpectralBandLoader(NDVISettings(metadata_path=metadata)).load(
            red_path=red, nir_path=nir
        )


def test_product_sensor_and_global_tags_are_recorded(tmp_path: Path) -> None:
    metadata = _sentinel_xml(tmp_path)
    red = _write(tmp_path / "B04.tif", np.full((3, 3), 2000, np.uint16))
    nir = _write(tmp_path / "B08.tif", np.full((3, 3), 6000, np.uint16))
    with rasterio.open(red, "r+") as output:
        output.update_tags(SYNTHETIC="test-only")
    pair = SpectralBandLoader(NDVISettings(metadata_path=metadata)).load(
        red_path=red, nir_path=nir
    )
    assert pair.sensor == "sentinel-2"
    assert pair.red_metadata["dataset_tags"]["SYNTHETIC"] == "test-only"
    assert pair.red_metadata["tags"]["SYNTHETIC"] == "test-only"
    assert np.allclose(pair.red, 0.1)
    assert np.allclose(pair.nir, 0.51)


@pytest.mark.parametrize("difference", ["half_pixel", "shape", "crs"])
def test_strict_alignment_rejects_mismatched_grids(
    tmp_path: Path, difference: str
) -> None:
    red = _write(tmp_path / "red.tif", np.full((4, 4), 0.2, np.float32))
    transform = (
        _GRID * Affine.translation(0.5, 0) if difference == "half_pixel" else _GRID
    )
    shape = (2, 2) if difference == "shape" else (4, 4)
    crs = "EPSG:32642" if difference == "crs" else "EPSG:32643"
    nir = _write(
        tmp_path / "nir.tif",
        np.full(shape, 0.6, np.float32),
        transform=transform,
        crs=crs,
    )
    with pytest.raises(SatelliteAnalyzerError, match="alignment do not match"):
        SpectralBandLoader().load(red_path=red, nir_path=nir)


@pytest.mark.parametrize("resampling", ["nearest", "bilinear"])
def test_opt_in_reprojection_aligns_nir_with_nan_support(
    tmp_path: Path, resampling: str
) -> None:
    red = _write(tmp_path / "red.tif", np.full((4, 4), 0.2, np.float32))
    values = np.array([[0.6, np.nan], [0.6, 0.6]], np.float32)
    nir = _write(
        tmp_path / "nir.tif", values, transform=from_origin(500_000, 5_500_000, 20, 20)
    )
    settings = NDVISettings.model_validate(
        {"alignment": "reproject", "resampling": resampling}
    )
    pair = SpectralBandLoader(settings).load(red_path=red, nir_path=nir)
    assert pair.red.shape == pair.nir.shape == (4, 4)
    assert pair.transform == _GRID
    assert pair.alignment["applied"]
    assert pair.alignment["resampling"] == resampling
    assert pair.alignment["mask_resampling"] == "nearest"
    assert not pair.valid_mask[:2, 2:].any()
    assert np.isnan(pair.nir[:2, 2:]).all()
    assert np.allclose(pair.nir[pair.valid_mask], 0.6)


def test_reprojection_rejects_no_overlap(tmp_path: Path) -> None:
    red, _ = _pair(tmp_path)
    nir = _write(
        tmp_path / "far_nir.tif",
        np.full((4, 4), 0.6, np.float32),
        transform=from_origin(800_000, 5_500_000, 10, 10),
    )
    with pytest.raises(SatelliteAnalyzerError, match="no overlapping"):
        SpectralBandLoader(NDVISettings(alignment="reproject")).load(
            red_path=red, nir_path=nir
        )


def test_all_invalid_source_does_not_claim_no_spatial_overlap(tmp_path: Path) -> None:
    red = _write(tmp_path / "red.tif", np.full((4, 4), np.nan, np.float32))
    nir = _write(
        tmp_path / "nir.tif",
        np.full((2, 2), 0.6, np.float32),
        transform=from_origin(500_000, 5_500_000, 20, 20),
    )
    pair = SpectralBandLoader(NDVISettings(alignment="reproject")).load(
        red_path=red, nir_path=nir
    )
    assert not pair.valid_mask.any()


@pytest.mark.parametrize("invalid", ["missing_crs", "identity", "singular", "complex"])
def test_invalid_spatial_or_numeric_inputs_rejected(
    tmp_path: Path, invalid: str
) -> None:
    red, _ = _pair(tmp_path)
    crs = None if invalid == "missing_crs" else "EPSG:32643"
    transform = (
        Affine.identity()
        if invalid == "identity"
        else Affine(10, 0, 500_000, 0, 0, 5_500_000) if invalid == "singular" else _GRID
    )
    dtype = np.complex64 if invalid == "complex" else np.float32
    nir = _write(
        tmp_path / "invalid.tif", np.ones((4, 4), dtype), crs=crs, transform=transform
    )
    with pytest.raises(SatelliteAnalyzerError):
        SpectralBandLoader().load(red_path=red, nir_path=nir)


def test_budget_checked_before_band_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    red, nir = _pair(tmp_path)
    loader = SpectralBandLoader(NDVISettings(max_pixels=8))
    decoded = False

    def forbidden_read(*arguments: Any, **keywords: Any) -> None:
        nonlocal decoded
        decoded = True
        raise AssertionError("Pixel budget must be checked before decoding")

    monkeypatch.setattr(loader, "_read_band", forbidden_read)
    with pytest.raises(SatelliteAnalyzerError, match="pixel budget"):
        loader.load(red_path=red, nir_path=nir)
    assert not decoded


def test_missing_corrupt_and_unsupported_files_rejected(tmp_path: Path) -> None:
    red, _ = _pair(tmp_path)
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"not an image")
    unsupported = tmp_path / "not_tiff.png"
    unsupported.write_bytes(b"not an image")
    for path in (tmp_path / "missing.tif", corrupt, unsupported):
        with pytest.raises(SatelliteAnalyzerError):
            SpectralBandLoader().load(red_path=red, nir_path=path)
