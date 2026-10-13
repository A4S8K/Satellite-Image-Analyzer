"""End-to-end NDVI CLI checks with actual synthetic spectral GeoTIFF inputs."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import Affine, from_origin
from typer.testing import CliRunner

from satellite_analyzer.cli import app

runner = CliRunner()
TRANSFORM = Affine(10, 0.5, 500_000, 0.25, -10, 4_700_000)


def _write_band(
    path: Path, values: np.ndarray, *, transform: Affine = TRANSFORM
) -> None:
    """Preserve synthetic source values and georeferencing without RGB conversion."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        nodata=float("nan") if np.issubdtype(values.dtype, np.floating) else 0,
        crs="EPSG:32643",
        transform=transform,
    ) as dataset:
        dataset.write(values, 1)
        dataset.update_tags(SYNTHETIC="true", purpose="CLI verification only")


@pytest.fixture()
def spectral_pair(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    """Include positive, zero, negative NDVI and four invalid source/sum cases."""
    red = np.array(
        [[0.3, 0.2, 0.45, 0.3], [0.2, 0.1, 0, np.nan], [0, -0.1, 0.2, 0.2]],
        dtype=np.float32,
    )
    nir = np.array(
        [[0.1, 0.2, 0.55, 0.5], [0.6, 0.9, 1, np.nan], [0, 0.4, -0.1, 0.2]],
        dtype=np.float32,
    )
    red_path, nir_path = (
        tmp_path / "red reflectance.tif",
        tmp_path / "nir reflectance.tif",
    )
    _write_band(red_path, red)
    _write_band(nir_path, nir)
    return red_path, nir_path, red, nir


def _invoke_json(arguments: list[str], output: Path) -> dict[str, Any]:
    """Parse only stdout and require a finite JSON payload on successful exit."""
    result = runner.invoke(app, ["ndvi", *arguments, "--output", str(output), "--json"])
    assert result.exit_code == 0, result.output
    payload: dict[str, Any] = json.loads(result.stdout)
    json.dumps(payload, allow_nan=False)
    return payload


def _separate_arguments(red: Path, nir: Path) -> list[str]:
    return ["--red", str(red), "--nir", str(nir), "--input-units", "reflectance"]


def test_ndvi_cli_formula_statistics_and_geospatial_artifacts(
    spectral_pair: tuple[Path, Path, np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """CLI export agrees with independently computed NDVI, masks, and statistics."""
    red_path, nir_path, red, nir = spectral_pair
    before = {path: path.read_bytes() for path in (red_path, nir_path)}
    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & (red >= 0)
        & (nir >= 0)
        & (red + nir > 1e-6)
    )
    expected = np.full(red.shape, np.nan, dtype=np.float32)
    expected[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
    expected_values = expected[valid]
    output = tmp_path / "spectral results"

    report = _invoke_json(
        [*_separate_arguments(red_path, nir_path), "--visualize", "--verbose"], output
    )

    assert report["analysis_type"] == "vegetation_ndvi"
    assert report["formula"] == "(NIR - RED) / (NIR + RED)"
    assert report["resolution"] == {"width": 4, "height": 3}
    assert report["crs"] == "EPSG:32643"
    assert report["transform"] == list(TRANSFORM)[:6]
    assert report["percentage_denominator"] == "valid NDVI pixels only"
    stats = report["statistics"]
    assert stats["valid_pixels"] == int(valid.sum()) == 8
    assert stats["invalid_pixels"] == int((~valid).sum()) == 4
    assert stats["mean_ndvi"] == pytest.approx(
        float(np.mean(expected_values, dtype=np.float64))
    )
    assert stats["min_ndvi"] == pytest.approx(float(expected_values.min()))
    assert stats["max_ndvi"] == pytest.approx(float(expected_values.max()))
    assert stats["median_ndvi"] == pytest.approx(float(np.median(expected_values)))
    assert stats["std_ndvi"] == pytest.approx(
        float(np.std(expected_values, dtype=np.float64))
    )
    assert stats["vegetation_coverage_percent"] == pytest.approx(
        100 * np.count_nonzero(expected_values >= 0.2) / expected_values.size
    )
    assert sum(item["pixel_count"] for item in report["classes"]) == valid.sum()
    assert sum(item["percentage"] for item in report["classes"]) == pytest.approx(100)
    assert not {"fire", "fire_confidence", "wildfire_risk"}.intersection(report)
    assert set(report["artifacts"]) == {
        "ndvi_map",
        "vegetation_classes",
        "statistics",
        "ndvi_heatmap",
        "vegetation_classes_preview",
        "ndvi_histogram",
    }
    artifacts = {key: Path(path) for key, path in report["artifacts"].items()}
    run = artifacts["statistics"].parent
    assert list(output.iterdir()) == [run]
    assert all(path.is_absolute() and path.parent == run for path in artifacts.values())
    assert json.loads(artifacts["statistics"].read_text(encoding="utf-8")) == report
    with rasterio.open(artifacts["ndvi_map"]) as dataset:
        assert dataset.dtypes == ("float32",)
        assert dataset.crs is not None and dataset.crs.to_string() == "EPSG:32643"
        assert dataset.transform == TRANSFORM
        assert dataset.nodata is not None and np.isnan(dataset.nodata)
        np.testing.assert_allclose(dataset.read(1), expected, equal_nan=True)
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, valid)
        assert len(dataset.files) == 1
    with rasterio.open(artifacts["vegetation_classes"]) as dataset:
        assert dataset.dtypes == ("uint8",)
        assert dataset.nodata == 0
        assert dataset.transform == TRANSFORM
        classes = dataset.read(1)
        assert not classes[~valid].any()
        assert set(np.unique(classes[valid])) <= {1, 2, 3, 4, 5}
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, valid)
    for key in ("ndvi_heatmap", "vegetation_classes_preview", "ndvi_histogram"):
        with Image.open(artifacts[key]) as image:
            image.verify()
    assert all(path.read_bytes() == content for path, content in before.items())


def _sentinel_metadata(path: Path) -> None:
    """Write explicit synthetic PB04 calibration; do not imitate a real acquisition."""
    path.write_text(
        "<Level-2A_User_Product><PROCESSING_LEVEL>Level-2A</PROCESSING_LEVEL>"
        "<PROCESSING_BASELINE>04.00</PROCESSING_BASELINE>"
        "<SPACECRAFT_NAME>Sentinel-2A</SPACECRAFT_NAME>"
        "<BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>"
        '<BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET>'
        '<BOA_ADD_OFFSET band_id="7">-1000</BOA_ADD_OFFSET>'
        "<SYNTHETIC>true</SYNTHETIC></Level-2A_User_Product>",
        encoding="utf-8",
    )


@pytest.mark.parametrize("declare_sensor", [False, True])
def test_ndvi_cli_sentinel_metadata_applies_additive_offsets(
    tmp_path: Path, declare_sensor: bool
) -> None:
    """Sentinel DN calibration, including additive offsets, changes the actual index."""
    red_path, nir_path = tmp_path / "B04.tif", tmp_path / "B08.tif"
    _write_band(red_path, np.array([[2000, 4000], [0, 1000]], np.uint16))
    _write_band(nir_path, np.array([[6000, 2000], [0, 1000]], np.uint16))
    metadata = tmp_path / "MTD_MSIL2A.xml"
    _sentinel_metadata(metadata)

    arguments = [
        "--red",
        str(red_path),
        "--nir",
        str(nir_path),
        "--metadata",
        str(metadata),
    ]
    if declare_sensor:
        arguments.extend(["--sensor", "sentinel-2"])
    report = _invoke_json(
        arguments,
        tmp_path / "results",
    )

    assert report["sensor"] == "sentinel-2"
    assert report["statistics"]["valid_pixels"] == 2
    assert report["statistics"]["invalid_pixels"] == 2
    for role in ("red", "nir"):
        calibration = report["input_metadata"][role]["calibration"]
        assert calibration["source"] == "sentinel_product_metadata"
        assert calibration["scale"] == pytest.approx(0.0001)
        assert calibration["offset"] == pytest.approx(-0.1)
        assert calibration["reflectance_type"] == "surface"
    expected = np.array([[2 / 3, -0.5], [np.nan, np.nan]], np.float32)
    with rasterio.open(report["artifacts"]["ndvi_map"]) as dataset:
        np.testing.assert_allclose(dataset.read(1), expected, equal_nan=True, atol=1e-6)
    assert report["statistics"]["mean_ndvi"] == pytest.approx(
        (2 / 3 - 0.5) / 2, abs=1e-6
    )


def test_ndvi_cli_dn_without_calibration_is_rejected(tmp_path: Path) -> None:
    """Naming a satellite is not permission to invent DN scale/offset coefficients."""
    red_path, nir_path = tmp_path / "B04.tif", tmp_path / "B08.tif"
    _write_band(red_path, np.full((2, 2), 2000, np.uint16))
    _write_band(nir_path, np.full((2, 2), 6000, np.uint16))
    output = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "ndvi",
            "--red",
            str(red_path),
            "--nir",
            str(nir_path),
            "--sensor",
            "sentinel-2",
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Unknown radiometric scale" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_ndvi_cli_alignment_requires_opt_in_and_records_reference_grid(
    tmp_path: Path,
) -> None:
    """Strict mode fails on resolution mismatch; opt-in maps NIR onto Red pixels."""
    red_path, nir_path = tmp_path / "red.tif", tmp_path / "nir.tif"
    transform = from_origin(500_000, 4_700_000, 10, 10)
    _write_band(red_path, np.full((4, 4), 0.2, np.float32), transform=transform)
    _write_band(
        nir_path,
        np.full((2, 2), 0.6, np.float32),
        transform=from_origin(500_000, 4_700_000, 20, 20),
    )
    output = tmp_path / "results"
    arguments = _separate_arguments(red_path, nir_path)

    rejected = runner.invoke(
        app, ["ndvi", *arguments, "--output", str(output), "--json"]
    )

    assert rejected.exit_code != 0
    assert rejected.stdout == ""
    assert "alignment" in rejected.stderr
    assert not output.exists()

    report = _invoke_json(
        [*arguments, "--align-nir", "--resampling", "nearest"], output
    )

    assert report["alignment"]["policy"] == "reproject"
    assert report["alignment"]["applied"] is True
    assert report["alignment"]["reference_grid"] == "red"
    assert report["alignment"]["resampling"] == "nearest"
    assert report["resolution"] == {"width": 4, "height": 4}
    assert report["input_metadata"]["nir"]["width"] == 2
    assert report["statistics"]["valid_pixels"] == 16
    with rasterio.open(report["artifacts"]["ndvi_map"]) as dataset:
        assert dataset.transform == transform
        np.testing.assert_allclose(dataset.read(1), 0.5, atol=1e-6)


def _write_stack(path: Path, red: np.ndarray, nir: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=red.shape[1],
        height=red.shape[0],
        count=2,
        dtype="float32",
        nodata=float("nan"),
        crs="EPSG:32643",
        transform=TRANSFORM,
    ) as dataset:
        dataset.write(red, 1)
        dataset.write(nir, 2)
        dataset.set_band_description(1, "Measured Red - synthetic fixture")
        dataset.set_band_description(2, "Measured NIR - synthetic fixture")


def test_ndvi_cli_multiband_uses_explicit_indexes(
    spectral_pair: tuple[Path, Path, np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """Independent spectral arrays can be explicitly mapped in a stack."""
    _, _, red, nir = spectral_pair
    stack = tmp_path / "spectral stack.tif"
    _write_stack(stack, red, nir)

    report = _invoke_json(
        [
            "--multiband",
            str(stack),
            "--red-band",
            "1",
            "--nir-band",
            "2",
            "--input-units",
            "reflectance",
        ],
        tmp_path / "results",
    )

    assert report["input_metadata"]["red"]["band_index"] == 1
    assert report["input_metadata"]["nir"]["band_index"] == 2
    assert report["statistics"]["valid_pixels"] == 8
    assert report["statistics"]["mean_ndvi"] == pytest.approx(0.26875, abs=1e-6)


@pytest.mark.parametrize(
    "mode", ["missing", "one_band", "mixed", "unmapped", "same_band"]
)
def test_ndvi_cli_rejects_ambiguous_input_modes(
    spectral_pair: tuple[Path, Path, np.ndarray, np.ndarray], tmp_path: Path, mode: str
) -> None:
    """CLI input modes must identify two distinct measured spectral channels."""
    red_path, nir_path, red, nir = spectral_pair
    stack = tmp_path / "stack.tif"
    _write_stack(stack, red, nir)
    cases = {
        "missing": [],
        "one_band": ["--red", str(red_path)],
        "mixed": [
            "--red",
            str(red_path),
            "--nir",
            str(nir_path),
            "--multiband",
            str(stack),
            "--red-band",
            "1",
            "--nir-band",
            "2",
        ],
        "unmapped": ["--multiband", str(stack)],
        "same_band": ["--multiband", str(stack), "--red-band", "1", "--nir-band", "1"],
    }
    output = tmp_path / "results"

    result = runner.invoke(
        app, ["ndvi", *cases[mode], "--output", str(output), "--json"]
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_ndvi_cli_visible_rgb_cannot_supply_nir(tmp_path: Path) -> None:
    """Explicit indexes do not legitimize using a visible blue band as infrared."""
    image = tmp_path / "visible_rgb.tif"
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=TRANSFORM,
    ) as dataset:
        dataset.write(np.full((3, 2, 2), 100, np.uint8))
        dataset.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    output = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "ndvi",
            "--multiband",
            str(image),
            "--red-band",
            "1",
            "--nir-band",
            "3",
            "--input-units",
            "reflectance",
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "RGB channels are not substitutes" in " ".join(result.stderr.split())
    assert not output.exists()


def test_ndvi_cli_all_invalid_statistics_are_null(tmp_path: Path) -> None:
    """A validly described all-NoData pair exports unavailable summaries, not zeros."""
    red_path, nir_path = tmp_path / "red.tif", tmp_path / "nir.tif"
    for path in (red_path, nir_path):
        _write_band(path, np.full((2, 3), np.nan, np.float32))

    report = _invoke_json(_separate_arguments(red_path, nir_path), tmp_path / "results")

    assert report["statistics"]["valid_pixels"] == 0
    assert report["statistics"]["invalid_pixels"] == 6
    for name in (
        "mean_ndvi",
        "min_ndvi",
        "max_ndvi",
        "median_ndvi",
        "std_ndvi",
        "vegetation_coverage_percent",
    ):
        assert report["statistics"][name] is None
    assert all(item["percentage"] is None for item in report["classes"])
    with rasterio.open(report["artifacts"]["ndvi_map"]) as dataset:
        assert np.isnan(dataset.read(1)).all()
        assert not dataset.read_masks(1).any()


def test_ndvi_cli_config_changes_coverage_and_classes(
    spectral_pair: tuple[Path, Path, np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """The NDVI config table affects actual classification and valid-pixel coverage."""
    red_path, nir_path, _, _ = spectral_pair
    config = tmp_path / "ndvi.toml"
    config.write_text(
        "[ndvi]\ncoverage_threshold = 0.6\nsparse_max = 0.15\n"
        "low_max = 0.2\nmoderate_max = 0.7\n",
        encoding="utf-8",
    )

    report = _invoke_json(
        [*_separate_arguments(red_path, nir_path), "--config", str(config)],
        tmp_path / "results",
    )

    assert report["settings"]["coverage_threshold"] == 0.6
    assert report["statistics"]["vegetation_coverage_percent"] == 25
    classes = {item["name"]: item["pixel_count"] for item in report["classes"]}
    assert classes["low"] == 0
    assert classes["moderate"] == 2


def test_ndvi_cli_cannot_overwrite_source(
    spectral_pair: tuple[Path, Path, np.ndarray, np.ndarray],
) -> None:
    """Choosing the source as output fails without modifying the geospatial input."""
    red_path, nir_path, _, _ = spectral_pair
    before = red_path.read_bytes()

    result = runner.invoke(
        app,
        [
            "ndvi",
            *_separate_arguments(red_path, nir_path),
            "--output",
            str(red_path),
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "input file path" in result.stderr
    assert red_path.read_bytes() == before
