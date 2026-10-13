"""Scientific-grid, provenance, and display regressions for NDVI export."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from matplotlib.figure import Figure
from PIL import Image
from rasterio.enums import MaskFlags
from rasterio.transform import Affine

from satellite_analyzer.core.exceptions import ImageProcessingError
from satellite_analyzer.core.ndvi_config import NDVISettings
from satellite_analyzer.reporting.ndvi import save_ndvi_artifacts
from satellite_analyzer.vegetation.analyzer import NDVIAnalyzer
from satellite_analyzer.vegetation.models import NDVIAnalysisResult
from satellite_analyzer.visualization.ndvi import (
    save_ndvi_heatmap,
    save_ndvi_histogram,
    save_vegetation_classes,
)

TRANSFORM = Affine(10, 1.5, 500_000, 0.5, -10, 4_300_000)


def _write_band(path: Path, values: np.ndarray) -> None:
    """Synthetic reflectance with a rotated affine grid and a defined CRS."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        nodata=float("nan"),
        crs="EPSG:32642",
        transform=TRANSFORM,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def _result(
    directory: Path, *, large: bool = False, invalid: bool = False
) -> NDVIAnalysisResult:
    """Use real band loading/calculation rather than manufactured report objects."""
    if large:
        red = np.full((96, 128), 0.2, dtype=np.float32)
        nir = np.linspace(0, 1, red.size, dtype=np.float32).reshape(red.shape)
    else:
        red = np.array([[0.3, 0.2, 0.45, 0.3], [0.2, 0.1, 0, np.nan]], np.float32)
        nir = np.array([[0.1, 0.2, 0.55, 0.5], [0.6, 0.9, 1, np.nan]], np.float32)
    if invalid:
        red[:] = np.nan
        nir[:] = np.nan
    red_path, nir_path = directory / "red.tif", directory / "nir.tif"
    _write_band(red_path, red)
    _write_band(nir_path, nir)
    return NDVIAnalyzer(
        NDVISettings(input_units="reflectance", preview_max_dimension=32)
    ).analyze(red_path=red_path, nir_path=nir_path)


def test_exports_preserve_full_grid_values_validity_and_georeferencing(
    tmp_path: Path,
) -> None:
    """Float NDVI and discrete classes round-trip on the actual scientific grid."""
    result = _result(tmp_path)
    source_paths = [
        Path(result.pair.red_metadata["path"]),
        Path(result.pair.nir_metadata["path"]),
    ]
    before = {path: path.read_bytes() for path in source_paths}

    artifacts = save_ndvi_artifacts(result, tmp_path / "results")

    assert set(artifacts) == {"ndvi_map", "vegetation_classes", "statistics"}
    with rasterio.open(artifacts["ndvi_map"]) as dataset:
        assert dataset.driver == "GTiff"
        assert dataset.dtypes == ("float32",)
        assert dataset.shape == result.ndvi.values.shape
        assert dataset.crs == result.pair.crs
        assert dataset.transform == TRANSFORM == result.pair.transform
        assert dataset.nodata is not None and np.isnan(dataset.nodata)
        np.testing.assert_allclose(dataset.read(1), result.ndvi.values, equal_nan=True)
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, result.ndvi.valid_mask)
        assert MaskFlags.per_dataset in dataset.mask_flag_enums[0]
        assert len(dataset.files) == 1
    with rasterio.open(artifacts["vegetation_classes"]) as dataset:
        assert dataset.dtypes == ("uint8",)
        assert dataset.nodata == 0
        assert dataset.shape == result.ndvi.values.shape
        assert dataset.crs == result.pair.crs
        assert dataset.transform == TRANSFORM
        np.testing.assert_array_equal(dataset.read(1), result.vegetation.classification)
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, result.ndvi.valid_mask)
        labels = json.loads(dataset.tags()["class_labels"])
        assert labels["0"] == "Invalid"
        assert labels["5"] == "dense"
        assert len(dataset.files) == 1
    for source in source_paths:
        assert source.read_bytes() == before[source]
    assert len(list(Path(artifacts["statistics"]).parent.iterdir())) == 3


def test_statistics_json_is_strict_complete_and_has_valid_denominator(
    tmp_path: Path,
) -> None:
    """Raster nodata must become JSON null/statistical exclusion, never NaN."""
    result = _result(tmp_path)
    artifacts = save_ndvi_artifacts(result, tmp_path / "results")

    payload = json.loads(Path(artifacts["statistics"]).read_text(encoding="utf-8"))

    json.dumps(payload, allow_nan=False)
    assert payload["statistics"]["valid_pixels"] == 7
    assert payload["statistics"]["invalid_pixels"] == 1
    assert payload["resolution"] == {"width": 4, "height": 2}
    assert payload["crs"] == "EPSG:32642"
    assert payload["transform"] == list(TRANSFORM)[:6]
    assert payload["artifacts"] == artifacts
    assert payload["percentage_denominator"] == "valid NDVI pixels only"
    assert sum(item["pixel_count"] for item in payload["classes"]) == 7
    assert sum(item["percentage"] for item in payload["classes"]) == pytest.approx(100)
    assert all(
        Path(path).is_absolute() and Path(path).is_file() for path in artifacts.values()
    )


def test_optional_visualizations_do_not_resample_scientific_outputs(
    tmp_path: Path,
) -> None:
    """Small preview settings bound PNG rendering but leave GeoTIFF pixels intact."""
    result = _result(tmp_path, large=True)

    artifacts = save_ndvi_artifacts(result, tmp_path / "results", visualize=True)

    assert {"ndvi_heatmap", "vegetation_classes_preview", "ndvi_histogram"} <= set(
        artifacts
    )
    for key in ("ndvi_heatmap", "vegetation_classes_preview", "ndvi_histogram"):
        with Image.open(artifacts[key]) as preview:
            preview.verify()
        with Image.open(artifacts[key]) as preview:
            assert preview.format == "PNG"
            assert 100 <= preview.width <= 1600
            assert 100 <= preview.height <= 1000
    with rasterio.open(artifacts["ndvi_map"]) as dataset:
        assert dataset.shape == (96, 128)
        np.testing.assert_allclose(dataset.read(1), result.ndvi.values)
    with rasterio.open(artifacts["vegetation_classes"]) as dataset:
        assert dataset.shape == (96, 128)
        np.testing.assert_array_equal(dataset.read(1), result.vegetation.classification)


def test_all_invalid_export_uses_nan_mask_and_null_statistics(tmp_path: Path) -> None:
    """No valid samples produce explicit unavailable summaries, not invented zeros."""
    result = _result(tmp_path, invalid=True)

    artifacts = save_ndvi_artifacts(result, tmp_path / "results", visualize=True)
    payload = json.loads(Path(artifacts["statistics"]).read_text(encoding="utf-8"))

    json.dumps(payload, allow_nan=False)
    assert payload["statistics"]["valid_pixels"] == 0
    assert payload["statistics"]["invalid_pixels"] == 8
    assert payload["statistics"]["mean_ndvi"] is None
    assert payload["statistics"]["vegetation_coverage_percent"] is None
    assert all(item["percentage"] is None for item in payload["classes"])
    with rasterio.open(artifacts["ndvi_map"]) as dataset:
        assert np.isnan(dataset.read(1)).all()
        assert not dataset.read_masks(1).any()
    with rasterio.open(artifacts["vegetation_classes"]) as dataset:
        assert not dataset.read(1).any()
        assert not dataset.read_masks(1).any()


def test_repeated_runs_keep_earlier_products(tmp_path: Path) -> None:
    """Two analyses into the same output root must have isolated artifact sets."""
    result = _result(tmp_path)
    first = save_ndvi_artifacts(result, tmp_path / "results")
    first_bytes = {key: Path(value).read_bytes() for key, value in first.items()}

    second = save_ndvi_artifacts(result, tmp_path / "results")

    assert Path(first["statistics"]).parent != Path(second["statistics"]).parent
    for key, path in first.items():
        assert Path(path).read_bytes() == first_bytes[key]
    assert all(Path(path).is_file() for path in second.values())


@pytest.mark.parametrize("band", ["red", "nir"])
def test_output_root_cannot_replace_input_band(tmp_path: Path, band: str) -> None:
    """Both spectral source files receive overwrite protection."""
    result = _result(tmp_path)
    metadata = result.pair.red_metadata if band == "red" else result.pair.nir_metadata
    source = Path(metadata["path"])
    before = source.read_bytes()

    with pytest.raises(ImageProcessingError, match="input file"):
        save_ndvi_artifacts(result, source)

    assert source.read_bytes() == before


def test_histogram_counts_every_valid_pixel_and_marks_true_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The histogram population must not be reduced to the PNG preview population."""
    result = _result(tmp_path, large=True)
    captured: dict[str, Any] = {}

    def inspect_figure(figure: Figure, *args: object, **kwargs: object) -> None:
        axis = figure.axes[0]
        captured["total"] = sum(patch.get_height() for patch in axis.patches)
        captured["means"] = [list(line.get_xdata()) for line in axis.lines]

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    save_ndvi_histogram(result, tmp_path / "histogram.png")

    assert captured["total"] == 96 * 128
    assert captured["total"] == result.vegetation.statistics.valid_pixels
    assert captured["means"] == [
        pytest.approx([result.vegetation.statistics.mean_ndvi] * 2)
    ]


def test_all_invalid_histogram_has_no_mean_and_explicit_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-data distribution must not show a fabricated mean at zero."""
    result = _result(tmp_path, invalid=True)
    captured: dict[str, Any] = {}

    def inspect_figure(figure: Figure, *args: object, **kwargs: object) -> None:
        axis = figure.axes[0]
        captured["mean_lines"] = len(axis.lines)
        captured["messages"] = [text.get_text() for text in axis.texts]

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    save_ndvi_histogram(result, tmp_path / "invalid.png")

    assert captured["mean_lines"] == 0
    assert "No valid NDVI pixels" in captured["messages"]


@pytest.mark.parametrize("kind", ["heatmap", "classes"])
def test_png_previews_are_bounded_and_use_source_pixel_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Display-only resize keeps the source-pixel frame even on a rotated CRS grid."""
    result = _result(tmp_path, large=True)
    captured: dict[str, Any] = {}

    def inspect_figure(figure: Figure, *args: object, **kwargs: object) -> None:
        axis = figure.axes[0]
        image = axis.images[0]
        captured["shape"] = image.get_array().shape
        captured["extent"] = image.get_extent()
        captured["scale"] = image.get_clim()
        captured["label"] = axis.get_xlabel()

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    renderer = save_ndvi_heatmap if kind == "heatmap" else save_vegetation_classes
    renderer(result, tmp_path / "preview.png")

    assert captured["shape"] == (24, 32)
    assert tuple(captured["extent"]) == (0, 128, 96, 0)
    assert captured["label"] == "Source pixel column"
    if kind == "heatmap":
        assert captured["scale"] == (-1, 1)
