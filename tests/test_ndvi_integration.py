"""Integration with the application facade and georeferenced calibration metadata."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satellite_analyzer.analyzer import SatelliteAnalyzer
from satellite_analyzer.core.exceptions import SatelliteAnalyzerError
from satellite_analyzer.core.ndvi_config import NDVISettings, load_ndvi_settings


def test_shared_analyzer_uses_real_tiff_calibration(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("B04.tif", "B08.tif")]
    for path, value in zip(paths, (2000, 6000), strict=True):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=8,
            height=6,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=from_origin(300000, 4700000, 10, 10),
            nodata=0,
        ) as dataset:
            array = np.full((6, 8), value, dtype=np.uint16)
            array[0, 0] = 0
            dataset.write(array, 1)
            dataset.scales = (0.0001,)
            dataset.offsets = (-0.1,)
    before = [path.read_bytes() for path in paths]
    result = SatelliteAnalyzer().analyze_ndvi(
        red_path=paths[0], nir_path=paths[1], settings=NDVISettings(sensor="sentinel-2")
    )
    assert result.ndvi.values[1, 1] == pytest.approx(2 / 3)
    assert np.isnan(result.ndvi.values[0, 0])
    assert result.vegetation.statistics.valid_pixels == 47
    assert result.vegetation.statistics.vegetation_coverage_percent == 100.0
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["crs"] == "EPSG:32643"
    assert payload["input_metadata"]["red"]["calibration"]["offset"] == -0.1
    assert [path.read_bytes() for path in paths] == before


def test_ndvi_toml_overrides_and_relative_metadata(tmp_path: Path) -> None:
    config = tmp_path / "ndvi.toml"
    config.write_text(
        '[ndvi]\nsensor="sentinel-2"\nmetadata_path="MTD_MSIL2A.xml"\n'
        "coverage_threshold=0.5\nhistogram_bins=16\n",
        encoding="utf-8",
    )
    settings = load_ndvi_settings(config)
    assert settings.metadata_path == tmp_path / "MTD_MSIL2A.xml"
    assert settings.coverage_threshold == 0.5
    assert settings.histogram_bins == 16


@pytest.mark.parametrize(
    "contents",
    ["[ndvi]\nsparse_max=-0.1\n", 'ndvi="wrong"\n', "[ndvi]\nunknown_option=1\n"],
)
def test_bad_configuration_fails_clearly(tmp_path: Path, contents: str) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(contents, encoding="utf-8")
    with pytest.raises(SatelliteAnalyzerError):
        load_ndvi_settings(config)
