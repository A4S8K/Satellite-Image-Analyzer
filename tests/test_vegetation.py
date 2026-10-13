"""Classification boundaries, valid-pixel statistics, and configurable coverage."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from pydantic import ValidationError

from satellite_analyzer.core.ndvi_config import NDVISettings
from satellite_analyzer.vegetation.classification import analyze_vegetation


def test_exact_class_boundaries_and_valid_denominator() -> None:
    ndvi = np.array(
        [[-1, -0.5, 0, 0.01, 0.199, 0.2, 0.4, 0.6, 1, np.nan]], dtype=np.float32
    )
    result = analyze_vegetation(ndvi)
    np.testing.assert_array_equal(
        result.classification, [[1, 1, 1, 2, 2, 3, 4, 5, 5, 0]]
    )
    assert result.statistics.valid_pixels == 9
    assert result.statistics.invalid_pixels == 1
    assert sum(item.pixel_count for item in result.classes) == 9
    assert sum(
        item.percentage for item in result.classes if item.percentage is not None
    ) == pytest.approx(100)
    assert result.statistics.vegetation_pixels == 4
    assert result.statistics.vegetation_coverage_percent == pytest.approx(400 / 9)


def test_statistics_exclude_nodata_and_use_population_std() -> None:
    ndvi = np.array([[-0.5, 0, 0.5, 1, np.nan]], dtype=np.float32)
    result = analyze_vegetation(ndvi)
    stats = result.statistics
    assert stats.mean_ndvi == pytest.approx(0.25)
    assert stats.median_ndvi == pytest.approx(0.25)
    assert stats.min_ndvi == -0.5
    assert stats.max_ndvi == 1
    assert stats.std_ndvi == pytest.approx(np.std([-0.5, 0, 0.5, 1], ddof=0))


def test_custom_thresholds_and_external_validity() -> None:
    settings = NDVISettings(
        non_vegetation_max=0.1,
        sparse_max=0.3,
        low_max=0.5,
        moderate_max=0.7,
        coverage_threshold=0.7,
    )
    values = np.array([[0.1, 0.2, 0.3, 0.5, 0.7, 0.8]], dtype=np.float32)
    valid = np.array([[True, True, True, True, True, False]])
    result = analyze_vegetation(values, valid, settings)
    np.testing.assert_array_equal(result.classification, [[1, 2, 3, 4, 5, 0]])
    assert result.statistics.vegetation_coverage_percent == 20.0


def test_all_invalid_is_explicitly_undefined() -> None:
    result = analyze_vegetation(
        np.array([[np.nan, np.inf, 1.1, -1.1]], dtype=np.float32)
    )
    assert result.statistics.valid_pixels == 0
    assert result.statistics.mean_ndvi is None
    assert result.statistics.vegetation_coverage_percent is None
    assert all(
        item.pixel_count == 0 and item.percentage is None for item in result.classes
    )
    assert not result.classification.any()
    payload = json.loads(json.dumps(asdict(result.statistics), allow_nan=False))
    assert payload["mean_ndvi"] is None
    assert payload["invalid_pixels"] == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"sparse_max": -0.1},
        {"low_max": 0.2},
        {"moderate_max": 1.1},
        {"coverage_threshold": np.nan},
        {"red_scale": 0.0001},
        {
            "red_scale": 0.0001,
            "red_offset": 0.0,
            "nir_scale": 0.0001,
            "nir_offset": 0.0,
            "input_units": "reflectance",
        },
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        NDVISettings.model_validate(overrides)
