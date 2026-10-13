"""Configuration errors are explicit, and defaults do not require repo files."""

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from satellite_analyzer.core.exceptions import SatelliteAnalyzerError
from satellite_analyzer.core.fire_config import (
    AnalysisSettings,
    FireDetectionSettings,
    load_analysis_settings,
    load_fire_settings,
)

SettingsLoader = Callable[[Path | None], FireDetectionSettings | AnalysisSettings]


def test_builtin_defaults_and_serialization() -> None:
    assert load_fire_settings() == FireDetectionSettings()
    assert load_analysis_settings() == AnalysisSettings()
    values = load_fire_settings().model_dump(mode="json")
    assert values["min_region_pixels"] == 16
    assert values["connectivity"] == 8
    assert load_analysis_settings().model_dump(mode="json")["max_pixels"] == 40_000_000


def test_explicit_toml_overrides_only_requested_fields(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text(
        "[fire]\nmin_region_pixels = 3\nconnectivity = 4\n"
        "[analysis]\nmax_dimension = 512\noverlay_alpha = 0.65\n",
        encoding="utf-8",
    )
    assert load_fire_settings(path).min_region_pixels == 3
    assert load_fire_settings(path).connectivity == 4
    assert load_fire_settings(path).min_value == 140
    assert load_analysis_settings(path).max_dimension == 512
    assert load_analysis_settings(path).overlay_alpha == 0.65


def test_bundled_default_matches_model_defaults() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
    assert load_fire_settings(path) == FireDetectionSettings()
    assert load_analysis_settings(path) == AnalysisSettings()


@pytest.mark.parametrize("loader", [load_fire_settings, load_analysis_settings])
def test_missing_config_raises_domain_error(
    tmp_path: Path, loader: SettingsLoader
) -> None:
    with pytest.raises(SatelliteAnalyzerError, match="Cannot read configuration"):
        loader(tmp_path / "missing.toml")


@pytest.mark.parametrize("loader", [load_fire_settings, load_analysis_settings])
def test_malformed_config_raises_domain_error(
    tmp_path: Path, loader: SettingsLoader
) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("[fire\nmin_value = 240", encoding="utf-8")
    with pytest.raises(SatelliteAnalyzerError, match="Cannot read configuration"):
        loader(path)


@pytest.mark.parametrize(
    "values",
    [
        {"hue_min": -1},
        {"hue_max": 180},
        {"hue_min": 40, "hue_max": 35},
        {"red_hue_min": 179, "red_hue_max": 170},
        {"min_saturation": 256},
        {"min_value": -1},
        {"min_red": -1},
        {"min_red_green_ratio": 0.9},
        {"min_red_green_ratio": 256.0},
        {"min_red_blue_ratio": float("inf")},
        {"min_region_pixels": 0},
        {"min_region_pixels": True},
        {"morphology_open_kernel": 1},
        {"morphology_open_kernel": 4},
        {"connectivity": 6},
        {"unknown_threshold": 1},
    ],
)
def test_invalid_fire_settings(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FireDetectionSettings.model_validate(values)


@pytest.mark.parametrize(
    "values",
    [
        {"max_dimension": 31},
        {"max_pixels": 0},
        {"overlay_alpha": -0.1},
        {"overlay_alpha": 1.1},
        {"overlay_alpha": float("nan")},
        {"unexpected": 1},
    ],
)
def test_invalid_analysis_settings(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AnalysisSettings.model_validate(values)


@pytest.mark.parametrize(
    "contents, loader, message",
    [
        ("fire = 123", load_fire_settings, "must be a TOML table"),
        ("analysis = 123", load_analysis_settings, "must be a TOML table"),
        ("[fire]\nmin_value = 999", load_fire_settings, "Invalid fire configuration"),
        (
            "[analysis]\noverlay_alpha = 3.0",
            load_analysis_settings,
            "Invalid analysis configuration",
        ),
    ],
)
def test_invalid_toml_values_are_domain_errors(
    tmp_path: Path, contents: str, loader: SettingsLoader, message: str
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(SatelliteAnalyzerError, match=message):
        loader(path)
