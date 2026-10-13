"""Validated, reproducible settings for RGB fire analysis.

Defaults live in the models so an installed wheel works without the repository's
``configs`` directory. An explicitly supplied TOML file is never silently ignored.
"""

import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from satellite_analyzer.core.exceptions import SatelliteAnalyzerError


class FireDetectionSettings(BaseModel):
    """Thresholds in OpenCV HSV units (H: 0..179, S/V: 0..255).

    Regions are filtered using their pixel area at the *analysis resolution*.
    Scores describe RGB evidence strength, not calibrated fire probabilities.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    hue_min: int = Field(default=0, ge=0, le=179)
    hue_max: int = Field(default=35, ge=0, le=179)
    red_hue_min: int = Field(default=170, ge=0, le=179)
    red_hue_max: int = Field(default=179, ge=0, le=179)
    min_saturation: int = Field(default=80, ge=0, le=255)
    min_value: int = Field(default=140, ge=0, le=255)
    min_red: int = Field(default=140, ge=0, le=255)
    # Equality permits yellow pixels; requiring R > G would exclude pure yellow.
    min_red_green_ratio: float = Field(
        default=1.0, ge=1.0, le=255.0, allow_inf_nan=False
    )
    min_red_blue_ratio: float = Field(
        default=1.2, ge=1.0, le=255.0, allow_inf_nan=False
    )
    min_region_pixels: int = Field(default=16, ge=1)
    morphology_open_kernel: int = Field(default=0, ge=0, le=31)
    connectivity: Literal[4, 8] = 8

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """Reject reversed hue ranges and invalid morphology kernel sizes."""

        if self.hue_min > self.hue_max:
            raise ValueError("hue_min must not exceed hue_max")
        if self.red_hue_min > self.red_hue_max:
            raise ValueError("red_hue_min must not exceed red_hue_max")
        kernel = self.morphology_open_kernel
        if kernel != 0 and (kernel < 3 or kernel % 2 == 0):
            raise ValueError("morphology_open_kernel must be 0 or an odd size >= 3")
        return self


class AnalysisSettings(BaseModel):
    """Resource and visualization settings for the RGB analysis pipeline."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_dimension: int = Field(default=2048, ge=32)
    max_pixels: int = Field(default=40_000_000, ge=1)
    overlay_alpha: float = Field(default=0.4, ge=0.0, le=1.0, allow_inf_nan=False)


def _load_section(path: Path, section: str) -> dict[str, Any]:
    """Read one TOML table and turn expected input errors into domain errors."""

    try:
        with path.open("rb") as config_file:
            values = tomllib.load(config_file)
    except (OSError, ValueError) as exc:
        raise SatelliteAnalyzerError(
            f"Cannot read configuration '{path}': {exc}"
        ) from exc
    table = values.get(section, {})
    if not isinstance(table, dict):
        raise SatelliteAnalyzerError(
            f"Configuration '{path}': '{section}' must be a TOML table"
        )
    return table


def load_fire_settings(path: Path | None = None) -> FireDetectionSettings:
    """Load the optional ``[fire]`` table, or return built-in defaults."""

    if path is None:
        return FireDetectionSettings()
    try:
        return FireDetectionSettings.model_validate(_load_section(path, "fire"))
    except ValidationError as exc:
        raise SatelliteAnalyzerError(
            f"Invalid fire configuration '{path}': {exc}"
        ) from exc


def load_analysis_settings(path: Path | None = None) -> AnalysisSettings:
    """Load the optional ``[analysis]`` table, or return built-in defaults."""

    if path is None:
        return AnalysisSettings()
    try:
        return AnalysisSettings.model_validate(_load_section(path, "analysis"))
    except ValidationError as exc:
        raise SatelliteAnalyzerError(
            f"Invalid analysis configuration '{path}': {exc}"
        ) from exc
