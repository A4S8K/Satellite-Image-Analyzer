"""Validated settings for real Red/NIR analysis and vegetation classification."""

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from satellite_analyzer.core.exceptions import SatelliteAnalyzerError

SensorName = Literal["sentinel-2", "landsat-8", "landsat-9", "custom"]


class NDVISettings(BaseModel):
    """Explicit spectral, radiometric, spatial, and display assumptions.

    A sensor name identifies spectral channels, never an assumed processing
    baseline or calibration. Scale and additive offset must be known per band.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    sensor: SensorName = "custom"
    input_units: Literal["auto", "reflectance", "dn"] = "auto"
    metadata_path: Path | None = None
    red_scale: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    red_offset: float | None = Field(default=None, allow_inf_nan=False)
    nir_scale: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    nir_offset: float | None = Field(default=None, allow_inf_nan=False)
    alignment: Literal["strict", "reproject"] = "strict"
    resampling: Literal["nearest", "bilinear"] = "bilinear"
    max_pixels: int = Field(default=150_000_000, ge=1)
    denominator_epsilon: float = Field(default=1e-6, ge=0, le=1, allow_inf_nan=False)
    non_vegetation_max: float = Field(default=0.0, ge=-1, lt=1, allow_inf_nan=False)
    sparse_max: float = Field(default=0.2, gt=-1, lt=1, allow_inf_nan=False)
    low_max: float = Field(default=0.4, gt=-1, lt=1, allow_inf_nan=False)
    moderate_max: float = Field(default=0.6, gt=-1, lt=1, allow_inf_nan=False)
    coverage_threshold: float = Field(default=0.2, ge=-1, le=1, allow_inf_nan=False)
    histogram_bins: int = Field(default=50, ge=2, le=4096)
    preview_max_dimension: int = Field(default=2048, ge=32)

    @model_validator(mode="after")
    def validate_combinations(self) -> Self:
        """Reject ambiguous calibration and non-monotonic class boundaries."""
        coefficients = (
            self.red_scale,
            self.red_offset,
            self.nir_scale,
            self.nir_offset,
        )
        if any(value is not None for value in coefficients) and any(
            value is None for value in coefficients
        ):
            raise ValueError("Supply all four red/nir scale and offset coefficients")
        if self.metadata_path is not None and self.red_scale is not None:
            raise ValueError("Choose metadata or explicit coefficients, not both")
        if self.input_units == "reflectance" and (
            self.metadata_path is not None or self.red_scale is not None
        ):
            raise ValueError("Reflectance input must not be calibrated a second time")
        if not (
            self.non_vegetation_max < self.sparse_max < self.low_max < self.moderate_max
        ):
            raise ValueError(
                "Vegetation classification boundaries must increase strictly"
            )
        return self


def load_ndvi_settings(path: Path | None = None) -> NDVISettings:
    """Read ``[ndvi]`` TOML overrides or use self-contained installed defaults."""
    if path is None:
        return NDVISettings()
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle).get("ndvi", {})
        if not isinstance(raw, dict):
            raise ValueError("ndvi must be a TOML table")
        if "metadata_path" in raw:
            raw["metadata_path"] = (path.parent / Path(raw["metadata_path"])).resolve()
        return NDVISettings.model_validate(raw)
    except (OSError, ValueError, TypeError, ValidationError) as error:
        raise SatelliteAnalyzerError(
            f"Invalid NDVI configuration '{path}': {error}"
        ) from error
