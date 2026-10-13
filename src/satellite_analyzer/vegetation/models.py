"""Serializable vegetation analysis product for the shared analysis pipeline."""

from dataclasses import asdict, dataclass
from typing import Any

from satellite_analyzer.core.ndvi_config import NDVISettings
from satellite_analyzer.io.spectral_loader import SpectralPair
from satellite_analyzer.vegetation.classification import VegetationAnalysis
from satellite_analyzer.vegetation.ndvi import NDVIData


@dataclass(frozen=True, slots=True)
class NDVIAnalysisResult:
    """Spectral provenance, NDVI, and vegetation summaries on a georeferenced grid."""

    pair: SpectralPair
    ndvi: NDVIData
    vegetation: VegetationAnalysis
    settings: NDVISettings
    processing_time_seconds: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize provenance/statistics while keeping rasters in their own files."""
        return {
            "schema_version": "1.0",
            "sensor": self.pair.sensor,
            "analysis_type": "vegetation_ndvi",
            "formula": "(NIR - RED) / (NIR + RED)",
            "resolution": {
                "width": self.ndvi.values.shape[1],
                "height": self.ndvi.values.shape[0],
            },
            "crs": self.pair.crs.to_string(),
            "transform": list(self.pair.transform)[:6],
            "input_metadata": {
                "red": self.pair.red_metadata,
                "nir": self.pair.nir_metadata,
            },
            "alignment": self.pair.alignment,
            "settings": self.settings.model_dump(mode="json"),
            "statistics": asdict(self.vegetation.statistics),
            "classes": [asdict(item) for item in self.vegetation.classes],
            "percentage_denominator": "valid NDVI pixels only",
            "vegetation_coverage_rule": "NDVI >= coverage_threshold",
            "classification_boundary_rule": (
                "non-vegetation includes its upper boundary; sparse excludes both "
                "boundaries; low/moderate include the lower boundary; dense includes 1"
            ),
            "processing_time_seconds": self.processing_time_seconds,
            "timing_scope": (
                "loading, calibration, alignment, NDVI, and statistics; excludes export"
            ),
            "timestamp": self.timestamp,
            "limitations": [
                "NDVI is a spectral vegetation-condition indicator, not fire "
                "detection or a direct moisture measurement.",
                "Initial vegetation thresholds are configurable starting points, "
                "not universal ecological standards.",
                "Cloud, shadow, and snow quality masks are not automatically applied.",
                "Nonfinite or negative reflectance and near-zero denominators "
                "are excluded.",
                "No ground-truth accuracy evaluation was performed with this "
                "software release.",
            ],
        }
