"""Typed Week 2 results with a JSON boundary independent of image arrays."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from satellite_analyzer.detectors.base import FireRegion
from satellite_analyzer.io.metadata import ImageMetadata

FIRE_LIMITATION = (
    "This is a visual fire-like region detector for RGB imagery, "
    "not a thermal satellite fire detection system."
)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Measured fire-like area and connected regions on the original image grid."""

    image_path: Path
    metadata: ImageMetadata
    analysis_width: int
    analysis_height: int
    fire_pixel_count: int
    fire_percentage: float
    fire_confidence: float
    detected_fire_regions: list[FireRegion]
    invalid_pixel_count: int
    processing_time_seconds: float
    timestamp: str
    parameters: dict[str, Any]
    original_rgb: np.ndarray = field(repr=False, compare=False)
    fire_mask: np.ndarray = field(repr=False, compare=False)

    @property
    def width(self) -> int:
        """Original image width in pixels."""
        return self.metadata.width

    @property
    def height(self) -> int:
        """Original image height in pixels."""
        return self.metadata.height

    def to_dict(self) -> dict[str, Any]:
        """Return finite, JSON-ready report fields without NumPy arrays."""
        total = self.width * self.height
        return {
            "schema_version": "1.0",
            "analysis_scope": "fire_only_week_2",
            "image": self.metadata.filename,
            "image_path": str(self.image_path),
            "resolution": {"width": self.width, "height": self.height},
            "analysis_resolution": {
                "width": self.analysis_width,
                "height": self.analysis_height,
            },
            "metadata": self.metadata.model_dump(mode="json"),
            "fire": {
                "pixel_count": self.fire_pixel_count,
                "percentage": self.fire_percentage,
                "confidence": self.fire_confidence,
                "region_count": len(self.detected_fire_regions),
                "regions": [asdict(region) for region in self.detected_fire_regions],
            },
            # Week 2 does not classify vegetation, water, or urban surfaces yet.
            "unknown": {
                "pixel_count": total - self.fire_pixel_count,
                "percentage": 100.0 - self.fire_percentage,
            },
            "invalid_pixel_count": self.invalid_pixel_count,
            "area_denominator": "all original image pixels, including invalid pixels",
            "coordinate_system": (
                "original image pixels, origin top-left; "
                "boxes [x,x+width),[y,y+height)"
            ),
            "confidence_kind": (
                "RGB heuristic evidence strength, not a calibrated probability"
            ),
            "parameters": self.parameters,
            "processing_time_seconds": self.processing_time_seconds,
            "timestamp": self.timestamp,
            "limitations": [
                FIRE_LIMITATION,
                "Warm roofs, soil, and sunlit surfaces may be false positives.",
                "Downsampling may miss small regions; area uses the resized mask "
                "mapped to the original grid.",
                "Only fire-like and unknown classes are implemented in Week 2.",
            ],
        }
