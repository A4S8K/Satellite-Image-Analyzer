"""Fire-only Week 2 orchestration with accurate original-grid reporting."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

from satellite_analyzer.core.exceptions import ImageProcessingError
from satellite_analyzer.core.fire_config import AnalysisSettings, FireDetectionSettings
from satellite_analyzer.detectors.base import BaseDetector, FireDetection, FireRegion
from satellite_analyzer.detectors.fire import RGBFireDetector
from satellite_analyzer.io.rgb_loader import RGBImage, RGBImageLoader
from satellite_analyzer.models import AnalysisResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from satellite_analyzer.core.ndvi_config import NDVISettings
    from satellite_analyzer.vegetation.models import NDVIAnalysisResult


class SatelliteAnalyzer:
    """Run a replaceable fire detector and retain source-resolution results."""

    def __init__(
        self,
        fire_settings: FireDetectionSettings | None = None,
        analysis_settings: AnalysisSettings | None = None,
        detector: BaseDetector | None = None,
    ) -> None:
        self.fire_settings = fire_settings or FireDetectionSettings()
        self.analysis_settings = analysis_settings or AnalysisSettings()
        self.detector = detector or RGBFireDetector(self.fire_settings)
        self.loader = RGBImageLoader(self.analysis_settings.max_pixels)

    def analyze(self, image_path: str | Path) -> AnalysisResult:
        """Load visible RGB, detect fire-like components, and measure image area."""
        started = perf_counter()
        logger.info("Loading image: %s", image_path)
        source = self.loader.load(image_path)
        logger.info(
            "Image resolution: %sx%s", source.metadata.width, source.metadata.height
        )
        rgb, valid = self._prepare(source)
        logger.debug("Analysis grid: %sx%s", rgb.shape[1], rgb.shape[0])
        logger.info("Running fire-like detector: %s", type(self.detector).__name__)
        try:
            detected = self.detector.detect(rgb, valid_mask=valid)
            self._validate_detection(detected, rgb.shape[:2], valid)
            mask, regions = self._original_grid(detected, source)
        except cv2.error as error:
            raise ImageProcessingError(
                f"OpenCV fire analysis failed: {error}"
            ) from error
        count = int(np.count_nonzero(mask))
        confidence = (
            sum(region.confidence * region.pixel_count for region in regions) / count
            if count
            else 0.0
        )
        elapsed = perf_counter() - started
        logger.info(
            "Analysis completed in %.3fs (%s fire-like regions)", elapsed, len(regions)
        )
        return AnalysisResult(
            image_path=source.path.resolve(),
            metadata=source.metadata,
            analysis_width=rgb.shape[1],
            analysis_height=rgb.shape[0],
            fire_pixel_count=count,
            fire_percentage=count * 100.0 / mask.size,
            fire_confidence=confidence,
            detected_fire_regions=regions,
            invalid_pixel_count=int(np.count_nonzero(~source.valid_mask)),
            processing_time_seconds=elapsed,
            timestamp=datetime.now(UTC).isoformat(),
            parameters={
                "detector": type(self.detector).__name__,
                "fire": self.fire_settings.model_dump(mode="json"),
                "analysis": self.analysis_settings.model_dump(mode="json"),
                "minimum_region_area_unit": "analysis-grid pixels",
                "timing_scope": (
                    "input loading, RGB preparation, detection, and statistics; "
                    "excludes artifact export"
                ),
            },
            original_rgb=source.rgb,
            fire_mask=mask,
        )

    def analyze_ndvi(
        self,
        *,
        red_path: Path | None = None,
        nir_path: Path | None = None,
        multiband_path: Path | None = None,
        red_band: int | None = None,
        nir_band: int | None = None,
        settings: "NDVISettings | None" = None,
    ) -> "NDVIAnalysisResult":
        """Add spectral vegetation to the application without changing fire scores."""
        from satellite_analyzer.vegetation.analyzer import NDVIAnalyzer

        return NDVIAnalyzer(settings).analyze(
            red_path=red_path,
            nir_path=nir_path,
            multiband_path=multiband_path,
            red_band=red_band,
            nir_band=nir_band,
        )

    def _prepare(self, source: RGBImage) -> tuple[np.ndarray, np.ndarray]:
        """Preserve aspect ratio and prevent invalid colors leaking into resized RGB."""
        height, width = source.rgb.shape[:2]
        limit = self.analysis_settings.max_dimension
        if max(height, width) <= limit:
            return source.rgb, source.valid_mask
        scale = limit / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if source.valid_mask.all():
            visible = source.rgb
        else:
            visible = np.where(source.valid_mask[..., None], source.rgb, 0)
        rgb = cv2.resize(visible, size, interpolation=cv2.INTER_AREA)
        # Require every contributing source pixel to be valid. A nodata edge
        # should reduce evidence, never turn hidden colours into visible fire.
        validity = cv2.resize(
            source.valid_mask.astype(np.float32), size, interpolation=cv2.INTER_AREA
        )
        logger.warning(
            "Resizing to %sx%s for detection; small regions can be missed.", *size
        )
        return rgb, validity >= 1.0 - 1e-6

    @staticmethod
    def _validate_detection(
        detected: FireDetection, shape: tuple[int, ...], valid: np.ndarray
    ) -> None:
        """Enforce the detector boundary before consuming a future model's output."""
        if detected.mask.shape != shape or detected.mask.dtype != np.bool_:
            raise ImageProcessingError(
                "Detector mask must be boolean and match the analysis grid."
            )
        labels = detected.labels
        if labels.shape != shape or not np.issubdtype(labels.dtype, np.integer):
            raise ImageProcessingError(
                "Detector labels must be integers on the analysis grid."
            )
        if labels.min() < 0 or labels.max() > len(detected.regions):
            raise ImageProcessingError(
                "Detector component labels do not match its regions."
            )
        if not np.array_equal(detected.mask, labels > 0) or np.any(
            detected.mask & ~valid
        ):
            raise ImageProcessingError(
                "Detector mask/labels contain inconsistent or invalid pixels."
            )
        if not np.isfinite(detected.confidence) or not 0 <= detected.confidence <= 1:
            raise ImageProcessingError(
                "Detector confidence must be finite and between 0 and 1."
            )
        counts = np.bincount(
            labels.ravel().astype(np.int64), minlength=len(detected.regions) + 1
        )
        # Compute all component extents in one O(pixels) pass. Checking only
        # bounds would let a future detector place an in-bounds box elsewhere.
        rows, columns = np.nonzero(detected.mask)
        component_ids = labels[rows, columns]
        min_x = np.full(len(counts), shape[1], dtype=np.int32)
        min_y = np.full(len(counts), shape[0], dtype=np.int32)
        max_x = np.full(len(counts), -1, dtype=np.int32)
        max_y = np.full(len(counts), -1, dtype=np.int32)
        np.minimum.at(min_x, component_ids, columns)
        np.minimum.at(min_y, component_ids, rows)
        np.maximum.at(max_x, component_ids, columns)
        np.maximum.at(max_y, component_ids, rows)
        for index, region in enumerate(detected.regions, start=1):
            if (
                region.pixel_count != int(counts[index])
                or region.pixel_count < 1
                or region.x < 0
                or region.y < 0
                or region.width < 1
                or region.height < 1
                or region.x + region.width > shape[1]
                or region.y + region.height > shape[0]
                or not np.isfinite(region.confidence)
                or not 0 <= region.confidence <= 1
                or region.x != min_x[index]
                or region.y != min_y[index]
                or region.width != max_x[index] - min_x[index] + 1
                or region.height != max_y[index] - min_y[index] + 1
            ):
                raise ImageProcessingError(
                    "Detector region geometry, area, or confidence is invalid."
                )

    def _original_grid(
        self, detected: FireDetection, source: RGBImage
    ) -> tuple[np.ndarray, list[FireRegion]]:
        """Map labels and recompute boxes/areas from actual original-grid pixels."""
        if detected.mask.shape == source.valid_mask.shape:
            return detected.mask, detected.regions
        size = (source.metadata.width, source.metadata.height)
        expanded = cv2.resize(
            detected.labels.astype(np.int32), size, interpolation=cv2.INTER_NEAREST
        )
        expanded[~source.valid_mask] = 0
        mask = expanded > 0
        # Removing original invalid pixels may split a component. Recompute
        # connected regions so every reported box/area actually matches the mask.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=self.fire_settings.connectivity
        )
        parent_ids = np.zeros(count, dtype=np.int32)
        np.maximum.at(
            parent_ids, labels.ravel().astype(np.int32, copy=False), expanded.ravel()
        )
        scores = [0.0] + [region.confidence for region in detected.regions]
        regions = [
            FireRegion(
                x=int(stats[index, cv2.CC_STAT_LEFT]),
                y=int(stats[index, cv2.CC_STAT_TOP]),
                width=int(stats[index, cv2.CC_STAT_WIDTH]),
                height=int(stats[index, cv2.CC_STAT_HEIGHT]),
                pixel_count=int(stats[index, cv2.CC_STAT_AREA]),
                confidence=scores[int(parent_ids[index])],
            )
            for index in range(1, count)
        ]
        return mask, regions
