"""Configurable initial NDVI classes and statistics over valid pixels only."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from satellite_analyzer.core.exceptions import ImageProcessingError
from satellite_analyzer.core.ndvi_config import NDVISettings

CLASS_NAMES = ("non_vegetation", "sparse", "low", "moderate", "dense")


@dataclass(frozen=True, slots=True)
class NDVIStatistics:
    """Numerical summaries; undefined values are null for an all-invalid image."""

    mean_ndvi: float | None
    min_ndvi: float | None
    max_ndvi: float | None
    median_ndvi: float | None
    std_ndvi: float | None
    valid_pixels: int
    invalid_pixels: int
    vegetation_coverage_percent: float | None
    vegetation_pixels: int


@dataclass(frozen=True, slots=True)
class VegetationClassStatistics:
    """Class area on the input raster, using valid pixels as the denominator."""

    class_id: int
    name: str
    pixel_count: int
    percentage: float | None


@dataclass(frozen=True, slots=True)
class VegetationAnalysis:
    """Class map plus numerical summaries; zero labels represent invalid pixels."""

    classification: NDArray[np.uint8]
    statistics: NDVIStatistics
    classes: list[VegetationClassStatistics]


def analyze_vegetation(
    ndvi: NDArray[np.float32],
    valid_mask: NDArray[np.bool_] | None = None,
    settings: NDVISettings | None = None,
) -> VegetationAnalysis:
    """Classify NDVI with explicit boundary rules and no universal accuracy claim.

    Defaults: non-vegetation [-1,0], sparse (0,.2), low [.2,.4),
    moderate [.4,.6), dense [.6,1]. Coverage is NDVI >= coverage_threshold.
    """
    config = settings or NDVISettings()
    if ndvi.ndim != 2 or not ndvi.size or not np.issubdtype(ndvi.dtype, np.floating):
        raise ImageProcessingError(
            "Vegetation analysis requires a nonempty floating 2D NDVI array"
        )
    if valid_mask is not None and (
        valid_mask.shape != ndvi.shape or valid_mask.dtype != np.bool_
    ):
        raise ImageProcessingError("Vegetation validity mask must match the NDVI grid")
    valid = np.isfinite(ndvi) & (ndvi >= -1) & (ndvi <= 1)
    if valid_mask is not None:
        valid &= valid_mask
    values = ndvi[valid]
    total_valid = int(values.size)
    classification = np.zeros(ndvi.shape, dtype=np.uint8)
    boundaries = np.array(
        [
            config.non_vegetation_max,
            config.sparse_max,
            config.low_max,
            config.moderate_max,
        ],
        dtype=np.float32,
    )
    if total_valid:
        classes = np.digitize(values, boundaries, right=False).astype(np.uint8) + 1
        classes[values <= config.non_vegetation_max] = 1
        classification[valid] = classes
        coverage = int(np.count_nonzero(values >= config.coverage_threshold))
        statistics = NDVIStatistics(
            mean_ndvi=float(np.mean(values, dtype=np.float64)),
            min_ndvi=float(values.min()),
            max_ndvi=float(values.max()),
            median_ndvi=float(np.median(values)),
            std_ndvi=float(np.std(values, dtype=np.float64, ddof=0)),
            valid_pixels=total_valid,
            invalid_pixels=int(ndvi.size - total_valid),
            vegetation_coverage_percent=100.0 * coverage / total_valid,
            vegetation_pixels=coverage,
        )
    else:
        statistics = NDVIStatistics(
            None, None, None, None, None, 0, int(ndvi.size), None, 0
        )
    counts = np.bincount(classification.ravel(), minlength=6)
    class_statistics = [
        VegetationClassStatistics(
            class_id=index,
            name=name,
            pixel_count=int(counts[index]),
            percentage=(
                100.0 * int(counts[index]) / total_valid if total_valid else None
            ),
        )
        for index, name in enumerate(CLASS_NAMES, start=1)
    ]
    classification.setflags(write=False)
    return VegetationAnalysis(classification, statistics, class_statistics)
