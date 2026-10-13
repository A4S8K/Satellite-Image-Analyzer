"""Coordinate spectral I/O, real NDVI, and vegetation-condition analysis."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from satellite_analyzer.core.ndvi_config import NDVISettings
from satellite_analyzer.io.spectral_loader import SpectralBandLoader
from satellite_analyzer.vegetation.classification import analyze_vegetation
from satellite_analyzer.vegetation.models import NDVIAnalysisResult
from satellite_analyzer.vegetation.ndvi import compute_ndvi

logger = logging.getLogger(__name__)


class NDVIAnalyzer:
    """Analyze actual Red/NIR data independently of the RGB fire heuristic."""

    def __init__(self, settings: NDVISettings | None = None) -> None:
        self.settings = settings or NDVISettings()

    def analyze(
        self,
        *,
        red_path: Path | None = None,
        nir_path: Path | None = None,
        multiband_path: Path | None = None,
        red_band: int | None = None,
        nir_band: int | None = None,
    ) -> NDVIAnalysisResult:
        """Return georeferenced NDVI and valid-pixel vegetation statistics."""
        started = perf_counter()
        logger.info("Loading measured Red/NIR spectral bands")
        pair = SpectralBandLoader(self.settings).load(
            red_path=red_path,
            nir_path=nir_path,
            multiband_path=multiband_path,
            red_band=red_band,
            nir_band=nir_band,
        )
        logger.info(
            "Calculating NDVI on %sx%s pixels", pair.red.shape[1], pair.red.shape[0]
        )
        data = compute_ndvi(
            pair.red,
            pair.nir,
            pair.valid_mask,
            epsilon=self.settings.denominator_epsilon,
        )
        vegetation = analyze_vegetation(data.values, data.valid_mask, self.settings)
        elapsed = perf_counter() - started
        if vegetation.statistics.valid_pixels == 0:
            logger.warning("No valid NDVI pixels; numerical summaries are undefined")
        logger.info("NDVI analysis completed in %.3fs", elapsed)
        return NDVIAnalysisResult(
            pair=pair,
            ndvi=data,
            vegetation=vegetation,
            settings=self.settings,
            processing_time_seconds=elapsed,
            timestamp=datetime.now(UTC).isoformat(),
        )
