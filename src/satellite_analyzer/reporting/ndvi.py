"""Full-resolution, georeferenced NDVI and classification artifact export."""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio

from satellite_analyzer.core.exceptions import ImageProcessingError
from satellite_analyzer.vegetation.models import NDVIAnalysisResult
from satellite_analyzer.visualization.ndvi import save_ndvi_visualizations

logger = logging.getLogger(__name__)

RASTER_FILENAMES = {
    "ndvi_map": "ndvi_map.tif",
    "vegetation_classes": "vegetation_classes.tif",
    "statistics": "vegetation_statistics.json",
}
PREVIEW_FILENAMES = {
    "ndvi_heatmap": "ndvi_heatmap.png",
    "vegetation_classes_preview": "vegetation_classes.png",
    "ndvi_histogram": "ndvi_histogram.png",
}


def _source_paths(result: NDVIAnalysisResult) -> set[Path]:
    """Protect the source rasters and optional calibration metadata file."""
    sources = {
        Path(str(metadata["path"])).expanduser().resolve()
        for metadata in (result.pair.red_metadata, result.pair.nir_metadata)
    }
    if result.settings.metadata_path is not None:
        sources.add(result.settings.metadata_path.expanduser().resolve())
    return sources


def _write_rasters(result: NDVIAnalysisResult, paths: dict[str, Path]) -> None:
    """Keep values, valid-pixel masks, CRS, and affine transform on the source grid."""
    height, width = result.ndvi.values.shape
    common = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "crs": result.pair.crs,
        "transform": result.pair.transform,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    # Keeping the mask inside the TIFF prevents a detached .msk sidecar from
    # silently changing valid-pixel interpretation when results are copied.
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            paths["ndvi_map"], "w", dtype="float32", nodata=float("nan"), **common
        ) as dataset:
            dataset.write(result.ndvi.values, 1)
            dataset.write_mask(result.ndvi.valid_mask)
            dataset.set_band_description(1, "NDVI = (NIR - Red) / (NIR + Red)")
            dataset.update_tags(
                sensor=result.pair.sensor,
                units="dimensionless",
                valid_range="-1,1",
                grid="aligned Red reference grid; full resolution",
            )
        with rasterio.open(
            paths["vegetation_classes"], "w", dtype="uint8", nodata=0, **common
        ) as dataset:
            dataset.write(result.vegetation.classification, 1)
            dataset.write_mask(result.ndvi.valid_mask)
            dataset.set_band_description(1, "Configurable NDVI threshold classes")
            dataset.update_tags(
                class_labels=json.dumps(
                    {0: "Invalid"}
                    | {item.class_id: item.name for item in result.vegetation.classes}
                ),
                classification_basis="NDVI thresholds; not a trained land-cover model",
            )


def save_ndvi_artifacts(
    result: NDVIAnalysisResult,
    output_directory: str | Path,
    visualize: bool = False,
) -> dict[str, str]:
    """Export isolated run products without overwriting inputs or earlier outputs."""
    output = Path(output_directory).expanduser()
    sources = _source_paths(result)
    if output.resolve() in sources:
        raise ImageProcessingError("Output directory cannot be an input file path.")
    values = result.ndvi.values
    valid = result.ndvi.valid_mask
    classes = result.vegetation.classification
    if (
        values.ndim != 2
        or not values.size
        or values.dtype != np.float32
        or valid.shape != values.shape
        or valid.dtype != np.bool_
        or classes.shape != values.shape
        or classes.dtype != np.uint8
    ):
        raise ImageProcessingError("NDVI rasters and masks must share a nonempty grid.")
    filenames = RASTER_FILENAMES | (PREVIEW_FILENAMES if visualize else {})
    try:
        output.mkdir(parents=True, exist_ok=True)
        red_path = Path(str(result.pair.red_metadata["path"]))
        stem = re.sub(r"[^\w.-]+", "-", red_path.stem).strip(".-") or "image"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run = output / f"{stem[:60]}-ndvi-{timestamp}-{uuid4().hex[:8]}"
        run.mkdir()
        paths = {key: (run / filename).resolve() for key, filename in filenames.items()}
        if any(path in sources or path.exists() for path in paths.values()):
            raise ImageProcessingError(
                "Output would overwrite an input or prior artifact."
            )
        artifacts = {key: str(path) for key, path in paths.items()}
        # Serialize first so invalid report data cannot leave apparently complete
        # scientific rasters without their corresponding provenance report.
        payload = result.to_dict()
        payload["artifacts"] = artifacts
        serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        _write_rasters(result, paths)
        if visualize:
            save_ndvi_visualizations(result, paths)
        with paths["statistics"].open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        raise ImageProcessingError(f"Could not save NDVI artifacts: {error}") from error
    logger.info("Saved NDVI artifacts: %s", run)
    return artifacts
