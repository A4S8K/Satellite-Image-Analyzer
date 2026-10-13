"""Export every analysis into its own directory, preserving source files."""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from satellite_analyzer.core.exceptions import ImageProcessingError
from satellite_analyzer.models import AnalysisResult
from satellite_analyzer.reporting.json_report import write_json_report
from satellite_analyzer.visualization.overlay import create_fire_overlay, draw_regions
from satellite_analyzer.visualization.plots import save_analysis_summary

logger = logging.getLogger(__name__)
ARTIFACT_FILENAMES = {
    "analyzed_image": "analyzed_image.png",
    "fire_mask": "fire_mask.png",
    "fire_overlay": "fire_overlay.png",
    "summary": "analysis_summary.png",
    "report": "analysis_report.json",
}


def save_analysis_artifacts(
    result: AnalysisResult, output_directory: str | Path, overlay_alpha: float = 0.4
) -> dict[str, str]:
    """Save annotated RGB, binary mask, overlay, figure, and report for one run."""
    output = Path(output_directory).expanduser()
    source = result.image_path.resolve()
    if output.resolve() == source:
        raise ImageProcessingError("Output directory cannot be the source image path.")
    if not 0 <= overlay_alpha <= 1:
        raise ImageProcessingError("overlay_alpha must be between 0 and 1.")
    try:
        output.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w.-]+", "-", source.stem).strip(".-") or "image"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_directory = output / f"{stem[:60]}-{timestamp}-{uuid4().hex[:8]}"
        run_directory.mkdir()
        paths = {
            key: (run_directory / filename).resolve()
            for key, filename in ARTIFACT_FILENAMES.items()
        }
        if any(path == source or path.exists() for path in paths.values()):
            raise ImageProcessingError(
                "Output would overwrite a source or existing artifact."
            )
        artifacts = {key: str(path) for key, path in paths.items()}
        overlay = create_fire_overlay(
            result.original_rgb,
            result.fire_mask,
            result.detected_fire_regions,
            overlay_alpha,
        )
        Image.fromarray(
            draw_regions(result.original_rgb, result.detected_fire_regions)
        ).save(paths["analyzed_image"])
        Image.fromarray(result.fire_mask.astype("uint8") * 255).save(paths["fire_mask"])
        Image.fromarray(overlay).save(paths["fire_overlay"])
        save_analysis_summary(result, overlay, paths["summary"])
        write_json_report(result, paths["report"], artifacts)
    except (OSError, ValueError) as error:
        raise ImageProcessingError(
            f"Could not save analysis artifacts: {error}"
        ) from error
    logger.info("Saved analysis artifacts: %s", run_directory)
    return artifacts
