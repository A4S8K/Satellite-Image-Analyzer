"""Typer command-line interface for Satellite Image Analyzer."""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

from satellite_analyzer import __version__
from satellite_analyzer.analyzer import SatelliteAnalyzer
from satellite_analyzer.core.config import (
    PreprocessingSettings,
    load_preprocessing_settings,
)
from satellite_analyzer.core.exceptions import SatelliteAnalyzerError
from satellite_analyzer.core.fire_config import (
    load_analysis_settings,
    load_fire_settings,
)
from satellite_analyzer.io.image_loader import ImageLoader
from satellite_analyzer.io.metadata import extract_metadata
from satellite_analyzer.models import FIRE_LIMITATION
from satellite_analyzer.ndvi_cli import ndvi as ndvi_command
from satellite_analyzer.preprocessing.pipeline import PreprocessingPipeline
from satellite_analyzer.reporting.artifacts import save_analysis_artifacts
from satellite_analyzer.reporting.json_report import report_payload
from satellite_analyzer.utils.console import configure_cli_streams
from satellite_analyzer.utils.logger import configure_logging

app = typer.Typer(
    name="sat-analyzer",
    help="Inspect and preprocess satellite image files; analyze RGB fire-like regions.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True, markup=False)
app.command("ndvi")(ndvi_command)


def _handle_error(error: SatelliteAnalyzerError) -> None:
    error_console.print(f"Error: {error}")
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed application version."""

    console.print(f"Satellite Image Analyzer {__version__}")


@app.command()
def info(
    image_path: Annotated[Path, typer.Argument(help="Path to a supported image file.")],
) -> None:
    """Display standard image metadata and available raster details."""

    try:
        metadata = extract_metadata(ImageLoader().load(image_path))
    except SatelliteAnalyzerError as error:
        _handle_error(error)

    table = Table(title="Satellite Image Analyzer", show_header=False)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("File", metadata.filename)
    table.add_row("Format", metadata.format)
    table.add_row("Width", str(metadata.width))
    table.add_row("Height", str(metadata.height))
    table.add_row("Channels", str(metadata.channels))
    table.add_row("Dtype", metadata.dtype)
    table.add_row("File size", f"{metadata.file_size:,} bytes")
    if metadata.raster:
        table.add_row("Raster driver", str(metadata.raster["driver"]))
        table.add_row("CRS", str(metadata.raster["crs"] or "Not available"))
    console.print(table)


@app.command()
def analyze(
    image_path: Annotated[Path, typer.Argument(help="Path to visible RGB imagery.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Root directory for analysis runs.")
    ] = Path("results"),
    config: Annotated[
        Path | None, typer.Option(help="TOML file overriding fire/analysis thresholds.")
    ] = None,
    save_overlay: Annotated[
        bool, typer.Option(help="Save an overlay (included in every run).")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print only machine-readable JSON to stdout.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option(help="Enable debug logging to stderr.")
    ] = False,
) -> None:
    """Detect fire-like RGB regions and save images, statistics, and a JSON report."""
    configure_logging(verbose)
    try:
        fire_settings = load_fire_settings(config)
        analysis_settings = load_analysis_settings(config)
        result = SatelliteAnalyzer(fire_settings, analysis_settings).analyze(image_path)
        artifacts = save_analysis_artifacts(
            result, output, analysis_settings.overlay_alpha
        )
    except SatelliteAnalyzerError as error:
        _handle_error(error)
    except (OSError, ValueError) as error:
        _handle_error(SatelliteAnalyzerError(str(error)))
    if json_output:
        typer.echo(
            json.dumps(
                report_payload(result, artifacts), ensure_ascii=False, allow_nan=False
            )
        )
        return
    console.print("SATELLITE IMAGE ANALYSIS", markup=False)
    console.print(f"Image: {result.metadata.filename}", markup=False)
    console.print(f"Resolution: {result.width}x{result.height}")
    table = Table(title="Week 2: fire-like regions", show_header=True)
    table.add_column("Class")
    table.add_column("Pixels", justify="right")
    table.add_column("Area", justify="right")
    table.add_row(
        "Fire-like", f"{result.fire_pixel_count:,}", f"{result.fire_percentage:.2f}%"
    )
    table.add_row(
        "Unknown",
        f"{result.width * result.height - result.fire_pixel_count:,}",
        f"{100 - result.fire_percentage:.2f}%",
    )
    console.print(table)
    console.print(f"Fire confidence (evidence score): {result.fire_confidence:.1%}")
    console.print(f"Detected regions: {len(result.detected_fire_regions)}")
    console.print(f"Processing time: {result.processing_time_seconds:.3f}s")
    console.print(f"Report: {artifacts['report']}", markup=False)
    console.print(f"Summary: {artifacts['summary']}", markup=False)
    console.print(FIRE_LIMITATION, markup=False)


@app.command()
def preprocess(
    image_path: Annotated[Path, typer.Argument(help="Path to a supported image file.")],
    width: Annotated[int | None, typer.Option(help="Output width in pixels.")] = None,
    height: Annotated[int | None, typer.Option(help="Output height in pixels.")] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Destination image path.")
    ] = None,
    normalize: Annotated[
        bool | None,
        typer.Option("--normalize/--no-normalize", help="Scale values to 0..1."),
    ] = None,
    denoise: Annotated[
        bool | None,
        typer.Option("--denoise/--no-denoise", help="Apply gentle noise reduction."),
    ] = None,
) -> None:
    """Resize, normalize, and denoise an image using default or supplied settings."""

    try:
        source = ImageLoader().load(image_path)
        default_settings = load_preprocessing_settings()
        settings = PreprocessingSettings(
            target_width=width if width is not None else default_settings.target_width,
            target_height=(
                height if height is not None else default_settings.target_height
            ),
            normalize=(
                normalize if normalize is not None else default_settings.normalize
            ),
            denoise=denoise if denoise is not None else default_settings.denoise,
        )
        processed = PreprocessingPipeline(settings).process(source)
        output_path = output or _default_output_path(image_path)
        _save_image(processed.array, output_path)
    except SatelliteAnalyzerError as error:
        _handle_error(error)
    except (OSError, ValueError) as error:
        _handle_error(
            SatelliteAnalyzerError(f"Could not write processed image: {error}")
        )

    console.print(f"[green]Saved processed image:[/green] {output_path}")
    console.print(
        f"Dimensions: {processed.width} × {processed.height}; dtype: {processed.dtype}"
    )


def _default_output_path(image_path: Path) -> Path:
    """Return a portable default path beside the source image."""

    return image_path.with_name(f"{image_path.stem}_preprocessed.png")


def _save_image(array: np.ndarray, path: Path) -> None:
    """Save a processed array to a standard image format."""

    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise SatelliteAnalyzerError(
            "Output format must be PNG, JPG, JPEG, TIF, or TIFF."
        )
    if array.ndim not in {2, 3}:
        raise SatelliteAnalyzerError("Only 2D or 3D image arrays can be saved.")

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_prepare_output_array(array)).save(path)


def _prepare_output_array(array: np.ndarray) -> np.ndarray:
    """Convert normalized floating-point output to an image-writable representation."""

    if np.issubdtype(array.dtype, np.floating):
        return np.rint(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    return array


def main() -> None:
    """Run the command-line application."""

    configure_cli_streams()
    app()


if __name__ == "__main__":
    main()
