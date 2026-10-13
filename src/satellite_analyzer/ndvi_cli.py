"""Real NDVI command integrated into the existing Typer application."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from satellite_analyzer.core.exceptions import SatelliteAnalyzerError
from satellite_analyzer.core.ndvi_config import NDVISettings, load_ndvi_settings
from satellite_analyzer.reporting.ndvi import save_ndvi_artifacts
from satellite_analyzer.utils.logger import configure_logging
from satellite_analyzer.vegetation.analyzer import NDVIAnalyzer


def ndvi(
    red: Annotated[Path | None, typer.Option(help="Single-band Red GeoTIFF.")] = None,
    nir: Annotated[Path | None, typer.Option(help="Single-band NIR GeoTIFF.")] = None,
    multiband: Annotated[
        Path | None, typer.Option(help="Multispectral GeoTIFF with explicit mapping.")
    ] = None,
    red_band: Annotated[
        int | None, typer.Option(min=1, help="1-based Red index in multiband input.")
    ] = None,
    nir_band: Annotated[
        int | None, typer.Option(min=1, help="1-based NIR index in multiband input.")
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Root for separate NDVI runs.")
    ] = Path("results/ndvi"),
    sensor: Annotated[
        str | None, typer.Option(help="sentinel-2, landsat-8, landsat-9, or custom.")
    ] = None,
    metadata: Annotated[
        Path | None,
        typer.Option(help="Product Sentinel XML or Landsat Level-2 MTL.txt."),
    ] = None,
    input_units: Annotated[
        str | None,
        typer.Option(help="auto, reflectance, or dn; never guesses DN calibration."),
    ] = None,
    red_scale: Annotated[
        float | None, typer.Option(help="Explicit Red multiplicative scale.")
    ] = None,
    red_offset: Annotated[
        float | None, typer.Option(help="Explicit Red additive offset.")
    ] = None,
    nir_scale: Annotated[
        float | None, typer.Option(help="Explicit NIR multiplicative scale.")
    ] = None,
    nir_offset: Annotated[
        float | None, typer.Option(help="Explicit NIR additive offset.")
    ] = None,
    align_nir: Annotated[
        bool,
        typer.Option("--align-nir", help="Explicitly reproject NIR to the Red grid."),
    ] = False,
    resampling: Annotated[
        str | None, typer.Option(help="NIR alignment method: nearest or bilinear.")
    ] = None,
    max_pixels: Annotated[
        int | None, typer.Option(min=1, help="Maximum pixels per input band.")
    ] = None,
    config: Annotated[
        Path | None, typer.Option(help="TOML overrides in the NDVI section.")
    ] = None,
    visualize: Annotated[
        bool,
        typer.Option(
            "--visualize", help="Save heatmap, class map, and histogram PNGs."
        ),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Also print only report JSON to stdout.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Enable debug diagnostics on stderr.")
    ] = False,
) -> None:
    """Compute real Red/NIR NDVI, vegetation statistics, and georeferenced outputs."""
    configure_logging(verbose)
    try:
        defaults = load_ndvi_settings(config)
        overrides = {
            key: value
            for key, value in {
                "sensor": sensor,
                "metadata_path": metadata,
                "input_units": input_units,
                "red_scale": red_scale,
                "red_offset": red_offset,
                "nir_scale": nir_scale,
                "nir_offset": nir_offset,
                "resampling": resampling,
                "max_pixels": max_pixels,
            }.items()
            if value is not None
        }
        if align_nir:
            overrides["alignment"] = "reproject"
        settings = NDVISettings.model_validate({**defaults.model_dump(), **overrides})
        result = NDVIAnalyzer(settings).analyze(
            red_path=red,
            nir_path=nir,
            multiband_path=multiband,
            red_band=red_band,
            nir_band=nir_band,
        )
        artifacts = save_ndvi_artifacts(result, output, visualize=visualize)
        payload = result.to_dict()
        payload["artifacts"] = artifacts
    except (SatelliteAnalyzerError, OSError, ValueError) as error:
        Console(stderr=True, markup=False).print(f"Error: {error}")
        raise typer.Exit(code=1) from error
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return
    console = Console(markup=False)
    stats = result.vegetation.statistics
    console.print("VEGETATION / REAL NDVI ANALYSIS")
    console.print(
        f"Sensor: {result.pair.sensor}; valid pixels: {stats.valid_pixels:,}; "
        f"invalid pixels: {stats.invalid_pixels:,}"
    )
    if stats.mean_ndvi is None:
        console.print("No valid pixels: NDVI statistics and coverage are undefined.")
    else:
        console.print(
            f"Mean NDVI: {stats.mean_ndvi:.4f}; "
            f"coverage: {stats.vegetation_coverage_percent:.2f}%"
        )
    table = Table(title="Initial vegetation classes (valid pixels only)")
    table.add_column("Class")
    table.add_column("Pixels", justify="right")
    table.add_column("Percentage", justify="right")
    for item in result.vegetation.classes:
        percentage = (
            f"{item.percentage:.2f}%" if item.percentage is not None else "undefined"
        )
        table.add_row(item.name, f"{item.pixel_count:,}", percentage)
    console.print(table)
    console.print(f"GeoTIFF: {artifacts['ndvi_map']}")
    console.print(f"JSON: {artifacts['statistics']}")
    console.print(
        "NDVI indicates spectral vegetation condition; "
        "it does not directly detect fire or measure moisture."
    )
