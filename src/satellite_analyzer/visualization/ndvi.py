"""Headless NDVI previews; scientific rasters retain their original grid."""

from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from satellite_analyzer.vegetation.models import NDVIAnalysisResult

CLASS_COLORS = ("#9aa0a6", "#a6611a", "#ffe082", "#b7df7d", "#4f9c47", "#005a32")


def _preview(values: np.ndarray, max_dimension: int) -> np.ndarray:
    """Limit display memory without averaging classes or modifying science data."""
    height, width = values.shape
    if max(height, width) <= max_dimension:
        return values
    scale = max_dimension / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(values, size, interpolation=cv2.INTER_NEAREST)


def _figure(width: float, height: float) -> Figure:
    """Use an Agg canvas directly rather than initializing an interactive backend."""
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _save(figure: Figure, path: Path) -> None:
    """Release figure artists even when writing the destination fails."""
    try:
        figure.savefig(path, dpi=140, format="png")
    finally:
        figure.clear()


def _pixel_extent(result: NDVIAnalysisResult) -> tuple[int, int, int, int]:
    """Label preview coordinates as source pixels, including for rotated rasters."""
    height, width = result.ndvi.values.shape
    return (0, width, height, 0)


def save_ndvi_heatmap(result: NDVIAnalysisResult, path: Path) -> None:
    """Save a fixed -1..1 NDVI scale with masked pixels shown in gray."""
    preview = _preview(result.ndvi.values, result.settings.preview_max_dimension)
    colors = LinearSegmentedColormap.from_list(
        "satellite_ndvi",
        [
            (0.0, "#2166ac"),
            (0.4999, "#9bd8f2"),
            (0.5, "#a6611a"),
            (0.6, "#ffe082"),
            (0.75, "#b7df7d"),
            (1.0, "#005a32"),
        ],
    ).with_extremes(bad=CLASS_COLORS[0])
    figure = _figure(9, 6)
    axis = figure.subplots()
    plotted = axis.imshow(
        np.ma.masked_invalid(preview),
        cmap=colors,
        vmin=-1,
        vmax=1,
        extent=_pixel_extent(result),
        interpolation="nearest",
    )
    axis.set_title(f"NDVI from calibrated Red and NIR · {result.pair.sensor}")
    axis.set_xlabel("Source pixel column")
    axis.set_ylabel("Source pixel row")
    figure.colorbar(plotted, ax=axis, label="NDVI", ticks=[-1, -0.5, 0, 0.5, 1])
    figure.supxlabel("Gray: invalid data. Preview only; GeoTIFF retains the full grid.")
    _save(figure, path)


def save_vegetation_classes(result: NDVIAnalysisResult, path: Path) -> None:
    """Render discrete classes without interpolating class IDs into other classes."""
    preview = _preview(
        result.vegetation.classification, result.settings.preview_max_dimension
    )
    names = {
        0: "Invalid",
        1: "Non-vegetation",
        2: "Sparse vegetation",
        3: "Low vegetation",
        4: "Moderate vegetation",
        5: "Dense vegetation",
    }
    for item in result.vegetation.classes:
        names[item.class_id] = item.name.replace("_", " ").title()
    figure = _figure(10, 6)
    axis = figure.subplots()
    axis.imshow(
        preview,
        cmap=ListedColormap(CLASS_COLORS),
        vmin=-0.5,
        vmax=5.5,
        extent=_pixel_extent(result),
        interpolation="nearest",
    )
    axis.set_title("NDVI vegetation classes")
    axis.set_xlabel("Source pixel column")
    axis.set_ylabel("Source pixel row")
    axis.legend(
        handles=[
            Patch(color=CLASS_COLORS[index], label=names[index]) for index in names
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    figure.supxlabel(
        "Threshold classes are configurable; labels are not land-cover truth."
    )
    _save(figure, path)


def save_ndvi_histogram(result: NDVIAnalysisResult, path: Path) -> None:
    """Count every valid source pixel, independently of preview downsampling."""
    stats = result.vegetation.statistics
    figure = _figure(10, 5)
    axis = figure.subplots()
    if stats.valid_pixels:
        counts, edges = _full_histogram(result)
        axis.bar(
            (edges[:-1] + edges[1:]) / 2,
            counts,
            width=np.diff(edges),
            color="#4f8c42",
            edgecolor="white",
            linewidth=0.3,
        )
        if stats.mean_ndvi is not None:
            axis.axvline(
                stats.mean_ndvi,
                color="#b33b2e",
                linestyle="--",
                label=f"Mean NDVI: {stats.mean_ndvi:.3f}",
            )
            axis.legend(loc="upper left")
    else:
        axis.text(
            0.5,
            0.5,
            "No valid NDVI pixels",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_ylim(0, 1)
    axis.set_xlim(-1, 1)
    axis.set_xlabel("NDVI")
    axis.set_ylabel("Pixel count (all valid source pixels)")
    axis.set_title("NDVI distribution")
    coverage = (
        f"{stats.vegetation_coverage_percent:.2f}%"
        if stats.vegetation_coverage_percent is not None
        else "Not available"
    )
    figure.supxlabel(
        f"Valid pixels: {stats.valid_pixels:,} · Invalid: {stats.invalid_pixels:,} · "
        "Vegetation coverage (NDVI >= "
        f"{result.settings.coverage_threshold:g}): {coverage}"
    )
    _save(figure, path)


def _full_histogram(result: NDVIAnalysisResult) -> tuple[np.ndarray, np.ndarray]:
    """Count all valid values without allocating a full-tile indexed copy."""
    values, valid = result.ndvi.values, result.ndvi.valid_mask
    bins = result.settings.histogram_bins
    counts = np.zeros(bins, dtype=np.int64)
    edges = np.linspace(-1, 1, bins + 1)
    rows_per_block = max(1, 1_000_000 // values.shape[1])
    for start in range(0, values.shape[0], rows_per_block):
        stop = start + rows_per_block
        block = values[start:stop][valid[start:stop]]
        block_counts, _ = np.histogram(block, bins=edges)
        counts += block_counts
    return counts, edges


def save_ndvi_visualizations(
    result: NDVIAnalysisResult, paths: dict[str, Path]
) -> None:
    """Save the three independent, bounded-memory PNG visualization products."""
    save_ndvi_heatmap(result, paths["ndvi_heatmap"])
    save_vegetation_classes(result, paths["vegetation_classes_preview"])
    save_ndvi_histogram(result, paths["ndvi_histogram"])
