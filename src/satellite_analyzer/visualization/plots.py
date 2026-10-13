"""Headless Matplotlib figure generation for a fire-only RGB analysis."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from satellite_analyzer.models import AnalysisResult


def save_analysis_summary(
    result: AnalysisResult, overlay: np.ndarray, path: Path
) -> None:
    """Render original/overlay/statistics without opening a GUI on Windows."""
    figure = Figure(figsize=(15, 6), layout="constrained")
    FigureCanvasAgg(figure)
    original_axis, overlay_axis, stats_axis = figure.subplots(1, 3)
    original_axis.imshow(result.original_rgb)
    original_axis.set_title("Original visible RGB")
    overlay_axis.imshow(overlay)
    overlay_axis.set_title("Fire-like mask and region boxes")
    for axis in (original_axis, overlay_axis):
        axis.axis("off")
    values = [result.fire_percentage, 100 - result.fire_percentage]
    stats_axis.barh(["Fire-like", "Unknown"], values, color=["#ee5437", "#9eacb8"])
    stats_axis.set_xlim(0, 100)
    stats_axis.set_ylim(-0.6, 2.4)
    stats_axis.set_xlabel("Original image area (%)")
    stats_axis.set_title("Week 2: fire-like regions only")
    for index, value in enumerate(values):
        stats_axis.text(
            min(value + 2, 98),
            index,
            f"{value:.2f}%",
            ha="right" if value > 85 else "left",
            va="center",
        )
    stats_axis.text(
        0.0,
        0.98,
        f"Regions: {len(result.detected_fire_regions)}\n"
        f"Evidence score: {result.fire_confidence:.1%} (uncalibrated)\n"
        f"Processing: {result.processing_time_seconds:.3f}s\n"
        f"Detection grid: {result.analysis_width} x {result.analysis_height}",
        transform=stats_axis.transAxes,
        fontsize=10,
        va="top",
    )
    figure.suptitle(
        f"Satellite Image Analyzer · {result.metadata.filename}", fontsize=16
    )
    figure.supxlabel(
        "Visual RGB fire-like evidence only. "
        "Thermal fire confirmation requires appropriate sensor data.",
        fontsize=9,
    )
    figure.savefig(path, dpi=140, format="png")
    figure.clear()
