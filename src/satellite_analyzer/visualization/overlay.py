"""Non-destructive RGB annotation for detected fire-like components."""

import cv2
import numpy as np

from satellite_analyzer.detectors.base import FireRegion


def draw_regions(image_rgb: np.ndarray, regions: list[FireRegion]) -> np.ndarray:
    """Return an annotated copy with boxes matching exclusive region extents."""
    annotated = image_rgb.copy()
    thickness = max(1, round(max(image_rgb.shape[:2]) / 600))
    for region in regions:
        cv2.rectangle(
            annotated,
            (region.x, region.y),
            (region.x + region.width - 1, region.y + region.height - 1),
            (255, 50, 30),
            thickness,
        )
    return annotated


def create_fire_overlay(
    image_rgb: np.ndarray,
    fire_mask: np.ndarray,
    regions: list[FireRegion],
    alpha: float = 0.4,
) -> np.ndarray:
    """Tint only masked fire-like pixels, retaining all other image information."""
    if not 0 <= alpha <= 1:
        raise ValueError("overlay_alpha must be between 0 and 1")
    overlay = image_rgb.copy()
    overlay[fire_mask] = np.rint(
        overlay[fire_mask].astype(np.float32) * (1 - alpha)
        + np.array([255, 50, 30], dtype=np.float32) * alpha
    ).astype(np.uint8)
    return draw_regions(overlay, regions)
