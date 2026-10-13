"""Replaceable detector contracts for RGB and future trained implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FireRegion:
    """One connected region; coordinates and area refer to analysis pixels.

    Bounding boxes use an exclusive right/bottom edge: ``x + width`` and
    ``y + height``. ``confidence`` is an evidence score in [0, 1].
    """

    x: int
    y: int
    width: int
    height: int
    pixel_count: int
    confidence: float


@dataclass(frozen=True, slots=True)
class FireDetection:
    """Detected pixel mask, area-weighted evidence score, and regions.

    A frozen result prevents field reassignment; the arrays are also read-only.
    Labels are contiguous IDs 1..len(regions), zero denotes background, and
    ``regions[label - 1]`` describes each retained component.
    """

    mask: NDArray[np.bool_]
    labels: NDArray[np.int32]
    confidence: float
    regions: list[FireRegion]


class BaseDetector(ABC):
    """Contract a future segmentation or trained fire detector can implement."""

    @abstractmethod
    def detect(
        self,
        image_rgb: NDArray[np.uint8],
        *,
        valid_mask: NDArray[np.bool_] | None = None,
    ) -> FireDetection:
        """Inspect uint8 RGB pixels, excluding an optional H x W validity mask."""
