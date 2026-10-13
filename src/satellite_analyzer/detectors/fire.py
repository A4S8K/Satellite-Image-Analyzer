"""Classical RGB fire-like region detection.

This is a visual fire-like region detector for RGB imagery, not a thermal
satellite fire detection system. Orange roofs, sunsets and bright soil can match
these thresholds; real fires hidden by clouds or smoke may be missed.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from satellite_analyzer.core.exceptions import ImageProcessingError
from satellite_analyzer.core.fire_config import FireDetectionSettings
from satellite_analyzer.detectors.base import BaseDetector, FireDetection, FireRegion


class RGBFireDetector(BaseDetector):
    """Find bright, saturated warm colors with red dominance using real pixels."""

    def __init__(self, settings: FireDetectionSettings | None = None) -> None:
        self.settings = settings if settings is not None else FireDetectionSettings()

    def detect(
        self,
        image_rgb: NDArray[np.uint8],
        *,
        valid_mask: NDArray[np.bool_] | None = None,
    ) -> FireDetection:
        """Return filtered connected regions and a non-probabilistic score.

        All filtering and scores use the input resolution. Label lookup and
        bincount avoid scanning the whole image once for every region.
        """

        self._validate_image(image_rgb)
        if valid_mask is not None and (
            not isinstance(valid_mask, np.ndarray)
            or valid_mask.dtype != np.bool_
            or valid_mask.shape != image_rgb.shape[:2]
        ):
            raise ImageProcessingError(
                "valid_mask must be a boolean array matching the RGB image height/width"
            )
        hsv = np.asarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV), dtype=np.uint8)
        candidates = self._candidate_mask(image_rgb, hsv)
        if valid_mask is not None:
            # Exclude transparency/nodata *before* components so invalid pixels
            # cannot bridge otherwise separate visual regions.
            candidates &= valid_mask
        if self.settings.morphology_open_kernel:
            size = self.settings.morphology_open_kernel
            kernel = np.ones((size, size), dtype=np.uint8)
            opened = cv2.morphologyEx(
                candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel
            )
            # Border handling must never introduce pixels lacking RGB evidence.
            candidates &= opened.astype(bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidates.astype(np.uint8), connectivity=self.settings.connectivity
        )
        keep = stats[:, cv2.CC_STAT_AREA] >= self.settings.min_region_pixels
        keep[0] = False  # Label zero is the background, regardless of its size.
        retained = np.flatnonzero(keep)
        label_map = np.zeros(count, dtype=np.int32)
        label_map[retained] = np.arange(1, len(retained) + 1, dtype=np.int32)
        output_labels = label_map[labels]
        mask = output_labels > 0
        output_labels.setflags(write=False)
        if not mask.any():
            mask.setflags(write=False)
            return FireDetection(
                mask=mask, labels=output_labels, confidence=0.0, regions=[]
            )

        scores = self._evidence_scores(image_rgb[mask], hsv[mask])
        # Sum each retained pixel's score by label, in a single vectorized pass.
        score_sums = np.bincount(
            output_labels[mask], weights=scores, minlength=len(retained) + 1
        )
        regions = [
            FireRegion(
                x=int(stats[label, cv2.CC_STAT_LEFT]),
                y=int(stats[label, cv2.CC_STAT_TOP]),
                width=int(stats[label, cv2.CC_STAT_WIDTH]),
                height=int(stats[label, cv2.CC_STAT_HEIGHT]),
                pixel_count=int(stats[label, cv2.CC_STAT_AREA]),
                confidence=float(
                    score_sums[output_id] / stats[label, cv2.CC_STAT_AREA]
                ),
            )
            for output_id, label in enumerate(retained, start=1)
        ]
        mask.setflags(write=False)
        return FireDetection(
            mask=mask,
            labels=output_labels,
            confidence=float(scores.mean()),
            regions=regions,
        )

    @staticmethod
    def _validate_image(image_rgb: NDArray[np.uint8]) -> None:
        """Fail clearly when direct callers bypass the application's loader."""

        if (
            not isinstance(image_rgb, np.ndarray)
            or image_rgb.dtype != np.uint8
            or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or image_rgb.shape[0] == 0
            or image_rgb.shape[1] == 0
        ):
            raise ImageProcessingError(
                "Fire detection requires a nonempty H x W x 3 uint8 RGB image"
            )

    def _candidate_mask(
        self, image_rgb: NDArray[np.uint8], hsv: NDArray[np.uint8]
    ) -> NDArray[np.bool_]:
        """Require HSV warmth/brightness and RGB red-dominance jointly."""

        settings = self.settings
        hue, saturation, value = cv2.split(hsv)
        red, green, blue = cv2.split(image_rgb)
        warm_hue = ((hue >= settings.hue_min) & (hue <= settings.hue_max)) | (
            (hue >= settings.red_hue_min) & (hue <= settings.red_hue_max)
        )
        return (
            warm_hue
            & (saturation >= settings.min_saturation)
            & (value >= settings.min_value)
            & (red >= settings.min_red)
            & (
                red
                >= np.multiply(green, settings.min_red_green_ratio, dtype=np.float32)
            )
            & (red >= np.multiply(blue, settings.min_red_blue_ratio, dtype=np.float32))
        )

    def _evidence_scores(
        self, rgb: NDArray[np.uint8], hsv: NDArray[np.uint8]
    ) -> NDArray[np.float32]:
        """Average five normalized margins of retained candidate pixels.

        The margins describe hue centrality, saturation, brightness, red level,
        and red-over-blue dominance. Equal weighting is deliberately interpretable
        and is not fitted or calibrated against true satellite fire labels.
        """

        settings = self.settings
        rgb_float = rgb.astype(np.float32)
        hsv_float = hsv.astype(np.float32)
        hue = hsv_float[:, 0]
        # OpenCV red wraps from 179 to 0. When the ranges meet at that boundary,
        # unwrap them into one interval so pure red is not scored as a boundary.
        if settings.hue_min == 0 and settings.red_hue_max == 179:
            hue = np.where(hue >= settings.red_hue_min, hue - 180, hue)
            hue_strength = self._interval_strength(
                hue, settings.red_hue_min - 180, settings.hue_max
            )
        else:
            hue_strength = np.maximum(
                self._interval_strength(hue, settings.hue_min, settings.hue_max),
                self._interval_strength(
                    hue, settings.red_hue_min, settings.red_hue_max
                ),
            )
        saturation_strength = self._threshold_margin(
            hsv_float[:, 1], settings.min_saturation
        )
        value_strength = self._threshold_margin(hsv_float[:, 2], settings.min_value)
        red_strength = self._threshold_margin(rgb_float[:, 0], settings.min_red)
        dominance_strength = np.clip(
            (rgb_float[:, 0] - settings.min_red_blue_ratio * rgb_float[:, 2]) / 255,
            0,
            1,
        )
        return np.asarray(
            (
                hue_strength
                + saturation_strength
                + value_strength
                + red_strength
                + dominance_strength
            )
            / 5,
            dtype=np.float32,
        )

    @staticmethod
    def _threshold_margin(
        channel: NDArray[np.float32], minimum: int
    ) -> NDArray[np.float32]:
        if minimum == 255:
            return np.ones_like(channel)
        return np.clip((channel - minimum) / (255 - minimum), 0, 1)

    @staticmethod
    def _interval_strength(
        hue: NDArray[np.float32], lower: int, upper: int
    ) -> NDArray[np.float32]:
        if lower == upper:
            return (hue == lower).astype(np.float32)
        center = (lower + upper) / 2
        return np.clip(1 - np.abs(hue - center) / ((upper - lower) / 2), 0, 1)
