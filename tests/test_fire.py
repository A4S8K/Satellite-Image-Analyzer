"""Deterministic positive and negative controls for RGB fire heuristics."""

import cv2
import numpy as np
import pytest

from satellite_analyzer.core.exceptions import ImageProcessingError
from satellite_analyzer.core.fire_config import FireDetectionSettings
from satellite_analyzer.detectors import BaseDetector, RGBFireDetector


@pytest.mark.parametrize("color", [(255, 0, 0), (255, 110, 0), (255, 255, 0)])
def test_bright_warm_colors_are_detected(color: tuple[int, int, int]) -> None:
    image = np.zeros((24, 30, 3), dtype=np.uint8)
    image[5:11, 8:15] = color
    original = image.copy()
    detector = RGBFireDetector()
    assert isinstance(detector, BaseDetector)
    detection = detector.detect(image)
    assert detection.mask.dtype == np.bool_
    assert int(detection.mask.sum()) == 42
    assert len(detection.regions) == 1
    region = detection.regions[0]
    assert (region.x, region.y, region.width, region.height) == (8, 5, 7, 6)
    assert region.pixel_count == 42
    assert 0 < detection.confidence <= 1
    assert region.confidence == pytest.approx(detection.confidence)
    assert np.array_equal(image, original)
    assert not detection.mask.flags.writeable
    assert not detection.labels.flags.writeable
    assert np.array_equal(detection.mask, detection.labels > 0)
    assert detection.labels.max() == 1


@pytest.mark.parametrize(
    "color",
    [(0, 220, 0), (0, 0, 255), (0, 200, 255), (255, 255, 255), (100, 40, 0)],
)
def test_negative_color_controls(color: tuple[int, int, int]) -> None:
    image = np.full((16, 16, 3), color, dtype=np.uint8)
    detection = RGBFireDetector().detect(image)
    assert not detection.mask.any()
    assert detection.regions == []
    assert detection.confidence == 0


def test_red_hue_wrap_is_detected() -> None:
    hsv = np.full((6, 6, 3), (175, 255, 255), dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    assert RGBFireDetector().detect(image).mask.all()


def test_minimum_region_area_removes_isolated_noise() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[4:8, 7:11] = (255, 100, 0)
    image[20, 20] = (255, 0, 0)
    image[22:25, 21:24] = (255, 255, 0)
    detection = RGBFireDetector().detect(image)
    assert len(detection.regions) == 1
    assert detection.regions[0].pixel_count == 16
    assert not detection.mask[20, 20]
    assert int(detection.mask.sum()) == 16


def test_connectivity_changes_diagonal_region_grouping() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[1, 1] = image[2, 2] = (255, 100, 0)
    connected = RGBFireDetector(
        FireDetectionSettings(min_region_pixels=2, connectivity=8)
    ).detect(image)
    separated = RGBFireDetector(
        FireDetectionSettings(min_region_pixels=2, connectivity=4)
    ).detect(image)
    assert len(connected.regions) == 1
    assert connected.regions[0].pixel_count == 2
    assert not separated.mask.any()


def test_morphology_cannot_add_nonfire_pixels() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[2:10, 2:10] = (255, 100, 0)
    image[6, 6] = (0, 255, 0)
    image[13, 13] = (255, 100, 0)
    raw = RGBFireDetector(FireDetectionSettings(min_region_pixels=1)).detect(image)
    opened = RGBFireDetector(
        FireDetectionSettings(min_region_pixels=1, morphology_open_kernel=3)
    ).detect(image)
    assert not np.any(opened.mask & ~raw.mask)
    assert not opened.mask[6, 6]
    assert not opened.mask[13, 13]
    assert opened.mask.any()


def test_confidence_is_area_weighted_evidence() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[1:5, 1:5] = (255, 110, 0)
    image[8:14, 8:14] = (180, 60, 0)
    detection = RGBFireDetector().detect(image)
    assert len(detection.regions) == 2
    assert detection.regions[0].confidence > detection.regions[1].confidence
    expected = sum(r.confidence * r.pixel_count for r in detection.regions) / 52
    assert detection.confidence == pytest.approx(expected)


def test_stricter_threshold_changes_detection() -> None:
    image = np.full((8, 8, 3), (200, 100, 0), dtype=np.uint8)
    assert RGBFireDetector().detect(image).mask.all()
    assert (
        not RGBFireDetector(FireDetectionSettings(min_red=230)).detect(image).mask.any()
    )


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((5, 5), dtype=np.uint8),
        np.zeros((5, 5, 4), dtype=np.uint8),
        np.zeros((5, 5, 3), dtype=np.float32),
        np.zeros((0, 5, 3), dtype=np.uint8),
        np.zeros((5, 0, 3), dtype=np.uint8),
    ],
)
def test_invalid_detector_input(image: np.ndarray) -> None:
    with pytest.raises(ImageProcessingError, match="uint8 RGB"):
        RGBFireDetector().detect(image)


def test_noncontiguous_input_is_supported() -> None:
    image = np.full((8, 16, 3), (255, 120, 0), dtype=np.uint8)[:, ::2]
    assert not image.flags.c_contiguous
    assert RGBFireDetector().detect(image).mask.all()


def test_invalid_pixels_cannot_bridge_fire_regions() -> None:
    image = np.full((6, 8, 3), (255, 100, 0), dtype=np.uint8)
    valid = np.ones((6, 8), dtype=bool)
    valid[:, 3:5] = False
    original = valid.copy()
    detection = RGBFireDetector().detect(image, valid_mask=valid)
    assert len(detection.regions) == 2
    assert [r.pixel_count for r in detection.regions] == [18, 18]
    assert not detection.mask[:, 3:5].any()
    assert np.array_equal(valid, original)
    assert np.array_equal(np.unique(detection.labels), [0, 1, 2])


def test_filtered_regions_have_contiguous_output_labels() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[1, 1] = (255, 0, 0)
    image[4:8, 4:8] = (255, 100, 0)
    image[10, 10] = (255, 0, 0)
    image[13:17, 13:17] = (255, 100, 0)
    detection = RGBFireDetector().detect(image)
    assert np.array_equal(np.unique(detection.labels), [0, 1, 2])
    assert detection.labels[4, 4] == 1
    assert detection.labels[13, 13] == 2


def test_all_invalid_pixels_produce_empty_detection() -> None:
    image = np.full((6, 6, 3), (255, 100, 0), dtype=np.uint8)
    detection = RGBFireDetector().detect(image, valid_mask=np.zeros((6, 6), dtype=bool))
    assert not detection.mask.any()
    assert not detection.labels.any()
    assert detection.regions == []
    assert detection.confidence == 0


def test_weak_evidence_scores_below_strong_evidence() -> None:
    settings = FireDetectionSettings(min_region_pixels=1)
    weak = np.full((2, 2, 3), (140, 100, 75), dtype=np.uint8)
    strong = np.full((2, 2, 3), (255, 110, 0), dtype=np.uint8)
    weak_result = RGBFireDetector(settings).detect(weak)
    strong_result = RGBFireDetector(settings).detect(strong)
    assert weak_result.mask.all()
    assert 0 <= weak_result.confidence < strong_result.confidence <= 1


def test_morphology_rejects_warm_noise_at_image_border() -> None:
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[0, 0] = (255, 100, 0)
    result = RGBFireDetector(
        FireDetectionSettings(min_region_pixels=1, morphology_open_kernel=3)
    ).detect(image)
    assert not result.mask.any()


@pytest.mark.parametrize(
    "valid",
    [np.ones((6, 6), dtype=np.uint8), np.ones((3, 3), dtype=bool)],
)
def test_invalid_validity_mask(valid: np.ndarray) -> None:
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    with pytest.raises(ImageProcessingError, match="valid_mask"):
        RGBFireDetector().detect(image, valid_mask=valid)
