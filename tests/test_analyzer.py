"""End-to-end fire analysis, coordinate integrity, and replaceable detector tests."""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from satellite_analyzer.analyzer import SatelliteAnalyzer
from satellite_analyzer.core.exceptions import ImageProcessingError, InvalidImageError
from satellite_analyzer.core.fire_config import AnalysisSettings
from satellite_analyzer.detectors.base import BaseDetector, FireDetection, FireRegion


def _save_rgb(path: Path, pixels: np.ndarray) -> Path:
    """Create a deterministic local fixture without any downloaded imagery."""

    Image.fromarray(pixels).save(path)
    return path


def test_original_resolution_statistics_and_input_preservation(tmp_path: Path) -> None:
    """Area and boxes describe the input pixels, and analysis preserves the input."""

    pixels = np.full((32, 60, 3), (15, 40, 80), dtype=np.uint8)
    pixels[5:17, 8:25] = (255, 150, 10)
    path = _save_rgb(tmp_path / "visible_fire.png", pixels)
    original_file = path.read_bytes()

    result = SatelliteAnalyzer().analyze(path)

    assert (result.width, result.height) == (60, 32)
    assert (result.analysis_width, result.analysis_height) == (60, 32)
    assert result.fire_mask.shape == pixels.shape[:2]
    assert result.fire_mask.dtype == np.bool_
    assert result.fire_pixel_count == 12 * 17
    assert result.fire_percentage == pytest.approx(100 * 12 * 17 / (32 * 60))
    assert result.invalid_pixel_count == 0
    assert len(result.detected_fire_regions) == 1
    region = result.detected_fire_regions[0]
    assert (region.x, region.y, region.width, region.height) == (8, 5, 17, 12)
    assert region.pixel_count == result.fire_pixel_count
    assert 0 < result.fire_confidence <= 1
    np.testing.assert_array_equal(result.original_rgb, pixels)
    assert path.read_bytes() == original_file
    with Image.open(path) as saved_image:
        np.testing.assert_array_equal(np.asarray(saved_image), pixels)


def test_no_fire_reports_all_pixels_as_unknown(tmp_path: Path) -> None:
    """The Week 2 result must not fabricate future vegetation/water classes."""

    pixels = np.full((19, 29, 3), (20, 90, 200), dtype=np.uint8)
    path = _save_rgb(tmp_path / "no_fire.png", pixels)

    result = SatelliteAnalyzer().analyze(path)
    report = result.to_dict()

    assert result.fire_pixel_count == 0
    assert result.fire_percentage == 0.0
    assert result.fire_confidence == 0.0
    assert result.detected_fire_regions == []
    assert not result.fire_mask.any()
    assert report["unknown"] == {"pixel_count": 19 * 29, "percentage": 100.0}
    assert "vegetation" not in report
    assert "water" not in report
    assert "urban" not in report


def test_report_is_finite_json_and_has_utc_timestamp(tmp_path: Path) -> None:
    """Reports serialize without array objects or nonstandard NaN/Infinity values."""

    pixels = np.full((12, 20, 3), (255, 145, 5), dtype=np.uint8)
    result = SatelliteAnalyzer().analyze(_save_rgb(tmp_path / "all_warm.png", pixels))

    report = result.to_dict()
    decoded = json.loads(json.dumps(report, allow_nan=False))

    assert decoded["resolution"] == {"width": 20, "height": 12}
    assert decoded["analysis_resolution"] == {"width": 20, "height": 12}
    assert decoded["fire"]["region_count"] == 1
    assert decoded["fire"]["pixel_count"] == 240
    assert decoded["unknown"]["pixel_count"] == 0
    assert decoded["metadata"]["channels"] == 3
    assert decoded["parameters"]["detector"] == "RGBFireDetector"
    assert decoded["limitations"]
    assert "not a thermal satellite fire detection system" in " ".join(
        decoded["limitations"]
    )
    assert np.isfinite(result.processing_time_seconds)
    assert result.processing_time_seconds >= 0
    timestamp = datetime.fromisoformat(decoded["timestamp"])
    assert timestamp.utcoffset() == timedelta(0)


def test_noninteger_downsampling_preserves_original_grid_region_geometry(
    tmp_path: Path,
) -> None:
    """Fractional resize ratios must not produce rounded-box/area discrepancies."""

    pixels = np.zeros((77, 109, 3), dtype=np.uint8)
    pixels[12:36, 14:46] = (255, 150, 10)
    pixels[45:68, 70:97] = (255, 150, 10)
    path = _save_rgb(tmp_path / "wide.png", pixels)

    result = SatelliteAnalyzer(
        analysis_settings=AnalysisSettings(max_dimension=32)
    ).analyze(path)

    assert (result.width, result.height) == (109, 77)
    assert (result.analysis_width, result.analysis_height) == (32, 23)
    assert result.fire_mask.shape == (77, 109)
    assert len(result.detected_fire_regions) == 2
    assert result.fire_pixel_count == int(result.fire_mask.sum())
    assert result.fire_percentage == pytest.approx(100 * result.fire_mask.mean())
    assert sum(r.pixel_count for r in result.detected_fire_regions) == (
        result.fire_pixel_count
    )
    covered = np.zeros(result.fire_mask.shape, dtype=bool)
    for region in result.detected_fire_regions:
        assert 0 <= region.x < region.x + region.width <= result.width
        assert 0 <= region.y < region.y + region.height <= result.height
        crop = result.fire_mask[
            region.y : region.y + region.height,
            region.x : region.x + region.width,
        ]
        assert int(crop.sum()) == region.pixel_count
        # Each edge must touch a detected pixel: these are tight pixel bounds.
        assert crop[0].any() and crop[-1].any()
        assert crop[:, 0].any() and crop[:, -1].any()
        covered[
            region.y : region.y + region.height,
            region.x : region.x + region.width,
        ] = True
    assert not (result.fire_mask & ~covered).any()
    np.testing.assert_array_equal(result.original_rgb, pixels)


def test_transparent_hot_pixels_are_not_fire(tmp_path: Path) -> None:
    """An invisible RGB region must contribute only to invalid/unknown counts."""

    pixels = np.zeros((24, 24, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    pixels[2:10, 2:10] = (255, 150, 10, 0)
    pixels[12:20, 12:20] = (255, 150, 10, 255)
    path = _save_rgb(tmp_path / "alpha.png", pixels)

    result = SatelliteAnalyzer().analyze(path)

    assert result.invalid_pixel_count == 64
    assert result.fire_pixel_count == 64
    assert not result.fire_mask[2:10, 2:10].any()
    assert result.fire_mask[12:20, 12:20].all()
    assert len(result.detected_fire_regions) == 1
    assert result.to_dict()["unknown"]["pixel_count"] == 24 * 24 - 64


def test_downsampling_cannot_reveal_transparent_hot_colors(tmp_path: Path) -> None:
    """Invalid warm source pixels cannot leak through area interpolation."""

    pixels = np.full((67, 101, 4), (255, 150, 10, 0), dtype=np.uint8)
    pixels[20:47, 35:65] = (10, 40, 100, 255)
    path = _save_rgb(tmp_path / "hidden_warm.png", pixels)

    result = SatelliteAnalyzer(
        analysis_settings=AnalysisSettings(max_dimension=32)
    ).analyze(path)

    assert result.fire_pixel_count == 0
    assert not result.fire_mask.any()
    assert result.invalid_pixel_count == 67 * 101 - 27 * 30


def test_analyzer_enforces_source_pixel_budget(tmp_path: Path) -> None:
    """Configured resource limits are applied before analysis preparation."""

    path = _save_rgb(tmp_path / "budget.png", np.zeros((11, 11, 3), dtype=np.uint8))
    analyzer = SatelliteAnalyzer(analysis_settings=AnalysisSettings(max_pixels=120))

    with pytest.raises(InvalidImageError):
        analyzer.analyze(path)


class _ContractDetector(BaseDetector):
    """Return fixed synthetic labels, with optional controlled contract defects."""

    def __init__(self, defect: str) -> None:
        self.defect = defect

    def detect(
        self, image_rgb: np.ndarray, *, valid_mask: np.ndarray | None = None
    ) -> FireDetection:
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[2:6, 3:7] = True
        labels = mask.astype(np.int32)
        region = FireRegion(3, 2, 4, 4, 16, 0.8)
        confidence = 0.8
        if self.defect == "mask_shape":
            mask = mask[:-1]
        elif self.defect == "mask_dtype":
            mask = mask.astype(np.uint8)
        elif self.defect == "label_range":
            labels[mask] = 2
        elif self.defect == "mask_label_mismatch":
            labels[2, 3] = 0
        elif self.defect == "negative_label":
            labels[0, 0] = -1
        elif self.defect == "invalid_pixels":
            assert valid_mask is not None
            mask[:] = ~valid_mask
            labels = mask.astype(np.int32)
            region = replace(region, pixel_count=int(mask.sum()))
        elif self.defect == "confidence_nan":
            confidence = float("nan")
        elif self.defect == "region_confidence_nan":
            region = replace(region, confidence=float("nan"))
        elif self.defect == "wrong_area":
            region = replace(region, pixel_count=15)
        elif self.defect == "out_of_bounds":
            region = replace(region, x=image_rgb.shape[1])
        elif self.defect == "wrong_bbox":
            region = replace(region, x=0, y=0)
        return FireDetection(
            mask=mask, labels=labels, confidence=confidence, regions=[region]
        )


@pytest.mark.parametrize(
    "defect",
    [
        "mask_shape",
        "mask_dtype",
        "label_range",
        "mask_label_mismatch",
        "negative_label",
        "invalid_pixels",
        "confidence_nan",
        "region_confidence_nan",
        "wrong_area",
        "out_of_bounds",
        "wrong_bbox",
    ],
)
def test_analyzer_rejects_invalid_detector_contracts(
    tmp_path: Path, defect: str
) -> None:
    """Malformed future-detector results receive a clear domain error."""

    pixels = np.full((12, 13, 4), (10, 40, 100, 255), dtype=np.uint8)
    pixels[8:10, 8:10, 3] = 0
    path = _save_rgb(tmp_path / "contract.png", pixels)
    analyzer = SatelliteAnalyzer(detector=_ContractDetector(defect))

    with pytest.raises(ImageProcessingError):
        analyzer.analyze(path)


def test_valid_replacement_detector_is_used(tmp_path: Path) -> None:
    """A conforming injected detector can replace the classical implementation."""

    path = _save_rgb(tmp_path / "custom.png", np.zeros((12, 13, 3), dtype=np.uint8))

    result = SatelliteAnalyzer(detector=_ContractDetector("valid")).analyze(path)

    assert result.fire_pixel_count == 16
    assert result.fire_confidence == pytest.approx(0.8)
    assert result.detected_fire_regions == [FireRegion(3, 2, 4, 4, 16, 0.8)]
    assert result.to_dict()["parameters"]["detector"] == "_ContractDetector"
