"""Replaceable detectors and their typed outputs."""

from satellite_analyzer.detectors.base import BaseDetector, FireDetection, FireRegion
from satellite_analyzer.detectors.fire import RGBFireDetector

__all__ = ["BaseDetector", "FireDetection", "FireRegion", "RGBFireDetector"]
