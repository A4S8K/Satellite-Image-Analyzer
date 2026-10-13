"""Validated visible RGB input for the Week 2 fire detector.

Week 1's loader retains native raster bands. This adapter instead requires a
known colour interpretation and 8-bit visual data; arbitrary spectral bands
must never be passed to HSV thresholds as though they were RGB.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.enums import ColorInterp
from rasterio.errors import RasterioError

from satellite_analyzer.core.exceptions import InvalidImageError, UnsupportedFormatError
from satellite_analyzer.io.image_loader import SUPPORTED_EXTENSIONS
from satellite_analyzer.io.metadata import ImageMetadata


@dataclass(frozen=True, slots=True)
class RGBImage:
    """Visible RGB data, valid-pixel mask, and original metadata."""

    rgb: np.ndarray
    valid_mask: np.ndarray
    metadata: ImageMetadata
    path: Path


class RGBImageLoader:
    """Decode 8-bit visual images while excluding transparency and raster nodata."""

    def __init__(self, max_pixels: int = 40_000_000) -> None:
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        self.max_pixels = max_pixels

    def load(self, path: str | Path) -> RGBImage:
        """Decode supported visual data, rejecting unsafe/ambiguous band inputs."""
        image_path = Path(path).expanduser()
        if not image_path.is_file():
            raise InvalidImageError(f"Image path is not an existing file: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                "Fire analysis supports PNG, JPG, JPEG, TIF, and TIFF files."
            )
        if image_path.suffix.lower() in {".tif", ".tiff"}:
            return self._load_tiff(image_path)
        return self._load_pillow(image_path)

    def _check_dimensions(self, width: int, height: int) -> None:
        if width < 1 or height < 1 or width * height > self.max_pixels:
            raise InvalidImageError(
                f"Image dimensions {width}x{height} exceed the valid pixel budget "
                f"({self.max_pixels:,} pixels)."
            )

    def _load_pillow(self, path: Path) -> RGBImage:
        try:
            with Image.open(path) as candidate:
                self._check_dimensions(*candidate.size)
                if candidate.format not in {"PNG", "JPEG"}:
                    raise InvalidImageError(
                        "Image content does not match a supported PNG/JPEG format."
                    )
                candidate.verify()
            with Image.open(path) as image:
                if image.mode not in {"RGB", "RGBA", "L", "LA", "P", "1"}:
                    raise InvalidImageError(
                        f"Fire analysis requires 8-bit visual RGB/grayscale; "
                        f"image mode '{image.mode}' requires explicit conversion."
                    )
                channels = len(image.getbands())
                rgb = np.asarray(image.convert("RGB"))
                # Palette transparency is a lookup, not a fourth data band.
                has_alpha = "A" in image.getbands() or "transparency" in image.info
                valid = (
                    np.asarray(image.convert("RGBA"))[..., 3] > 0
                    if has_alpha
                    else np.ones(rgb.shape[:2], dtype=bool)
                )
                image_format = image.format or "PNG"
        except (
            OSError,
            SyntaxError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as error:
            raise InvalidImageError(
                f"Could not decode image '{path}': {error}"
            ) from error
        metadata = ImageMetadata(
            filename=path.name,
            format=image_format,
            width=rgb.shape[1],
            height=rgb.shape[0],
            channels=channels,
            dtype="uint8",
            file_size=path.stat().st_size,
        )
        return RGBImage(rgb=rgb, valid_mask=valid, metadata=metadata, path=path)

    def _load_tiff(self, path: Path) -> RGBImage:
        try:
            with rasterio.open(path) as dataset:
                self._check_dimensions(dataset.width, dataset.height)
                interpretation = dataset.colorinterp
                rgb_names = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                if all(interpretation.count(color) == 1 for color in rgb_names):
                    if dataset.count not in {3, 4} or (
                        dataset.count == 4 and ColorInterp.alpha not in interpretation
                    ):
                        raise InvalidImageError(
                            "Multispectral TIFF needs explicit RGB band selection "
                            "before fire analysis."
                        )
                    indices = [interpretation.index(color) + 1 for color in rgb_names]
                elif dataset.count == 1 and interpretation[0] in {
                    ColorInterp.gray,
                    ColorInterp.undefined,
                }:
                    indices = [1]
                elif dataset.count == 2 and set(interpretation) == {
                    ColorInterp.gray,
                    ColorInterp.alpha,
                }:
                    indices = [interpretation.index(ColorInterp.gray) + 1]
                else:
                    raise InvalidImageError(
                        "TIFF bands must be identified as RGB, RGBA, or grayscale; "
                        "ambiguous spectral data requires explicit conversion."
                    )
                if any(dataset.dtypes[index - 1] != "uint8" for index in indices):
                    raise InvalidImageError(
                        "Fire analysis requires uint8 visual TIFF data; convert "
                        "higher-bit-depth or floating bands to display RGB explicitly."
                    )
                bands = dataset.read(indices)
                rgb = np.moveaxis(bands, 0, -1)
                if len(indices) == 1:
                    rgb = np.repeat(rgb, 3, axis=2)
                rgb = np.ascontiguousarray(rgb)
                valid = np.all(dataset.read_masks(indices) > 0, axis=0)
                if ColorInterp.alpha in interpretation:
                    valid &= (
                        dataset.read(interpretation.index(ColorInterp.alpha) + 1) > 0
                    )
                metadata = ImageMetadata(
                    filename=path.name,
                    format=dataset.driver.upper(),
                    width=dataset.width,
                    height=dataset.height,
                    channels=dataset.count,
                    dtype="uint8",
                    file_size=path.stat().st_size,
                    raster={
                        "driver": dataset.driver,
                        "band_count": dataset.count,
                        "crs": str(dataset.crs) if dataset.crs else None,
                        "transform": list(dataset.transform),
                        "bounds": list(dataset.bounds),
                        "rgb_band_indices": indices,
                        "color_interpretation": [
                            color.name for color in interpretation
                        ],
                        "nodata": (
                            str(dataset.nodata) if dataset.nodata is not None else None
                        ),
                    },
                )
        except (RasterioError, OSError, ValueError) as error:
            raise InvalidImageError(
                f"Could not decode TIFF '{path}': {error}"
            ) from error
        return RGBImage(rgb=rgb, valid_mask=valid, metadata=metadata, path=path)
