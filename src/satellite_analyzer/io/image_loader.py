"""Image loading with validation for supported raster formats."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.errors import RasterioError

from satellite_analyzer.core.exceptions import InvalidImageError, UnsupportedFormatError

SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


@dataclass(frozen=True, slots=True)
class ImageData:
    """A decoded image and its essential properties.

    The array is retained as loaded and is not copied by this model.
    """

    array: np.ndarray
    path: Path
    format: str

    @property
    def width(self) -> int:
        """Image width in pixels."""

        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        """Image height in pixels."""

        return int(self.array.shape[0])

    @property
    def channels(self) -> int:
        """Number of channels, treating a 2D image as single-channel."""

        return 1 if self.array.ndim == 2 else int(self.array.shape[2])

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """NumPy data type of the image pixels."""

        return self.array.dtype

    def with_array(self, array: np.ndarray) -> "ImageData":
        """Create an equivalent image object containing a processed array."""

        return ImageData(array=array, path=self.path, format=self.format)


class ImageLoader:
    """Load PNG, JPEG, TIFF, and GeoTIFF files into NumPy arrays."""

    def load(self, path: str | Path) -> ImageData:
        """Validate and decode an image at *path*.

        Raises:
            InvalidImageError: If the path is missing, not a file, or undecodable.
            UnsupportedFormatError: If the file extension is unsupported.
        """

        image_path = Path(path).expanduser()
        self._validate_path(image_path)
        extension = image_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedFormatError(
                f"Unsupported image format '{extension or '[no extension]'}'. "
                f"Supported formats: {supported}."
            )

        if extension in {".tif", ".tiff"}:
            return self._load_tiff(image_path)
        return self._load_pillow(image_path)

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.exists():
            raise InvalidImageError(f"Image file does not exist: {path}")
        if not path.is_file():
            raise InvalidImageError(f"Image path is not a file: {path}")

    @staticmethod
    def _load_pillow(path: Path) -> ImageData:
        try:
            with Image.open(path) as verification_image:
                verification_image.verify()
            with Image.open(path) as image:
                image.load()
                array = np.asarray(image)
                image_format = image.format or path.suffix.removeprefix(".").upper()
        except (OSError, SyntaxError, UnidentifiedImageError) as error:
            raise InvalidImageError(
                f"Could not decode image '{path}': {error}"
            ) from error

        return ImageData(array=array, path=path, format=image_format.upper())

    @staticmethod
    def _load_tiff(path: Path) -> ImageData:
        try:
            with rasterio.open(path) as dataset:
                bands = dataset.read()
                image_format = dataset.driver.upper()
        except (RasterioError, OSError, ValueError) as error:
            raise InvalidImageError(
                f"Could not decode TIFF image '{path}': {error}"
            ) from error

        array = bands[0] if bands.shape[0] == 1 else np.moveaxis(bands, 0, -1)
        return ImageData(array=array, path=path, format=image_format)
