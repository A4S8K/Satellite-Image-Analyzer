"""Composable preprocessing pipeline for image data."""

from dataclasses import dataclass

from satellite_analyzer.core.config import PreprocessingSettings
from satellite_analyzer.io.image_loader import ImageData
from satellite_analyzer.preprocessing.filters import reduce_noise
from satellite_analyzer.preprocessing.normalization import normalize_image
from satellite_analyzer.preprocessing.resize import resize_image


@dataclass(frozen=True, slots=True)
class PreprocessingPipeline:
    """Apply configured resizing, normalization, and noise reduction in sequence."""

    settings: PreprocessingSettings

    def process(self, image: ImageData) -> ImageData:
        """Process an image without mutating its original pixel array."""

        processed = resize_image(
            image.array,
            width=self.settings.target_width,
            height=self.settings.target_height,
        )
        if self.settings.normalize:
            processed = normalize_image(processed)
        if self.settings.denoise:
            processed = reduce_noise(processed)
        return image.with_array(processed)
