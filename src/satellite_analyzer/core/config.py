"""Configuration loading for the application."""

from pathlib import Path
import tomllib

from pydantic import BaseModel, Field


class PreprocessingSettings(BaseModel):
    """Settings controlling the default preprocessing pipeline."""

    target_width: int = Field(default=1024, gt=0)
    target_height: int = Field(default=1024, gt=0)
    normalize: bool = True
    denoise: bool = True


def default_config_path() -> Path:
    """Return the repository's bundled default configuration path."""

    return Path(__file__).resolve().parents[3] / "configs" / "default.toml"


def load_preprocessing_settings(path: Path | None = None) -> PreprocessingSettings:
    """Load preprocessing settings from a TOML configuration file."""

    config_path = path or default_config_path()
    with config_path.open("rb") as config_file:
        values = tomllib.load(config_file)
    return PreprocessingSettings.model_validate(values.get("preprocessing", {}))
