<<<<<<< HEAD
<<<<<<< HEAD
# Satellite Image Analyzer

## Overview

Satellite Image Analyzer is a modular Python foundation for loading, inspecting,
and preprocessing satellite imagery. Week 1 focuses on reliable raster I/O and
preparation; analytical features are intentionally deferred.

## Features

- Loads PNG, JPG, JPEG, TIFF, and GeoTIFF image files into NumPy arrays.
- Rejects missing, unsupported, and corrupted image files with clear errors.
- Extracts standard image metadata and available TIFF raster metadata.
- Resizes, normalizes, and gently denoises imagery through a configurable pipeline.
- Includes an installable Typer CLI, tests, linting, type checks, and CI.

## Architecture

The CLI calls the core configuration and exception layer, then uses I/O and
preprocessing modules with clear boundaries. See [architecture documentation](docs/architecture.md).

## Installation

Python 3.12 or newer is required. Create a virtual environment, then install the
project and development tools:

```bash
pip install -e ".[dev]"
```

## Usage

```bash
sat-analyzer info path/to/image.tif
sat-analyzer preprocess path/to/image.tif --width 512 --height 512
sat-analyzer preprocess image.jpg --output outputs/image_processed.png
```

The default preprocessing settings are in `configs/default.toml`.

## CLI

```bash
sat-analyzer --help
sat-analyzer version
sat-analyzer info IMAGE
sat-analyzer preprocess IMAGE [--width PIXELS] [--height PIXELS] [--output PATH]
```

`preprocess` writes a PNG beside the input by default, using the suffix
`_preprocessed.png`. Use `--no-normalize` or `--no-denoise` to override the
default processing switches for one command.

## Supported Formats

| Extension | Loader |
| --- | --- |
| `.png` | Pillow |
| `.jpg`, `.jpeg` | Pillow |
| `.tif`, `.tiff` | Rasterio |

## Development

```bash
pip install -e ".[dev]"
ruff check .
black --check .
mypy src
```

## Testing

```bash
pytest
pytest --cov
```

Tests generate small deterministic images locally and do not download external data.

## Project Roadmap

- **Week 1:** image I/O, metadata, preprocessing, CLI, and quality tooling.
- **Week 2 (planned):** fire detection foundations.
- **Week 3 (planned):** NDVI and vegetation analysis.
- **Week 4 (planned):** anomaly detection and risk-engine foundations.
- **Future:** ML, ONNX, FastAPI, and a web dashboard.

## License

Distributed under the [MIT License](LICENSE).
=======
# Satellite-Image-Analyzer
A modular Python system for analyzing satellite imagery using Computer Vision and geospatial technologies, with support for preprocessing, vegetation analysis, anomaly detection, and AI-powered environmental monitoring.
>>>>>>> 6e0b99a19fb49a216b48dab4e1672ac691690c31
=======
# Satellite-Image-Analyzer
A modular Python system for satellite image analysis using Computer Vision, geospatial technologies, and AI for environmental monitoring.
>>>>>>> 44aa14cb8af6ece8484df542a0324bc82e871d68
