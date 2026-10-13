"""Read genuine Red/NIR GeoTIFF bands with explicit radiometry and alignment.

Band indices are never inferred from an RGB array. The caller supplies separate
single-band files or explicitly maps a multispectral TIFF's spectral bands.
"""

import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.errors import RasterioError
from rasterio.warp import reproject

from satellite_analyzer.core.exceptions import InvalidImageError, SatelliteAnalyzerError
from satellite_analyzer.core.ndvi_config import NDVISettings
from satellite_analyzer.io.radiometry import resolve_calibration

BandRole = Literal["red", "nir"]
# Alignment error is measured in reference-grid pixels, not CRS map units.
_ALIGNMENT_PIXEL_TOLERANCE = 1e-6
_BAND_NAME = re.compile(r"(?<![A-Z0-9])B0?(\d{1,2})(A?)(?![A-Z0-9])", re.I)


@dataclass(frozen=True, slots=True)
class SpectralPair:
    """Calibrated bands on the Red reference grid, with JSON-safe provenance."""

    red: NDArray[np.float32]
    nir: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    crs: CRS
    transform: Affine
    sensor: str
    red_metadata: dict[str, Any]
    nir_metadata: dict[str, Any]
    alignment: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _LoadedBand:
    """One validated native grid before optional NIR reprojection."""

    reflectance: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    original_valid_pixels: int
    crs: CRS
    transform: Affine
    metadata: dict[str, Any]


class SpectralBandLoader:
    """Validate, calibrate and optionally align real spectral GeoTIFF inputs."""

    def __init__(self, settings: NDVISettings | None = None) -> None:
        self.settings = settings if settings is not None else NDVISettings()

    def load(
        self,
        *,
        red_path: Path | None = None,
        nir_path: Path | None = None,
        multiband_path: Path | None = None,
        red_band: int | None = None,
        nir_band: int | None = None,
    ) -> SpectralPair:
        """Decode separate single bands or explicitly mapped multispectral data.

        Grid, spectral identity and pixel budgets are checked for both inputs
        before either band is decoded. Source files are always opened read-only.
        """

        self._validate_input_mode(
            red_path, nir_path, multiband_path, red_band, nir_band
        )
        try:
            with ExitStack() as stack:
                if multiband_path is not None:
                    dataset = stack.enter_context(
                        rasterio.open(self._validate_path(multiband_path))
                    )
                    assert red_band is not None and nir_band is not None
                    red_source, nir_source = dataset, dataset
                    red_index, nir_index = red_band, nir_band
                else:
                    assert red_path is not None and nir_path is not None
                    red_source = stack.enter_context(
                        rasterio.open(self._validate_path(red_path))
                    )
                    nir_source = stack.enter_context(
                        rasterio.open(self._validate_path(nir_path))
                    )
                    if red_source.count != 1 or nir_source.count != 1:
                        raise InvalidImageError(
                            "Separate Red/NIR files must each contain one band; "
                            "use --multiband with explicit mapping "
                            "for multispectral TIFFs"
                        )
                    if Path(red_source.name).samefile(Path(nir_source.name)):
                        raise InvalidImageError(
                            "Separate Red/NIR inputs must be distinct physical bands; "
                            "the same file cannot supply both spectral channels"
                        )
                    red_index = nir_index = 1

                self._validate_dataset(red_source, red_index, "red")
                self._validate_dataset(nir_source, nir_index, "nir")
                same_grid = self._same_grid(red_source, nir_source)
                if not same_grid and self.settings.alignment == "strict":
                    raise InvalidImageError(
                        "Red/NIR CRS, dimensions or pixel alignment do not match; "
                        "explicitly choose reproject alignment to resample NIR"
                    )
                red = self._read_band(red_source, red_index, "red")
                nir = self._read_band(nir_source, nir_index, "nir")
        except (OSError, RasterioError, ValueError, OverflowError) as exc:
            raise InvalidImageError(f"Cannot load spectral imagery: {exc}") from exc

        alignment = {
            "policy": self.settings.alignment,
            "applied": not same_grid,
            "reference_grid": "red",
            "resampling": self.settings.resampling if not same_grid else "none",
            "mask_resampling": "nearest" if not same_grid else "none",
            "alignment_tolerance_pixels": _ALIGNMENT_PIXEL_TOLERANCE,
            "source_nir_crs": nir.crs.to_string(),
            "output_crs": red.crs.to_string(),
        }
        if same_grid:
            nir_values, nir_valid = nir.reflectance, nir.valid_mask
        else:
            nir_values, nir_valid = self._align_nir(nir, red)
        valid = red.valid_mask & nir_valid
        if (
            not same_grid
            and red.original_valid_pixels > 0
            and nir.original_valid_pixels > 0
            and not valid.any()
        ):
            raise InvalidImageError(
                "NIR reprojection produced no overlapping valid Red/NIR pixels"
            )
        # Paired arrays retain each band's native validity independently. NDVI
        # uses the intersection mask; no nodata is replaced with a plausible zero.
        for array in (red.reflectance, nir_values, valid):
            array.setflags(write=False)
        return SpectralPair(
            red=red.reflectance,
            nir=nir_values,
            valid_mask=valid,
            crs=red.crs,
            transform=red.transform,
            sensor=red.metadata["calibration"]["details"].get(
                "sensor", self.settings.sensor
            ),
            red_metadata=red.metadata,
            nir_metadata=nir.metadata,
            alignment=alignment,
        )

    @staticmethod
    def _validate_input_mode(
        red_path: Path | None,
        nir_path: Path | None,
        multiband_path: Path | None,
        red_band: int | None,
        nir_band: int | None,
    ) -> None:
        if multiband_path is not None:
            if red_path is not None or nir_path is not None:
                raise InvalidImageError(
                    "Choose separate Red/NIR paths or a multiband path, not both"
                )
            if (
                type(red_band) is not int
                or type(nir_band) is not int
                or red_band < 1
                or nir_band < 1
                or red_band == nir_band
            ):
                raise InvalidImageError(
                    "Multiband input requires distinct positive Red/NIR band indices"
                )
        elif (
            red_path is None
            or nir_path is None
            or red_band is not None
            or nir_band is not None
        ):
            raise InvalidImageError(
                "Supply both separate Red/NIR paths, "
                "or explicitly mapped multiband input"
            )

    @staticmethod
    def _validate_path(path: Path) -> Path:
        result = Path(path).expanduser()
        if not result.is_file():
            raise InvalidImageError(f"Spectral input is not an existing file: {result}")
        if result.suffix.lower() not in {".tif", ".tiff"}:
            raise InvalidImageError(
                "Spectral inputs must be GeoTIFF .tif or .tiff files"
            )
        return result

    def _validate_dataset(
        self, dataset: rasterio.DatasetReader, band: int, role: BandRole
    ) -> None:
        if dataset.driver != "GTiff":
            raise InvalidImageError("Spectral input contents must be a GeoTIFF")
        if not 1 <= band <= dataset.count:
            raise InvalidImageError(f"{role} band {band} exceeds TIFF band count")
        if (
            dataset.width < 1
            or dataset.height < 1
            or dataset.width * dataset.height > self.settings.max_pixels
        ):
            raise InvalidImageError(
                f"Spectral dimensions {dataset.width}x{dataset.height} exceed "
                f"the pixel budget ({self.settings.max_pixels:,})"
            )
        if dataset.crs is None:
            raise InvalidImageError("Spectral GeoTIFF is missing a CRS")
        transform = dataset.transform
        coefficients = tuple(transform)[:6]
        determinant = transform.a * transform.e - transform.b * transform.d
        if (
            not all(math.isfinite(value) for value in coefficients)
            or not math.isfinite(determinant)
            or determinant == 0
            or transform == Affine.identity()
        ):
            raise InvalidImageError(
                "Spectral GeoTIFF requires a finite, nonsingular spatial transform "
                "different from identity"
            )
        inverse = ~transform
        if not all(math.isfinite(value) for value in tuple(inverse)[:6]):
            raise InvalidImageError("Spectral transform cannot be inverted reliably")
        dtype = np.dtype(dataset.dtypes[band - 1])
        if not np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.complexfloating
        ):
            raise InvalidImageError(
                "Spectral bands must have real numeric pixel values"
            )
        self._validate_band_identity(dataset, band, role)

    def _validate_band_identity(
        self,
        dataset: rasterio.DatasetReader,
        band: int,
        role: BandRole,
        *,
        sensor: str | None = None,
    ) -> None:
        interpretation = dataset.colorinterp[band - 1]
        visible_nir = role == "nir" and interpretation in {
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        }
        wrong_red = role == "red" and interpretation in {
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        }
        if visible_nir or wrong_red:
            raise InvalidImageError(
                f"Selected {role} band has visible "
                f"{interpretation.name} interpretation; "
                "RGB channels are not substitutes for real Red/NIR spectral bands"
            )
        name_sources = [dataset.descriptions[band - 1] or ""]
        band_tags = dataset.tags(band)
        for key, value in band_tags.items():
            if key.upper() in {"BAND_NAME", "BANDNAME", "DESCRIPTION", "LONG_NAME"}:
                name_sources.append(value)
        # A multiband file's filename identifies its container, not each band.
        if dataset.count == 1:
            name_sources.append(Path(dataset.name).stem)
        identified = {
            (int(match.group(1)), match.group(2).upper())
            for source in name_sources
            for match in _BAND_NAME.finditer(source)
        }
        resolved_sensor = sensor if sensor is not None else self.settings.sensor
        if resolved_sensor == "sentinel-2":
            allowed = {(4, "")} if role == "red" else {(8, "")}
        elif resolved_sensor in {"landsat-8", "landsat-9"}:
            allowed = {(4, "")} if role == "red" else {(5, "")}
        else:
            allowed = {(4, "")} if role == "red" else {(5, ""), (8, "")}
        if identified and not identified.issubset(allowed):
            raise InvalidImageError(
                f"Known band name for {role} does not match {resolved_sensor} "
                "Red/NIR channels"
            )

    @staticmethod
    def _same_grid(red: rasterio.DatasetReader, nir: rasterio.DatasetReader) -> bool:
        if red.crs != nir.crs or red.shape != nir.shape:
            return False
        relative = ~red.transform * nir.transform
        # Corner comparisons include accumulated scale/rotation error across the
        # raster, unlike an absolute tolerance expressed in metres/degrees.
        for x, y in ((0, 0), (red.width, 0), (0, red.height), red.shape[::-1]):
            projected_x, projected_y = relative * (x, y)
            if (
                abs(projected_x - x) > _ALIGNMENT_PIXEL_TOLERANCE
                or abs(projected_y - y) > _ALIGNMENT_PIXEL_TOLERANCE
            ):
                return False
        return True

    def _read_band(
        self, dataset: rasterio.DatasetReader, band: int, role: BandRole
    ) -> _LoadedBand:
        raw = dataset.read(band)
        valid = (dataset.read_masks(band) > 0) & np.isfinite(raw)
        original_valid_pixels = int(np.count_nonzero(valid))
        calibration = resolve_calibration(
            settings=self.settings,
            role=role,
            dtype=raw.dtype,
            valid_values=raw.reshape(-1) if valid.all() else raw[valid],
            tiff_scale=dataset.scales[band - 1],
            tiff_offset=dataset.offsets[band - 1],
        )
        # Product metadata can identify a sensor when the user left "custom".
        # Recheck named bands against that sensor before using its coefficients;
        # Sentinel B05 must not accidentally receive Sentinel B08 calibration.
        self._validate_band_identity(
            dataset, band, role, sensor=calibration.details.get("sensor")
        )
        if calibration.zero_is_nodata:
            valid &= raw != 0
        reflectance = raw.astype(np.float32)
        # GDAL read() returns stored pixels, not pre-scaled reflectance. Scaling
        # is therefore applied exactly once, after source nodata is excluded.
        with np.errstate(over="ignore", invalid="ignore"):
            reflectance *= calibration.scale
            reflectance += calibration.offset
        valid &= np.isfinite(reflectance) & (reflectance >= 0)
        reflectance[~valid] = np.nan
        nodata = dataset.nodatavals[band - 1]
        metadata: dict[str, Any] = {
            "path": str(Path(dataset.name).resolve()),
            "band_index": band,
            "role": role,
            "description": dataset.descriptions[band - 1],
            "dtype": dataset.dtypes[band - 1],
            "width": dataset.width,
            "height": dataset.height,
            "crs": dataset.crs.to_string(),
            "transform": list(tuple(dataset.transform)[:6]),
            "nodata": None if nodata is None else str(nodata),
            "color_interpretation": dataset.colorinterp[band - 1].name,
            "tags": {**dataset.tags(), **dataset.tags(band)},
            "dataset_tags": dict(dataset.tags()),
            "valid_pixels_before_calibration": original_valid_pixels,
            "valid_pixels_after_calibration": int(np.count_nonzero(valid)),
            "calibration": {
                "scale": calibration.scale,
                "offset": calibration.offset,
                "source": calibration.source,
                "reflectance_type": calibration.reflectance_type,
                "zero_is_nodata": calibration.zero_is_nodata,
                "details": calibration.details,
            },
        }
        return _LoadedBand(
            reflectance=reflectance,
            valid_mask=valid,
            original_valid_pixels=original_valid_pixels,
            crs=dataset.crs,
            transform=dataset.transform,
            metadata=metadata,
        )

    def _align_nir(
        self, nir: _LoadedBand, red: _LoadedBand
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        destination = np.full(red.reflectance.shape, np.nan, dtype=np.float32)
        destination_mask = np.zeros(red.reflectance.shape, dtype=np.uint8)
        arguments = {
            "src_transform": nir.transform,
            "src_crs": nir.crs,
            "dst_transform": red.transform,
            "dst_crs": red.crs,
        }
        try:
            reproject(
                source=nir.reflectance,
                destination=destination,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling[self.settings.resampling],
                **arguments,
            )
            # Reflectance interpolation ignores NaN contributors. A separate
            # nearest mask additionally prevents filling missing source support.
            reproject(
                source=nir.valid_mask.astype(np.uint8),
                destination=destination_mask,
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.nearest,
                **arguments,
            )
        except (RasterioError, ValueError, OverflowError) as exc:
            raise SatelliteAnalyzerError(
                f"Cannot reproject NIR onto Red grid: {exc}"
            ) from exc
        valid = (destination_mask > 0) & np.isfinite(destination) & (destination >= 0)
        destination[~valid] = np.nan
        return destination, valid
