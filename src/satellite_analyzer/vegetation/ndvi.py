"""Vectorized NDVI for measured red and near-infrared reflectance arrays."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from satellite_analyzer.core.exceptions import ImageProcessingError


@dataclass(frozen=True, slots=True)
class NDVIData:
    """Float32 NDVI values; invalid samples are NaN and excluded by the mask."""

    values: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]


def compute_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    valid_mask: NDArray[np.bool_] | None = None,
    *,
    epsilon: float = 1e-6,
) -> NDVIData:
    """Calculate (NIR-RED)/(NIR+RED) from independently measured bands.

    Inputs must already have matching spatial grids and known reflectance units.
    Negative reflectance, nonfinite samples, and zero/near-zero denominators
    are excluded rather than converted into apparently meaningful vegetation.
    """
    if (
        not isinstance(red, np.ndarray)
        or not isinstance(nir, np.ndarray)
        or red.ndim != 2
        or nir.ndim != 2
        or red.shape != nir.shape
        or not red.size
        or not np.issubdtype(red.dtype, np.number)
        or not np.issubdtype(nir.dtype, np.number)
        or np.iscomplexobj(red)
        or np.iscomplexobj(nir)
    ):
        raise ImageProcessingError(
            "NDVI requires two nonempty real 2D bands of matching dimensions"
        )
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ImageProcessingError(
            "NDVI denominator epsilon must be finite and nonnegative"
        )
    if valid_mask is not None and (
        valid_mask.dtype != np.bool_ or valid_mask.shape != red.shape
    ):
        raise ImageProcessingError(
            "NDVI validity mask must be boolean and match the band grid"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        red_float = np.asarray(red, dtype=np.float32)
        nir_float = np.asarray(nir, dtype=np.float32)
        denominator = red_float + nir_float
        valid = (
            np.isfinite(red_float)
            & np.isfinite(nir_float)
            & np.isfinite(denominator)
            & (red_float >= 0)
            & (nir_float >= 0)
            & (denominator > epsilon)
        )
        if valid_mask is not None:
            valid &= valid_mask
        if np.ma.isMaskedArray(red):
            valid &= ~np.ma.getmaskarray(red)
        if np.ma.isMaskedArray(nir):
            valid &= ~np.ma.getmaskarray(nir)
        values = np.full(red.shape, np.nan, dtype=np.float32)
        np.divide(nir_float - red_float, denominator, out=values, where=valid)
    # Nonnegative reflectances imply [-1,1]; clipping only handles roundoff.
    np.clip(values, -1.0, 1.0, out=values)
    values.setflags(write=False)
    valid.setflags(write=False)
    return NDVIData(values=values, valid_mask=valid)
