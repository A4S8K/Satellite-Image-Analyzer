"""Offline numerical tests for actual, independently supplied Red/NIR arrays."""

import numpy as np
import pytest

from satellite_analyzer.core.exceptions import ImageProcessingError
from satellite_analyzer.vegetation.ndvi import compute_ndvi


def test_correct_signed_ndvi_and_zero() -> None:
    red = np.array([[0.1, 0.3, 0.3, 0, 1]], dtype=np.float32)
    nir = np.array([[0.3, 0.1, 0.3, 1, 0]], dtype=np.float32)
    before = red.copy()
    result = compute_ndvi(red, nir)
    np.testing.assert_allclose(result.values, [[0.5, -0.5, 0, 1, -1]], atol=1e-6)
    assert result.values.dtype == np.float32
    assert result.valid_mask.all()
    np.testing.assert_array_equal(red, before)


def test_invalid_values_are_nan_and_not_finite_statistics() -> None:
    red = np.array([[0, np.nan, np.inf, -0.1, 0.2, 1e-8]], dtype=np.float32)
    nir = np.array([[0, 0.2, 0.3, 0.4, 0.6, 1e-8]], dtype=np.float32)
    mask = np.array([[True, True, True, True, False, True]])
    result = compute_ndvi(red, nir, mask)
    assert not result.valid_mask.any()
    assert np.isnan(result.values).all()


def test_masked_arrays_preserve_nodata() -> None:
    red = np.ma.array([[0.1, 0.1]], mask=[[False, True]], dtype=np.float32)
    result = compute_ndvi(red, np.full((1, 2), 0.4, dtype=np.float32))
    assert result.values[0, 0] == pytest.approx(0.6)
    assert np.isnan(result.values[0, 1])


@pytest.mark.parametrize(
    "red,nir",
    [
        (np.zeros((2, 2)), np.zeros((3, 2))),
        (np.zeros((0, 2)), np.zeros((0, 2))),
        (np.zeros((1, 2, 3)), np.zeros((1, 2, 3))),
        (np.array([["x"]]), np.array([["y"]])),
        (np.ones((1, 1), dtype=np.complex64), np.ones((1, 1))),
    ],
)
def test_invalid_bands_fail_clearly(red: np.ndarray, nir: np.ndarray) -> None:
    with pytest.raises(ImageProcessingError):
        compute_ndvi(red, nir)


@pytest.mark.parametrize("epsilon", [-1.0, np.nan, np.inf])
def test_epsilon_must_be_valid(epsilon: float) -> None:
    with pytest.raises(ImageProcessingError):
        compute_ndvi(np.ones((1, 1)), np.ones((1, 1)), epsilon=epsilon)


def test_overflow_and_mask_dimensions_are_validated() -> None:
    result = compute_ndvi(np.full((1, 1), 1e38), np.full((1, 1), 1e38))
    assert result.values[0, 0] == 0
    overflow = compute_ndvi(np.full((1, 1), 1e300), np.full((1, 1), 1e300))
    assert not overflow.valid_mask.any()
    with pytest.raises(ImageProcessingError):
        compute_ndvi(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 2), dtype=bool))
