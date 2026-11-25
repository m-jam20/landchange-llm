"""Offline tests for indices.py."""

import numpy as np
import pytest
import xarray as xr

from change_monitor.indices import compute_indices, normalized_difference


def arr(*values):
    return xr.DataArray(np.array(values, dtype="float32"))


def test_normalized_difference_formula():
    # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6
    assert float(normalized_difference(arr(0.5), arr(0.1))[0]) == pytest.approx(2 / 3)


def test_zero_plus_zero_gives_nan_not_crash():
    assert np.isnan(normalized_difference(arr(0.0), arr(0.0))[0])


def test_nan_input_stays_nan():
    assert np.isnan(normalized_difference(arr(np.nan), arr(0.1))[0])


def test_typical_surfaces():
    # One pixel each: vegetation, water, building (realistic reflectances)
    ds = xr.Dataset(
        {
            "B02": arr(0.03, 0.05, 0.12),
            "B03": arr(0.06, 0.08, 0.13),
            "B04": arr(0.04, 0.05, 0.14),
            "B08": arr(0.40, 0.02, 0.18),
            "B11": arr(0.20, 0.01, 0.25),
        }
    )
    idx = compute_indices(ds)
    veg, water, building = 0, 1, 2
    assert idx.ndvi[veg] > 0.6  # plants: high NDVI
    assert idx.ndwi[water] > 0  # water: NDWI above 0
    assert idx.ndbi[building] > 0  # building: NDBI above 0
    assert idx.ndvi[veg] > idx.ndvi[building]
