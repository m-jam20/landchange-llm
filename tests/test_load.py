"""Offline tests for load.py, using tiny fake datasets (no internet needed)."""

import numpy as np
import pytest
import xarray as xr

from change_monitor.load import (
    REFLECTANCE_BANDS,
    clear_mask,
    reflectance_offset,
    to_reflectance,
    utm_crs_for_bbox,
)


def fake_raw(value: int) -> xr.Dataset:
    """A 1 x 2 'scene': one pixel with `value` in every band, one pixel with no data (0)."""
    data = np.array([[value, 0]], dtype="uint16")
    return xr.Dataset({band: (("y", "x"), data) for band in REFLECTANCE_BANDS})


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((13.50, 52.41, 13.58, 52.45), "EPSG:32633"),  # Berlin
        ((-0.2, 51.4, 0.0, 51.6), "EPSG:32630"),  # London
        ((151.1, -33.9, 151.3, -33.8), "EPSG:32756"),  # Sydney (southern hemisphere)
    ],
)
def test_utm_zone(bbox, expected):
    assert utm_crs_for_bbox(bbox) == expected


@pytest.mark.parametrize(
    ("baseline", "expected"),
    [("02.14", 0), ("03.01", 0), ("04.00", -1000), ("05.11", -1000), (None, 0)],
)
def test_offset_depends_on_processing_baseline(baseline, expected):
    assert reflectance_offset(baseline) == expected


def test_new_scene_subtracts_offset():
    refl = to_reflectance(fake_raw(3000), "05.11")
    assert float(refl["B04"][0, 0]) == pytest.approx(0.2)  # (3000 - 1000) / 10000


def test_old_scene_has_no_offset():
    refl = to_reflectance(fake_raw(3000), "03.01")
    assert float(refl["B04"][0, 0]) == pytest.approx(0.3)


def test_zero_is_no_data():
    refl = to_reflectance(fake_raw(3000), "05.11")
    assert np.isnan(refl["B04"][0, 1])


def test_dark_pixel_is_clipped_to_zero():
    refl = to_reflectance(fake_raw(900), "05.11")  # 900 - 1000 would be negative
    assert float(refl["B04"][0, 0]) == 0.0


def test_clear_mask_keeps_only_trusted_classes():
    scl = xr.DataArray(np.arange(12))  # every SCL class 0..11
    expected = [False, False, False, False, True, True, True, True, False, False, False, False]
    assert clear_mask(scl).values.tolist() == expected
