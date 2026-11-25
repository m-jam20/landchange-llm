"""Offline tests for the input checks in search.py (no internet needed)."""

from types import SimpleNamespace

import pytest

from change_monitor.search import bbox_area_km2, find_least_cloudy_scene, validate_bbox


def test_valid_bbox_is_returned_as_floats():
    assert validate_bbox((13, 52, 13.1, 52.1)) == (13.0, 52.0, 13.1, 52.1)


def test_swapped_min_and_max_is_rejected():
    with pytest.raises(ValueError, match="west, south, east, north"):
        validate_bbox((13.58, 52.41, 13.50, 52.45))


def test_lat_lon_swapped_is_rejected():
    with pytest.raises(ValueError, match="swap"):
        validate_bbox((52.41, 13.50, 52.45, 113.58))


def test_too_big_area_is_rejected():
    with pytest.raises(ValueError, match="limit"):
        validate_bbox((10.0, 50.0, 12.0, 52.0))  # roughly 140 x 220 km


def test_area_of_one_degree_at_equator():
    # 1° x 1° at the equator is about 111 x 111 km
    assert bbox_area_km2((0, -0.5, 1, 0.5)) == pytest.approx(111.32 * 110.57, rel=1e-3)


# --- error messages, with a fake catalog (no internet) ---------------------------

BERLIN = (13.50, 52.41, 13.58, 52.45)


def fake_catalog(cloud_covers):
    """Looks like a pystac-client Client: .search(...).items() returns fake scenes."""
    items = [SimpleNamespace(properties={"eo:cloud_cover": c}) for c in cloud_covers]
    return SimpleNamespace(search=lambda **kwargs: SimpleNamespace(items=lambda: iter(items)))


def test_no_images_at_all_says_when_the_archive_starts():
    with pytest.raises(LookupError, match="No Sentinel-2 images exist.*2015-06-27"):
        find_least_cloudy_scene(BERLIN, "2015-06-01/2015-06-20", catalog=fake_catalog([]))


def test_only_cloudy_images_says_how_cloudy():
    with pytest.raises(LookupError, match="Found 3 images.*clearest has 45% cloud"):
        find_least_cloudy_scene(BERLIN, "2024-01-01/2024-01-31", catalog=fake_catalog([80, 45, 95]))
