"""Offline tests for render.py: draw tiny fake images and check the PNG files appear."""

import numpy as np
import xarray as xr

from change_monitor.render import render_all, stretch_limits, true_color

BANDS = ("B02", "B03", "B04", "B08", "B11")


def fake_scene(brightness: float) -> xr.Dataset:
    data = np.full((4, 5), brightness, dtype="float32")
    data[0, 0] = np.nan  # one pixel with no data
    return xr.Dataset({b: (("y", "x"), data.copy()) for b in BANDS})


def test_true_color_is_between_0_and_1_and_has_3_channels():
    before = fake_scene(0.1)
    rgb = true_color(before, (0.0, 0.2))
    assert rgb.shape == (4, 5, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1
    assert np.all(rgb[0, 0] == 1.0)  # no data is shown as white


def test_shared_limits_keep_the_brighter_date_brighter():
    dark, bright = fake_scene(0.05), fake_scene(0.15)
    limits = stretch_limits(dark, bright)
    assert true_color(bright, limits)[1, 1, 0] > true_color(dark, limits)[1, 1, 0]


def test_render_all_writes_three_pngs(tmp_path):
    change = xr.DataArray(np.array([[-1, 0, 1, 2, 3], [4, 5, 0, 0, 0], [0] * 5, [0] * 5], dtype="int8"))
    paths = render_all(fake_scene(0.05), fake_scene(0.1), change, tmp_path)
    assert set(paths) == {"before", "after", "change_map"}
    for path in paths.values():
        assert path.exists() and path.read_bytes()[:4] == b"\x89PNG"
