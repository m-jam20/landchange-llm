"""Offline tests for change.py: one fake pixel per situation, with a known answer."""

import numpy as np
import pytest
import xarray as xr

from change_monitor.change import CATEGORIES, NO_CHANGE, NOT_CLEAR, classify_change, measure_change

CODE = {name: code for code, name in CATEGORIES.items()}

# Each row: (situation, before (ndvi, ndbi, ndwi), after (ndvi, ndbi, ndwi), clear, expected label)
CASES = [
    ("nothing happened", (0.6, -0.3, -0.5), (0.62, -0.3, -0.5), True, NO_CHANGE),
    ("meadow -> building", (0.6, -0.3, -0.5), (0.1, 0.1, -0.2), True, CODE["new_built_up"]),
    ("forest -> brown, no building", (0.7, -0.3, -0.6), (0.3, -0.2, -0.4), True, CODE["vegetation_loss"]),
    ("bare field -> crops", (0.2, 0.0, -0.3), (0.7, -0.3, -0.6), True, CODE["vegetation_gain"]),
    ("land -> new pond", (0.3, -0.1, -0.3), (-0.2, -0.3, 0.3), True, CODE["water_gain"]),
    ("pond dried out", (-0.2, -0.3, 0.3), (0.3, -0.1, -0.3), True, CODE["water_loss"]),
    ("huge change but cloudy", (0.7, -0.3, -0.6), (0.0, 0.2, 0.0), False, NOT_CLEAR),
    ("small change below threshold", (0.6, -0.3, -0.5), (0.45, -0.15, -0.4), True, NO_CHANGE),
]


def make_inputs(cases):
    def ds(column):
        rows = np.array([case[column] for case in cases], dtype="float32")
        return xr.Dataset({name: ("pixel", rows[:, i]) for i, name in enumerate(("ndvi", "ndbi", "ndwi"))})

    clear = xr.DataArray(np.array([case[3] for case in cases]), dims="pixel")
    return ds(1), ds(2), clear


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_each_situation_gets_the_right_label(case):
    before, after, clear = make_inputs([case])
    assert int(classify_change(before, after, clear, threshold=0.2)[0]) == case[4]


def test_built_up_wins_over_vegetation_loss():
    """Meadow -> building matches both rules; it must be counted once, as built-up."""
    before, after, clear = make_inputs([CASES[1]])
    assert int(classify_change(before, after, clear)[0]) == CODE["new_built_up"]


def test_hectares_add_up():
    before, after, clear = make_inputs(CASES)
    change_map = classify_change(before, after, clear)
    stats = measure_change(change_map, pixel_area_ha=0.01)

    assert stats["total_ha"] == pytest.approx(len(CASES) * 0.01)
    assert stats["clear_ha"] == pytest.approx(7 * 0.01)  # one case is cloudy
    changed = sum(stats["change_ha"].values())
    assert stats["no_change_ha"] + changed == pytest.approx(stats["clear_ha"])
    assert stats["change_ha"]["new_built_up"] == pytest.approx(0.01)


def test_higher_threshold_finds_less_change():
    before, after, clear = make_inputs(CASES)
    low = measure_change(classify_change(before, after, clear, threshold=0.1))
    high = measure_change(classify_change(before, after, clear, threshold=0.5))
    assert sum(high["change_ha"].values()) <= sum(low["change_ha"].values())
