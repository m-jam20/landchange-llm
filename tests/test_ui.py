"""Offline tests for the UI: the helper functions, and that app.py loads without errors."""

import io
import zipfile
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from change_monitor.ui_helpers import bbox_from_geojson, change_table, check_periods, date_range, zip_folder

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_bbox_from_drawn_rectangle():
    feature = {"geometry": {"type": "Polygon", "coordinates": [[
        [13.5, 52.41], [13.5, 52.45], [13.58, 52.45], [13.58, 52.41], [13.5, 52.41],
    ]]}}
    assert bbox_from_geojson(feature) == (13.5, 52.41, 13.58, 52.45)


def test_date_range_formats_for_stac():
    assert date_range((date(2023, 6, 1), date(2023, 8, 31))) == "2023-06-01/2023-08-31"


@pytest.mark.parametrize("dates", [(date(2023, 6, 1),), (date(2023, 9, 1), date(2023, 6, 1)), None])
def test_incomplete_or_reversed_dates_are_errors(dates):
    with pytest.raises(ValueError):
        date_range(dates)


def test_same_season_has_no_warnings():
    assert check_periods((date(2023, 6, 1), date(2023, 8, 31)), (date(2025, 6, 1), date(2025, 8, 31))) == []


def test_different_seasons_and_overlap_are_warned():
    warnings = check_periods((date(2025, 6, 1), date(2025, 8, 31)), (date(2025, 1, 1), date(2025, 2, 1)))
    assert len(warnings) == 2


def test_change_table_is_sorted_largest_first():
    stats = {
        "change_ha": {"water_gain": 1.0, "new_built_up": 30.0, "vegetation_loss": 19.0,
                      "water_loss": 2.0, "vegetation_gain": 23.0},
        "change_pct_of_clear": {"water_gain": 0.1, "new_built_up": 1.2, "vegetation_loss": 0.8,
                                "water_loss": 0.1, "vegetation_gain": 0.9},
    }
    rows = change_table(stats)
    assert rows[0]["Change"] == "New built-up / bare ground"
    assert [r["Hectares"] for r in rows] == sorted((r["Hectares"] for r in rows), reverse=True)


def test_zip_contains_every_file(tmp_path):
    run = tmp_path / "berlin_2023_2025"
    run.mkdir()
    (run / "stats.json").write_text("{}")
    (run / "change_map.png").write_bytes(b"png")
    names = zipfile.ZipFile(io.BytesIO(zip_folder(run))).namelist()
    assert sorted(names) == ["berlin_2023_2025/change_map.png", "berlin_2023_2025/stats.json"]


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()  # loading only: no download happens
    assert not at.exception
    assert [b.label for b in at.button] == ["Run analysis"]
