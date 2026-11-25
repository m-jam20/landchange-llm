"""Small pure helpers for the Streamlit app.

They live here (not in app.py) because app.py runs the whole page when imported,
which makes it hard to test. These functions are plain Python, so pytest can test them.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

from change_monitor.change import CATEGORIES
from change_monitor.render import LEGEND
from change_monitor.search import BBox


def bbox_from_geojson(feature: dict) -> BBox:
    """A drawn rectangle (GeoJSON Feature from the map) -> (west, south, east, north)."""
    ring = feature["geometry"]["coordinates"][0]  # list of [lon, lat] corner points
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    return (round(min(lons), 5), round(min(lats), 5), round(max(lons), 5), round(max(lats), 5))


def date_range(dates) -> str:
    """Streamlit's date range picker value -> 'YYYY-MM-DD/YYYY-MM-DD' for STAC.

    While the user is still picking, Streamlit returns only one date: that's an error here.
    """
    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        raise ValueError("Pick both a start and an end date.")
    start, end = dates
    if start > end:
        raise ValueError("The start date must be before the end date.")
    return f"{start.isoformat()}/{end.isoformat()}"


def check_periods(before: tuple[date, date], after: tuple[date, date]) -> list[str]:
    """Warnings (not errors) about the two periods. Empty list = all fine."""
    warnings = []
    if before[1] >= after[0]:
        warnings.append("The two periods overlap or are in the wrong order: 'before' should end before 'after' starts.")
    if abs(before[0].month - after[0].month) > 1:
        warnings.append(
            "The periods start in different seasons. Plants look different in spring and autumn, "
            "so you may see 'vegetation change' that is only the season."
        )
    return warnings


def change_table(stats: dict) -> list[dict]:
    """Rows for the results table, largest change first, with readable names."""
    names = {name: LEGEND[code][1] for code, name in CATEGORIES.items()}
    rows = [
        {
            "Change": names[name],
            "Hectares": ha,
            "% of clear area": stats["change_pct_of_clear"][name],
        }
        for name, ha in stats["change_ha"].items()
    ]
    return sorted(rows, key=lambda row: row["Hectares"], reverse=True)


def zip_folder(folder: str | Path) -> bytes:
    """All files in a results folder as one .zip (in memory), for a single download button."""
    folder = Path(folder)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                zf.write(path, arcname=f"{folder.name}/{path.name}")
    return buffer.getvalue()
