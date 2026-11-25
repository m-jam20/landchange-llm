"""Steps 1-2: check the input area and find the best Sentinel-2 scene per date range.

Every public function takes plain values (tuples, strings) and returns either a
pystac Item or a plain dict, so each one can later be handed to an agent as a tool.
"""

from __future__ import annotations

import math

import planetary_computer
import pystac
import pystac_client
from shapely.geometry import box, shape

from change_monitor import config

BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


def validate_bbox(bbox: BBox, max_area_km2: float = config.MAX_AREA_KM2) -> BBox:
    """Check a bounding box and return it as a tuple of floats.

    Raises ValueError with a readable message if the box is invalid or too big.
    """
    if len(bbox) != 4:
        raise ValueError("bbox needs 4 numbers: (min_lon, min_lat, max_lon, max_lat)")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)

    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180.")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90. Did you swap lon and lat?")
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox must be (west, south, east, north): min values first.")

    area = bbox_area_km2((min_lon, min_lat, max_lon, max_lat))
    if area > max_area_km2:
        raise ValueError(
            f"Area is {area:.0f} km², the limit is {max_area_km2:.0f} km². Draw a smaller box."
        )
    return (min_lon, min_lat, max_lon, max_lat)


def bbox_area_km2(bbox: BBox) -> float:
    """Approximate area of a lon/lat box in km² (good enough for a size check)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = math.radians((min_lat + max_lat) / 2)
    width_km = (max_lon - min_lon) * 111.32 * math.cos(mid_lat)  # degrees of lon shrink toward the poles
    height_km = (max_lat - min_lat) * 110.57
    return width_km * height_km


def open_catalog() -> pystac_client.Client:
    """Connect to the Planetary Computer STAC API (free, no key).

    `sign_inplace` adds a temporary access token to every file link we get back.
    """
    return pystac_client.Client.open(config.STAC_URL, modifier=planetary_computer.sign_inplace)


def find_least_cloudy_scene(
    bbox: BBox,
    date_range: str,
    max_cloud_pct: float = config.MAX_SCENE_CLOUD_PCT,
    catalog: pystac_client.Client | None = None,
) -> pystac.Item:
    """Return the least-cloudy Sentinel-2 L2A scene over `bbox` in `date_range`.

    date_range: "YYYY-MM-DD/YYYY-MM-DD", e.g. "2025-06-01/2025-08-31".

    Scenes whose footprint fully contains the box are preferred. Partly covered
    scenes (at the edge of a satellite pass) are only used if there is nothing else.
    """
    bbox = validate_bbox(bbox)
    catalog = catalog or open_catalog()

    # Search WITHOUT a cloud filter, then filter here. That way we can tell
    # "no images exist" apart from "images exist but all are too cloudy".
    search = catalog.search(collections=[config.COLLECTION], bbox=bbox, datetime=date_range)
    all_items = list(search.items())
    if not all_items:
        raise LookupError(
            f"No Sentinel-2 images exist for {date_range} over this area. "
            f"The archive starts on {config.SENTINEL2_START} and has regular coverage "
            f"from {config.SENTINEL2_FULL_COVERAGE}. Pick a later period."
        )

    items = [it for it in all_items if it.properties["eo:cloud_cover"] < max_cloud_pct]
    if not items:
        least = min(it.properties["eo:cloud_cover"] for it in all_items)
        raise LookupError(
            f"Found {len(all_items)} images for {date_range}, but all are too cloudy "
            f"(the clearest has {least:.0f}% cloud, the limit is {max_cloud_pct:.0f}%). "
            "Try a longer period or a drier season."
        )

    area = box(*bbox)
    full_cover = [it for it in items if shape(it.geometry).contains(area)]
    candidates = full_cover or items

    # Lowest cloud cover wins. On a tie, the earlier scene wins (stable, repeatable result).
    return min(candidates, key=lambda it: (it.properties["eo:cloud_cover"], it.datetime))


def find_scene_pair(
    bbox: BBox,
    before_range: str,
    after_range: str,
    max_cloud_pct: float = config.MAX_SCENE_CLOUD_PCT,
) -> tuple[pystac.Item, pystac.Item]:
    """Find the best 'before' and 'after' scenes, reusing one catalog connection."""
    catalog = open_catalog()
    before = find_least_cloudy_scene(bbox, before_range, max_cloud_pct, catalog)
    after = find_least_cloudy_scene(bbox, after_range, max_cloud_pct, catalog)
    return before, after


def describe_scene(item: pystac.Item, bbox: BBox | None = None) -> dict:
    """Plain-dict summary of a scene (easy to print, save as JSON, or show in the UI)."""
    info = {
        "id": item.id,
        "date": item.datetime.date().isoformat(),
        "satellite": item.properties.get("platform"),
        "tile": item.properties.get("s2:mgrs_tile"),
        "cloud_cover_pct": round(item.properties["eo:cloud_cover"], 1),
        "processing_baseline": item.properties.get("s2:processing_baseline"),
        "crs": item.properties.get("proj:code"),
    }
    if bbox is not None:
        info["covers_whole_area"] = shape(item.geometry).contains(box(*bbox))
    return info


if __name__ == "__main__":
    # Quick manual check: python -m change_monitor.search
    print(f"Area: {config.DEMO_NAME}, {bbox_area_km2(config.DEMO_BBOX):.1f} km²")
    before, after = find_scene_pair(config.DEMO_BBOX, config.DEMO_BEFORE, config.DEMO_AFTER)
    for label, item in (("before", before), ("after", after)):
        print(f"{label:>6}: {describe_scene(item, config.DEMO_BBOX)}")
