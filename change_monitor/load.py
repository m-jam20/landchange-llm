"""Step 3: load the bands for both dates on one shared grid and mask clouds.

Key ideas:
- Both dates go on the *same* 10 m grid, so pixel [i, j] is the same spot on the ground.
- Pixel values become reflectance (0-1), including the +1000 offset fix for newer scenes.
- A pixel is only "clear" if the SCL band says so on *both* dates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import odc.stac
import planetary_computer
import pystac
import xarray as xr

from change_monitor import config
from change_monitor.search import BBox, validate_bbox

# Reflectance bands we load (everything in config.BANDS except the SCL mask).
REFLECTANCE_BANDS = [code for name, code in config.BANDS.items() if name != "scl"]
SCL_BAND = config.BANDS["scl"]


@dataclass
class ScenePair:
    """Both dates on the same grid, plus the pixels that are clear on both."""

    before: xr.Dataset  # reflectance bands, float32, NaN = no data
    after: xr.Dataset
    clear: xr.DataArray  # True where both dates are cloud-free and have data
    crs: str

    @property
    def clear_pct(self) -> float:
        return float(self.clear.mean()) * 100

    def summary(self) -> dict:
        """Plain-dict summary (for printing, JSON, or the UI)."""
        height, width = self.clear.shape
        return {
            "crs": self.crs,
            "width_px": width,
            "height_px": height,
            "area_ha": round(width * height * config.PIXEL_AREA_HA, 1),
            "clear_pct": round(self.clear_pct, 1),
        }


def utm_crs_for_bbox(bbox: BBox) -> str:
    """The UTM zone (in metres) that contains the centre of the box, e.g. 'EPSG:32633'.

    Chosen from the box itself, not from a scene, so both dates always get the same grid.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = (min_lon + max_lon) / 2
    lat = (min_lat + max_lat) / 2
    zone = min(int(math.floor((lon + 180) / 6)) + 1, 60)
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def reflectance_offset(processing_baseline: str | None) -> int:
    """-1000 for scenes processed with baseline 04.00 or newer (from 25 Jan 2022), else 0."""
    if not processing_baseline:
        return 0

    def as_tuple(version: str) -> tuple[int, ...]:
        return tuple(int(part) for part in version.split("."))

    if as_tuple(processing_baseline) >= as_tuple(config.OFFSET_BASELINE):
        return config.REFLECTANCE_OFFSET_NEW
    return 0


def to_reflectance(raw: xr.Dataset, processing_baseline: str | None) -> xr.Dataset:
    """Stored whole numbers -> reflectance (0-1) as float32.

    Raw value 0 means "no data" and becomes NaN. Tiny negative results
    (very dark pixels after the offset) are clipped to 0.
    """
    offset = reflectance_offset(processing_baseline)
    out = {}
    for band in REFLECTANCE_BANDS:
        values = raw[band].astype("float32")
        refl = (values + offset) / config.REFLECTANCE_SCALE
        out[band] = refl.where(values > 0).clip(min=0)
    return xr.Dataset(out)


def clear_mask(scl: xr.DataArray) -> xr.DataArray:
    """True where the SCL class is one we trust (vegetation, bare, water, unclassified)."""
    return scl.isin(config.SCL_CLEAR_CLASSES)


def load_scene(item: pystac.Item, bbox: BBox | None = None, crs: str | None = None, geobox=None) -> xr.Dataset:
    """Download the bands of one scene, cut to the box, as stored (uint16, 0 = no data).

    Give either (bbox + crs) to define a new grid, or `geobox` to reuse an existing grid.
    """
    item = planetary_computer.sign(item)  # fresh access token; old ones expire
    grid = {"geobox": geobox} if geobox is not None else {
        "bbox": bbox,
        "crs": crs,
        "resolution": config.PIXEL_SIZE_M,
    }
    ds = odc.stac.load(
        [item],
        bands=[*REFLECTANCE_BANDS, SCL_BAND],
        dtype="uint16",
        nodata=0,
        resampling="nearest",  # keeps SCL classes intact and values unchanged
        **grid,
    )
    return ds.isel(time=0)  # one scene -> drop the time dimension


def load_pair(before_item: pystac.Item, after_item: pystac.Item, bbox: BBox) -> ScenePair:
    """Load both scenes on one shared grid and build the 'clear on both dates' mask."""
    bbox = validate_bbox(bbox)
    crs = utm_crs_for_bbox(bbox)

    raw_before = load_scene(before_item, bbox=bbox, crs=crs)
    raw_after = load_scene(after_item, geobox=raw_before.odc.geobox)  # exactly the same pixels

    before = to_reflectance(raw_before, before_item.properties.get("s2:processing_baseline"))
    after = to_reflectance(raw_after, after_item.properties.get("s2:processing_baseline"))

    has_data = before.to_array().notnull().all("variable") & after.to_array().notnull().all("variable")
    clear = clear_mask(raw_before[SCL_BAND]) & clear_mask(raw_after[SCL_BAND]) & has_data

    return ScenePair(before=before, after=after, clear=clear.drop_vars("time", errors="ignore"), crs=crs)


if __name__ == "__main__":
    # Quick manual check: python -m change_monitor.load
    from change_monitor.search import find_scene_pair

    b_item, a_item = find_scene_pair(config.DEMO_BBOX, config.DEMO_BEFORE, config.DEMO_AFTER)
    pair = load_pair(b_item, a_item, config.DEMO_BBOX)
    print(pair.summary())
    for label, ds in (("before", pair.before), ("after", pair.after)):
        medians = {band: round(float(np.nanmedian(ds[band])), 3) for band in REFLECTANCE_BANDS}
        print(f"{label:>6} median reflectance: {medians}")
