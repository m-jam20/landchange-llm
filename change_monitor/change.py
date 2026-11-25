"""Step 5: compare the two dates, label each pixel, and measure hectares.

Every clear pixel gets exactly ONE label, so the hectares always add up to the
clear area (nothing is counted twice). Labels are checked in priority order:

  1. water_gain       NDWI went from <= 0 to > 0, and rose by more than the threshold
  2. water_loss       NDWI went from > 0 to <= 0, and fell by more than the threshold
  3. new_built_up     NDBI rose AND NDVI fell, both by more than the threshold
                      (plants replaced by buildings, roads or bare soil)
  4. vegetation_loss  NDVI fell by more than the threshold (any other plant loss)
  5. vegetation_gain  NDVI rose by more than the threshold
  0. no_change        everything else that is clear
 -1. not_clear        cloud, shadow or no data on either date (never counted as change)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from change_monitor import config
from change_monitor.indices import compute_indices
from change_monitor.load import ScenePair

NOT_CLEAR = -1
NO_CHANGE = 0
# code -> name, in priority order (first match wins)
CATEGORIES = {
    1: "water_gain",
    2: "water_loss",
    3: "new_built_up",
    4: "vegetation_loss",
    5: "vegetation_gain",
}


def classify_change(
    before: xr.Dataset,
    after: xr.Dataset,
    clear: xr.DataArray,
    threshold: float = config.CHANGE_THRESHOLD,
    water: float = config.WATER_NDWI,
) -> xr.DataArray:
    """Label every pixel with one change code (see the table at the top of this file).

    `before` and `after` hold the indices (ndvi, ndbi, ndwi) of each date.
    """
    d = after - before  # positive = the index went up

    rules = {
        "water_gain": (before.ndwi <= water) & (after.ndwi > water) & (d.ndwi > threshold),
        "water_loss": (before.ndwi > water) & (after.ndwi <= water) & (d.ndwi < -threshold),
        "new_built_up": (d.ndbi > threshold) & (d.ndvi < -threshold),
        "vegetation_loss": d.ndvi < -threshold,
        "vegetation_gain": d.ndvi > threshold,
    }

    codes = xr.full_like(clear, NO_CHANGE, dtype="int8")
    # Go from lowest to highest priority, so higher-priority labels overwrite lower ones.
    for code, name in reversed(CATEGORIES.items()):
        codes = codes.where(~rules[name], code)
    return codes.where(clear, NOT_CLEAR).astype("int8").rename("change")


def measure_change(
    change_map: xr.DataArray,
    before: xr.Dataset | None = None,
    after: xr.Dataset | None = None,
    pixel_area_ha: float = config.PIXEL_AREA_HA,
) -> dict:
    """Hectares per label, plus context numbers the report needs. Returns a plain dict."""
    values = change_map.values
    total_px = values.size
    clear_px = int((values != NOT_CLEAR).sum())

    def ha(n_pixels: int) -> float:
        return round(n_pixels * pixel_area_ha, 2)

    stats = {
        "total_ha": ha(total_px),
        "clear_ha": ha(clear_px),
        "clear_pct": round(100 * clear_px / total_px, 1) if total_px else 0.0,
        "no_change_ha": ha(int((values == NO_CHANGE).sum())),
        "change_ha": {},
        "change_pct_of_clear": {},
    }
    for code, name in CATEGORIES.items():
        n = int((values == code).sum())
        stats["change_ha"][name] = ha(n)
        stats["change_pct_of_clear"][name] = round(100 * n / clear_px, 2) if clear_px else 0.0

    # Average index values over the clear pixels: useful context ("the area got greener").
    if before is not None and after is not None:
        clear = change_map != NOT_CLEAR
        stats["mean_index"] = {
            name: {
                "before": round(float(before[name].where(clear).mean()), 3),
                "after": round(float(after[name].where(clear).mean()), 3),
            }
            for name in ("ndvi", "ndbi", "ndwi")
        }
    return stats


@dataclass
class ChangeResult:
    """Everything step 5 produces: indices for both dates, the label map, and the numbers."""

    before_idx: xr.Dataset
    after_idx: xr.Dataset
    change_map: xr.DataArray
    stats: dict


def detect_change(pair: ScenePair, threshold: float = config.CHANGE_THRESHOLD) -> ChangeResult:
    """Steps 4 + 5 in one call: indices for both dates -> labels -> hectares."""
    before_idx = compute_indices(pair.before)
    after_idx = compute_indices(pair.after)
    change_map = classify_change(before_idx, after_idx, pair.clear, threshold)
    stats = measure_change(change_map, before_idx, after_idx)
    stats["threshold"] = threshold
    return ChangeResult(before_idx, after_idx, change_map, stats)


if __name__ == "__main__":
    # Quick manual check: python -m change_monitor.change
    import json

    from change_monitor.load import load_pair
    from change_monitor.search import find_scene_pair

    b_item, a_item = find_scene_pair(config.DEMO_BBOX, config.DEMO_BEFORE, config.DEMO_AFTER)
    result = detect_change(load_pair(b_item, a_item, config.DEMO_BBOX))
    print(json.dumps(result.stats, indent=2))
    print("unique codes:", sorted(np.unique(result.change_map.values).tolist()))
