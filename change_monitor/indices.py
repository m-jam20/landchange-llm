"""Step 4: spectral indices (one number per pixel, from -1 to +1).

NDVI  vegetation  (NIR - Red)  / (NIR + Red)   high = green plants
NDBI  built-up    (SWIR - NIR) / (SWIR + NIR)  high = buildings, bare soil
NDWI  water       (Green - NIR)/ (Green + NIR) above 0 = open water
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from change_monitor import config

B = config.BANDS  # short alias: B["red"] == "B04"


def normalized_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """(a - b) / (a + b). Gives NaN (not a crash, not infinity) where a + b == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (a - b) / (a + b)
    return nd.where(np.isfinite(nd)).astype("float32")


def compute_indices(ds: xr.Dataset) -> xr.Dataset:
    """NDVI, NDBI and NDWI for one date. `ds` must hold reflectance (0-1), not raw values."""
    return xr.Dataset(
        {
            "ndvi": normalized_difference(ds[B["nir"]], ds[B["red"]]),
            "ndbi": normalized_difference(ds[B["swir"]], ds[B["nir"]]),
            "ndwi": normalized_difference(ds[B["green"]], ds[B["nir"]]),
        }
    )
