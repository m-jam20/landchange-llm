"""Step 6: pictures a human can check — before/after true colour and a change map.

We use matplotlib's `Figure` directly instead of `pyplot`. pyplot keeps every
figure in a global list until you close it; in a long-running app (Streamlit)
that slowly eats memory. A plain `Figure` is freed like any other object.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from change_monitor import config
from change_monitor.change import CATEGORIES, NO_CHANGE, NOT_CLEAR

B = config.BANDS

# Colour and readable name for every code in the change map.
LEGEND = {
    NOT_CLEAR: ("#000000", "Not clear (cloud / shadow)"),
    NO_CHANGE: ("#e8e8e8", "No change"),
    1: ("#2b8cbe", "Water gain"),
    2: ("#a6611a", "Water loss"),
    3: ("#e41a1c", "New built-up / bare ground"),
    4: ("#fdae61", "Vegetation loss"),
    5: ("#1a9850", "Vegetation gain"),
}
assert set(LEGEND) == {NOT_CLEAR, NO_CHANGE, *CATEGORIES}, "every change code needs a colour"


def stretch_limits(*datasets: xr.Dataset, low: float = 2, high: float = 98) -> tuple[float, float]:
    """Brightness range shared by all images, from the 2nd-98th percentile.

    Using the SAME range for both dates means a brighter 'after' image really
    is brighter on the ground, not just stretched differently.
    """
    values = np.concatenate(
        [np.ravel(ds[band].values) for ds in datasets for band in (B["red"], B["green"], B["blue"])]
    )
    lo, hi = np.nanpercentile(values, (low, high))
    return float(lo), float(hi)


def true_color(ds: xr.Dataset, limits: tuple[float, float]) -> np.ndarray:
    """Height x width x 3 array (red, green, blue) scaled to 0-1 for display. No data = white."""
    lo, hi = limits
    rgb = np.dstack([ds[B["red"]].values, ds[B["green"]].values, ds[B["blue"]].values])
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(rgb, nan=1.0)


def render_true_color(ds: xr.Dataset, path: Path, title: str, limits: tuple[float, float]) -> Path:
    """Save one true-colour picture as PNG."""
    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot()
    ax.imshow(true_color(ds, limits))
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def render_change_map(change_map: xr.DataArray, path: Path, title: str) -> Path:
    """Save the change map as PNG, with a legend showing only the labels that occur."""
    codes = sorted(LEGEND)  # -1, 0, 1, ... 5
    cmap = ListedColormap([LEGEND[c][0] for c in codes])
    index = change_map.values - codes[0]  # shift so -1 -> colour 0

    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot()
    ax.imshow(index, cmap=cmap, vmin=0, vmax=len(codes) - 1, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    present = set(np.unique(change_map.values).tolist())
    handles = [Patch(color=LEGEND[c][0], label=LEGEND[c][1]) for c in codes if c in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def render_all(
    before: xr.Dataset,
    after: xr.Dataset,
    change_map: xr.DataArray,
    out_dir: str | Path,
    before_label: str = "Before",
    after_label: str = "After",
) -> dict[str, Path]:
    """Save before.png, after.png and change_map.png into `out_dir`. Returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = stretch_limits(before, after)
    return {
        "before": render_true_color(before, out_dir / "before.png", before_label, limits),
        "after": render_true_color(after, out_dir / "after.png", after_label, limits),
        "change_map": render_change_map(change_map, out_dir / "change_map.png", "Change"),
    }
