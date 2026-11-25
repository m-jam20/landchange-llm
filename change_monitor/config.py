"""All settings in one place.

Other modules import from here instead of hard-coding numbers, so a value
(a threshold, a band name, a limit) only ever needs to change in one file.
"""

# --- Where the data comes from -------------------------------------------
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Scenes with more cloud than this (in % of the whole 110 km tile) are skipped.
MAX_SCENE_CLOUD_PCT = 30

# First Sentinel-2 image in the catalog (Sentinel-2A launched 23 June 2015).
# Until 2017 only one satellite flew, so early years have about half as many images.
SENTINEL2_START = "2015-06-27"
SENTINEL2_FULL_COVERAGE = "2017"

# --- Bands ---------------------------------------------------------------
# Sentinel-2 asset names on Planetary Computer.
BANDS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "swir": "B11",
    "scl": "SCL",
}

# Stored pixel values -> reflectance (0-1): (value + OFFSET) / SCALE.
# Since processing baseline 04.00 (25 Jan 2022), values carry +1000,
# so the offset is -1000 for those scenes and 0 for older ones.
REFLECTANCE_SCALE = 10_000
REFLECTANCE_OFFSET_NEW = -1000
OFFSET_BASELINE = "04.00"

# --- Cloud mask (SCL classes we keep) ------------------------------------
# 4 vegetation, 5 not vegetated, 6 water, 7 unclassified.
# Everything else (no data, shadows, clouds, cirrus, snow) is masked out.
SCL_CLEAR_CLASSES = (4, 5, 6, 7)

# --- Grid ----------------------------------------------------------------
PIXEL_SIZE_M = 10
PIXEL_AREA_HA = PIXEL_SIZE_M * PIXEL_SIZE_M / 10_000  # 0.01 ha

# Largest area we accept, so a CPU-only machine doesn't run out of memory.
MAX_AREA_KM2 = 625  # 25 x 25 km

# --- Change thresholds ---------------------------------------------------
# An index must change by more than this to count as real change.
CHANGE_THRESHOLD = 0.2

# NDWI above this means open water (the usual cut-off for this index is 0).
WATER_NDWI = 0.0

# --- Outputs -------------------------------------------------------------
OUTPUT_DIR = "outputs"

# --- Report --------------------------------------------------------------
# Which AI writes the report. "auto" = Gemini if GEMINI_API_KEY is set, else Claude if
# ANTHROPIC_API_KEY is set, else a basic report written by code (no AI, always works).
REPORT_PROVIDER = "auto"
GEMINI_MODEL = "gemini-3.5-flash-lite"  # free tier; small and fast (~3 s). gemini-3.8-flash took 60-90 s
CLAUDE_MODEL = "claude-opus-5-5"  # paid
REPORT_MAX_TOKENS = 8000  # includes the model's thinking, so keep some room

# Gemini "thinks" before answering. A short summary of numbers needs little thinking,
# and less thinking = a much faster answer. Levels: minimal, low, medium, high.
GEMINI_THINKING_LEVEL = "minimal"
# Stop waiting for the AI after this many seconds and use the basic report instead.
REPORT_TIMEOUT_S = 60

# --- Demo area: south-east Berlin (Adlershof / Johannisthal) --------------
# bbox order: (min_lon, min_lat, max_lon, max_lat) = west, south, east, north
DEMO_NAME = "Berlin-Adlershof"
DEMO_BBOX = (13.50, 52.41, 13.58, 52.45)
DEMO_BEFORE = "2023-06-01/2023-08-31"  # same season in both years
DEMO_AFTER = "2025-06-01/2025-08-31"
