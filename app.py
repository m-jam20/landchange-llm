"""Streamlit web UI.  Start with:  streamlit run app.py

This file only does layout. All the real work happens in run_pipeline(),
the same function the command line uses.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from change_monitor import config
from change_monitor.pipeline import run_pipeline
from change_monitor.report import available_providers, describe_provider
from change_monitor.search import bbox_area_km2, validate_bbox
from change_monitor.ui_helpers import bbox_from_geojson, change_table, check_periods, date_range, zip_folder

st.set_page_config(page_title="LandChange LLM", page_icon="🛰️", layout="wide")

NO_AI = "none"  # dropdown value for the basic report


@st.cache_data(show_spinner=False, max_entries=20)
def run_cached(bbox, before, after, name, threshold, ai_choice, ai_description, _progress=None) -> dict:
    """Run the pipeline once per unique set of inputs.

    Streamlit reruns this whole file on every click. Without the cache, every rerun
    would download the images again (and call the AI again).
    `ai_choice` and `ai_description` (company + model) are part of the key: switching
    AI, or changing the model in .env, runs again instead of returning an old report.
    `_progress` starts with "_" so Streamlit leaves it out of the cache key.
    """
    result = run_pipeline(
        bbox, before, after, name=name, threshold=threshold,
        use_ai=ai_choice != NO_AI, ai_provider=ai_choice if ai_choice != NO_AI else "auto",
        progress=_progress or (lambda msg: None),
    )
    # Return plain data (bytes, dicts) — easy for the cache to store.
    return {
        "stats": result.stats,
        "images": {key: Path(path).read_bytes() for key, path in result.images.items()},
        "report": result.report.text,
        "report_source": result.report.source,
        "report_note": result.report.note,
        "out_dir": str(result.out_dir),
    }


# ---------------------------------------------------------------- sidebar: settings
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Area name", config.DEMO_NAME)
    first_day = date.fromisoformat(config.SENTINEL2_START)
    before_dates = st.date_input("Before period", value=(date(2023, 6, 1), date(2023, 8, 31)), min_value=first_day)
    after_dates = st.date_input("After period", value=(date(2025, 6, 1), date(2025, 8, 31)), min_value=first_day)
    st.caption(
        f"Images exist from {first_day:%d %b %Y}. Before {config.SENTINEL2_FULL_COVERAGE} there are "
        "about half as many, so use periods of 2–3 months."
    )
    threshold = st.slider(
        "Change threshold", 0.05, 0.5, config.CHANGE_THRESHOLD, 0.05,
        help="How much an index must change to count. Lower = more sensitive, but more noise.",
    )
    # Every AI that is set up in .env, plus the basic report. The first one is the default.
    ai_options = [*available_providers(), NO_AI]
    ai_choice = st.selectbox(
        "Report written by",
        ai_options,
        format_func=lambda option: "Basic report (no AI)" if option == NO_AI else describe_provider(option),
    )
    if ai_options == [NO_AI]:
        st.caption(
            "No AI key found, so a basic report is written by code. For an AI report, add a key to `.env` "
            "(a free one: [aistudio.google.com](https://aistudio.google.com) → `GEMINI_API_KEY`; "
            "other companies: see `.env.example`) and refresh this page."
        )

# ---------------------------------------------------------------- header
st.title("🛰️ LandChange LLM")
st.caption(
    "How did a place change between two dates? Free Sentinel-2 satellite images, "
    "measured in hectares. Draw a rectangle on the map, pick two periods, and run."
)

# ---------------------------------------------------------------- 1. area on a map
st.subheader("1. Choose the area")
bbox = st.session_state.get("bbox", config.DEMO_BBOX)
west, south, east, north = bbox

m = folium.Map(tiles="OpenStreetMap")
m.fit_bounds([[south, west], [north, east]])
folium.Rectangle(
    [[south, west], [north, east]], color="#e41a1c", weight=2, fill=False, tooltip="Current area"
).add_to(m)
Draw(
    draw_options={"rectangle": True, "polygon": False, "polyline": False,
                  "circle": False, "marker": False, "circlemarker": False},
    edit_options={"edit": False, "remove": False},
).add_to(m)
map_state = st_folium(m, key="map", height=420, use_container_width=True, returned_objects=["last_active_drawing"])

drawing = (map_state or {}).get("last_active_drawing")
if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
    new_bbox = bbox_from_geojson(drawing)
    if new_bbox != bbox:
        st.session_state["bbox"] = new_bbox
        st.rerun()  # redraw the map around the new box

area_km2 = bbox_area_km2(bbox)
st.caption(
    f"Area: **{area_km2:.1f} km²** · west {west}, south {south}, east {east}, north {north} · "
    "use the ▭ button on the map to draw a new rectangle"
)

# ---------------------------------------------------------------- 2. check inputs, run
st.subheader("2. Run")
problems = []
try:
    validate_bbox(bbox)
    before_range, after_range = date_range(before_dates), date_range(after_dates)
    for warning in check_periods(before_dates, after_dates):
        st.warning(warning)
except ValueError as e:
    problems.append(str(e))
for problem in problems:
    st.error(problem)

if st.button("Run analysis", type="primary", disabled=bool(problems)):
    with st.status("Running...", expanded=True) as status:
        try:
            st.session_state["result"] = run_cached(
                tuple(bbox), before_range, after_range, name, threshold,
                ai_choice, describe_provider(ai_choice) if ai_choice != NO_AI else NO_AI,
                _progress=status.write,
            )
            status.update(label="Done", state="complete", expanded=False)
        except (ValueError, LookupError) as e:  # bad input or no usable scene
            status.update(label="Could not run", state="error")
            st.error(str(e))
            # Also print to the terminal, so the problem shows up in the server log.
            message = f"[run failed] bbox={bbox} before={before_range} after={after_range}: {e}"
            print(message, file=sys.stderr, flush=True)

# ---------------------------------------------------------------- 3. results
result = st.session_state.get("result")
if result:
    stats = result["stats"]
    scenes = stats["scenes"]
    st.subheader(f"3. Results: {stats['name']}")

    total_change = sum(stats["change_ha"].values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Area", f"{stats['total_ha']:,.0f} ha")
    c2.metric("Clear on both dates", f"{stats['clear_pct']}%")
    c3.metric("Changed", f"{total_change:,.1f} ha")
    c4.metric("Threshold", stats["threshold"])
    if stats["clear_pct"] < 80:
        st.warning(
            f"Only {stats['clear_pct']}% of the area was clear on both dates, so the numbers cover only that part. "
            "Real clouds are one cause. Another: very bright ground (salt flats, white sand, snow) is often "
            "mistaken for cloud by the satellite's cloud labels (SCL). Compare the photos below to see which it is."
        )

    def scene_caption(scene: dict) -> str:
        return f"{scene['date']} · {scene['satellite']} · {scene['cloud_cover_pct']}% tile cloud"

    left, right = st.columns(2)
    left.image(result["images"]["before"], width="stretch", caption=scene_caption(scenes["before"]))
    right.image(result["images"]["after"], width="stretch", caption=scene_caption(scenes["after"]))

    left, right = st.columns(2)
    left.image(result["images"]["change_map"], width="stretch")
    with right:
        st.dataframe(change_table(stats), hide_index=True, width="stretch")
        st.caption(
            "An automatic estimate from two images and a threshold, not a field survey. "
            "Thin lines along rivers and roads are usually edge noise."
        )

    st.subheader("Report")
    st.caption(f"Written by: {result['report_source']}")
    if result["report_note"] and result["report_note"] != "AI report turned off.":
        st.info(result["report_note"])
    with st.container(border=True):
        st.markdown(result["report"])

    st.download_button(
        "Download all results (.zip)",
        data=zip_folder(result["out_dir"]),
        file_name=f"{Path(result['out_dir']).name}.zip",
        mime="application/zip",
    )
