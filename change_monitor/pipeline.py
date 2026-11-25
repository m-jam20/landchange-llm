"""Run every step in order and save the results.

    python -m change_monitor.pipeline                      # Berlin demo
    python -m change_monitor.pipeline --bbox 13.50 52.41 13.58 52.45 \\
        --before 2023-06-01/2023-08-31 --after 2025-06-01/2025-08-31 --name berlin

The command line (main) and the Streamlit UI both call run_pipeline(), so the
logic exists only once.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from change_monitor import config
from change_monitor.change import detect_change
from change_monitor.load import load_pair
from change_monitor.render import render_all
from change_monitor.report import PROVIDERS, Report, write_report
from change_monitor.search import BBox, describe_scene, find_scene_pair, validate_bbox


@dataclass
class PipelineResult:
    """Everything one run produces. `stats` is the full plain-dict record (also saved as stats.json)."""

    out_dir: Path
    stats: dict
    images: dict[str, Path]
    report: Report  # always present: from Gemini/Claude, or the basic one written by code
    extra: dict = field(default_factory=dict)  # in-memory objects (pair, change result) for the UI


def slugify(text: str) -> str:
    """'Berlin Adlershof!' -> 'berlin-adlershof' (safe for folder names)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "area"


def run_pipeline(
    bbox: BBox,
    before_range: str,
    after_range: str,
    name: str = "area",
    threshold: float = config.CHANGE_THRESHOLD,
    out_root: str | Path = config.OUTPUT_DIR,
    use_ai: bool = True,
    ai_provider: str = config.REPORT_PROVIDER,
    progress=print,
) -> PipelineResult:
    """Steps 1-7: find scenes -> load + mask -> indices + change -> pictures -> (report).

    `progress` is called with a short message before each step (print in the terminal,
    a status line in the UI).
    """
    bbox = validate_bbox(bbox)

    progress("1/5 Searching for the least-cloudy scenes...")
    before_item, after_item = find_scene_pair(bbox, before_range, after_range)
    scenes = {"before": describe_scene(before_item, bbox), "after": describe_scene(after_item, bbox)}

    progress("2/5 Loading bands and masking clouds...")
    pair = load_pair(before_item, after_item, bbox)

    progress("3/5 Computing indices and measuring change...")
    change = detect_change(pair, threshold)

    # Same inputs -> same folder name, so a re-run overwrites instead of piling up copies.
    run_name = f"{slugify(name)}_{scenes['before']['date']}_{scenes['after']['date']}"
    out_dir = Path(out_root) / run_name

    progress("4/5 Rendering pictures...")
    images = render_all(
        pair.before,
        pair.after,
        change.change_map,
        out_dir,
        before_label=f"Before: {scenes['before']['date']}",
        after_label=f"After: {scenes['after']['date']}",
    )

    record = {
        "name": name,
        "bbox": list(bbox),
        "before_range": before_range,
        "after_range": after_range,
        "scenes": scenes,
        "grid": pair.summary(),
        **change.stats,
    }

    progress("5/5 Writing the report (with AI this can take up to a minute)...")
    context = {
        "place": name,
        "bbox_lon_lat": list(bbox),
        "before_image_date": scenes["before"]["date"],
        "after_image_date": scenes["after"]["date"],
        "change_threshold": threshold,
    }
    report = write_report(change.stats, context, use_ai=use_ai, provider=ai_provider)
    if report.note:
        progress(f"    {report.note}")
    (out_dir / "report.md").write_text(report.text + "\n", encoding="utf-8")

    record["report_source"] = report.source
    record["report_note"] = report.note
    (out_dir / "stats.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    return PipelineResult(
        out_dir=out_dir,
        stats=record,
        images=images,
        report=report,
        extra={"pair": pair, "change": change},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m change_monitor.pipeline",
        description="Measure how a place changed between two dates using Sentinel-2 imagery.",
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=config.DEMO_BBOX, help="area in lon/lat (default: Berlin demo)",
    )
    parser.add_argument("--before", default=config.DEMO_BEFORE, help="date range YYYY-MM-DD/YYYY-MM-DD")
    parser.add_argument("--after", default=config.DEMO_AFTER, help="date range YYYY-MM-DD/YYYY-MM-DD")
    parser.add_argument("--name", default=config.DEMO_NAME, help="a short name for the area")
    parser.add_argument("--threshold", type=float, default=config.CHANGE_THRESHOLD)
    parser.add_argument("--out", default=config.OUTPUT_DIR, help="folder for results")
    parser.add_argument("--no-ai", action="store_true", help="write the basic report without AI")
    parser.add_argument(
        "--ai", default=config.REPORT_PROVIDER, choices=["auto", *PROVIDERS],
        help="which AI writes the report (default: the first one set up in .env)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_pipeline(
            tuple(args.bbox), args.before, args.after, name=args.name,
            threshold=args.threshold, out_root=args.out, use_ai=not args.no_ai, ai_provider=args.ai,
        )
    except (ValueError, LookupError) as e:  # bad input or no scenes: a clear message, no traceback
        print(f"Error: {e}")
        return 1

    s = result.stats
    print(f"\nResults saved to {result.out_dir}/")
    print(f"Area {s['total_ha']} ha, clear on both dates: {s['clear_pct']}%")
    for category, ha in s["change_ha"].items():
        print(f"  {category:<16} {ha:>8.2f} ha")
    print(f"\nReport ({result.report.source}):\n")
    print(result.report.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
