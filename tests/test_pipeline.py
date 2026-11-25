"""Offline tests for pipeline.py helpers (the full run needs the internet, so it's checked by hand)."""

from change_monitor import config
from change_monitor.pipeline import main, parse_args, slugify


def test_slugify_makes_safe_folder_names():
    assert slugify("Berlin Adlershof!") == "berlin-adlershof"
    assert slugify("  ") == "area"


def test_defaults_are_the_berlin_demo():
    args = parse_args([])
    assert tuple(args.bbox) == config.DEMO_BBOX
    assert args.before == config.DEMO_BEFORE
    assert args.no_ai is False
    assert args.ai == "auto"


def test_bad_bbox_gives_a_clear_error_not_a_crash(capsys):
    exit_code = main(["--bbox", "13.58", "52.41", "13.50", "52.45", "--no-ai"])
    assert exit_code == 1
    assert "Error:" in capsys.readouterr().out
