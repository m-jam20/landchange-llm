"""Step 7: turn the change numbers into a short plain-language report.

Three ways to write it, tried in this order (config.REPORT_PROVIDER = "auto"):
  1. Gemini  — if GEMINI_API_KEY is set (free tier)
  2. Claude  — if ANTHROPIC_API_KEY is set (paid)
  3. Basic   — fixed sentences filled in by code. No AI, free, always works.

The AI only sees numbers and context (place, dates, threshold, clear %), never
the images, so it cannot invent pixel counts. If the AI call fails for any
reason, the pipeline falls back to the basic report and records why.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from change_monitor import config
from change_monitor.change import CATEGORIES
from change_monitor.render import LEGEND

SYSTEM_PROMPT = """You write short, factual land-change reports for non-experts, \
based on satellite change statistics.

Rules:
- Use only the numbers you are given. Never invent figures.
- Take headline figures from "headline". "Visible on both dates" (no cloud) is NOT the same as \
"unchanged": never use one for the other.
- Round sensibly (e.g. "about 30 hectares"). One hectare is about 1.4 football pitches.
- Lead with the largest real change. Mention changes under ~0.1% of the area only as minor or likely noise.
- Do not name specific buildings, companies or projects: the data cannot identify them.
- "New built-up / bare ground" can be finished buildings, construction sites or cleared land.
- State the limits in one or two sentences: this is an automatic estimate from two satellite \
images and a threshold, not a field survey; thin lines along rivers and roads are often edge noise.
- Format: Markdown, a one-line title, then 3 short sections: **Summary**, **Main changes**, \
**How reliable is this?** Under 250 words in total."""

KEY_NAMES = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}
MODELS = {"gemini": config.GEMINI_MODEL, "claude": config.CLAUDE_MODEL}


class ReportUnavailable(Exception):
    """The AI report could not be written (no key, no internet, refused...). Not fatal."""


@dataclass
class Report:
    text: str  # Markdown
    source: str  # who wrote it, e.g. "gemini-3.8-flash" or "basic (no AI)"
    note: str | None = None  # why the AI wasn't used, if it wasn't


# ---------------------------------------------------------------- keys and choice
def get_api_key(provider: str) -> str | None:
    """The key for 'gemini' or 'claude' from the environment or .env, or None."""
    load_dotenv()
    return os.environ.get(KEY_NAMES[provider]) or None


def available_providers() -> list[str]:
    """Providers that have a key, in preference order (Gemini first: it's free)."""
    return [p for p in ("gemini", "claude") if get_api_key(p)]


def choose_provider(preference: str = config.REPORT_PROVIDER) -> str | None:
    """'gemini', 'claude', or None when no key is set."""
    if preference in KEY_NAMES:
        return preference if get_api_key(preference) else None
    available = available_providers()
    return available[0] if available else None


# ---------------------------------------------------------------- prompt
def build_prompt(stats: dict, context: dict) -> str:
    """The user message: context + numbers as JSON, with readable category names."""
    names = {name: LEGEND[code][1] for code, name in CATEGORIES.items()}
    changed_ha = sum(stats["change_ha"].values())
    clear_ha = stats["clear_ha"]
    # Pre-computed, clearly named headline numbers, so the model never has to derive
    # (and possibly mix up) them. Example mistake this prevents: reading the 98.6%
    # "clear" share as "98.6% unchanged".
    headline = {
        "total_area_ha": stats.get("total_ha"),
        "pct_of_area_visible_on_both_dates": stats["clear_pct"],
        "visible_area_ha": clear_ha,
        "changed_ha": round(changed_ha, 2),
        "pct_of_visible_area_changed": round(100 * changed_ha / clear_ha, 1) if clear_ha else 0.0,
        "pct_of_visible_area_unchanged": round(100 * (clear_ha - changed_ha) / clear_ha, 1) if clear_ha else 0.0,
    }
    payload = {"context": context, "headline": headline, "category_meaning": names, "statistics": stats}
    return "Write the report for this analysis:\n\n" + json.dumps(payload, indent=2)


# ---------------------------------------------------------------- 1. Gemini
def write_with_gemini(stats: dict, context: dict, client=None, model: str = config.GEMINI_MODEL) -> str:
    """Report text from Gemini. Raises ReportUnavailable on any problem."""
    from google import genai
    from google.genai import errors, types

    if client is None:
        key = get_api_key("gemini")
        if key is None:
            raise ReportUnavailable("No GEMINI_API_KEY set.")
        # The timeout is in milliseconds here.
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=config.REPORT_TIMEOUT_S * 1000))

    try:
        response = client.models.generate_content(
            model=model,
            contents=build_prompt(stats, context),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=config.REPORT_MAX_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_level=config.GEMINI_THINKING_LEVEL.upper()),
            ),
        )
    except errors.ClientError as e:  # 4xx: bad key, unknown model, free-tier limit reached
        if e.code == 429:
            raise ReportUnavailable("Gemini free-tier limit reached. Wait a minute and try again.") from e
        if e.code in (400, 401, 403) and "key" in str(e).lower():
            raise ReportUnavailable("The Gemini API key was rejected. Check GEMINI_API_KEY.") from e
        raise ReportUnavailable(f"Gemini error {e.code}: {e.message}") from e
    except errors.APIError as e:  # 5xx: Google's side
        raise ReportUnavailable(f"Gemini server error {e.code}. Try again later.") from e
    except Exception as e:  # network problems and timeouts come from the HTTP library, not from genai
        if "timeout" in type(e).__name__.lower():
            raise ReportUnavailable(f"Gemini did not answer within {config.REPORT_TIMEOUT_S} s.") from e
        raise ReportUnavailable(f"Could not reach Gemini ({type(e).__name__}). No internet?") from e

    text = (response.text or "").strip()
    if not text:
        reason = response.candidates[0].finish_reason if response.candidates else "blocked"
        raise ReportUnavailable(f"Gemini returned no text (reason: {reason}).")
    return text


# ---------------------------------------------------------------- 2. Claude
def write_with_claude(stats: dict, context: dict, client=None, model: str = config.CLAUDE_MODEL) -> str:
    """Report text from Claude. Raises ReportUnavailable on any problem."""
    import anthropic

    if client is None:
        key = get_api_key("claude")
        if key is None:
            raise ReportUnavailable("No ANTHROPIC_API_KEY set.")
        client = anthropic.Anthropic(api_key=key)

    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=config.REPORT_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(stats, context)}],
            output_config={"effort": "medium"},
            # If the model declines, the API retries on a fallback model automatically.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise ReportUnavailable("The Claude API key was rejected. Check ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise ReportUnavailable("Claude rate limit reached. Try again in a minute.") from e
    except anthropic.APIConnectionError as e:
        raise ReportUnavailable("Could not reach the Claude API (no internet?).") from e
    except anthropic.APIStatusError as e:
        raise ReportUnavailable(f"Claude API error {e.status_code}: {e.message}") from e

    if response.stop_reason == "refusal":
        raise ReportUnavailable("Claude declined to write this report.")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise ReportUnavailable(f"Claude returned no text (stop reason: {response.stop_reason}).")
    return text


# ---------------------------------------------------------------- 3. Basic (no AI)
def basic_report(stats: dict, context: dict) -> str:
    """A short Markdown report built from fixed sentences. Same numbers in -> same text out."""
    names = {name: LEGEND[code][1] for code, name in CATEGORIES.items()}
    changes = sorted(stats["change_ha"].items(), key=lambda kv: kv[1], reverse=True)
    total = sum(stats["change_ha"].values())
    clear_ha = stats["clear_ha"]
    threshold = stats.get("threshold", config.CHANGE_THRESHOLD)

    lines = [
        f"# Land change in {context.get('place', 'the area')}: "
        f"{context.get('before_image_date', '?')} to {context.get('after_image_date', '?')}",
        "",
        "**Summary**",
        "",
        f"Of {clear_ha:,.0f} ha that were clearly visible on both dates, about {total:,.1f} ha "
        f"({100 * total / clear_ha if clear_ha else 0:.1f}%) changed.",
        "",
        "**Main changes**",
        "",
    ]
    for name, ha in changes:
        pct = stats["change_pct_of_clear"][name]
        flag = " (minor, may be noise)" if pct < 0.1 else ""
        lines.append(f"- {names[name]}: {ha:,.1f} ha ({pct:.2f}%){flag}")
    lines += [
        "",
        "**How reliable is this?**",
        "",
        f"{stats['clear_pct']}% of the area was free of cloud and shadow on both dates; only that part was measured. "
        f"A pixel counted as changed when an index moved by more than {threshold}. "
        "This is an automatic estimate from two satellite images, not a field survey. "
        "Thin lines along rivers and roads are often edge noise.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- one entry point
WRITERS = {"gemini": write_with_gemini, "claude": write_with_claude}


def write_report(stats: dict, context: dict, use_ai: bool = True, provider: str = config.REPORT_PROVIDER) -> Report:
    """Always returns a Report: from the AI if possible, otherwise the basic one (with a note why)."""
    if not use_ai:
        return Report(basic_report(stats, context), "basic (no AI)", "AI report turned off.")

    chosen = choose_provider(provider)
    if chosen is None:
        note = "No GEMINI_API_KEY or ANTHROPIC_API_KEY in .env, so the report was written without AI."
        return Report(basic_report(stats, context), "basic (no AI)", note)

    try:
        return Report(WRITERS[chosen](stats, context), MODELS[chosen])
    except ReportUnavailable as e:
        return Report(basic_report(stats, context), "basic (no AI)", f"{e} Showing the basic report instead.")
