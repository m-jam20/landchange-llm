"""Offline tests for report.py: fake clients stand in for Gemini and Claude (no internet, no key)."""

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from change_monitor import report
from change_monitor.report import (
    ReportUnavailable,
    basic_report,
    build_prompt,
    choose_provider,
    write_report,
    write_with_claude,
    write_with_gemini,
    write_with_openai_compatible,
)

STATS = {
    "clear_ha": 2490.61,
    "clear_pct": 98.6,
    "threshold": 0.2,
    "change_ha": {"water_gain": 1.17, "water_loss": 1.9, "new_built_up": 29.65,
                  "vegetation_loss": 19.14, "vegetation_gain": 23.04},
    "change_pct_of_clear": {"water_gain": 0.05, "water_loss": 0.08, "new_built_up": 1.19,
                            "vegetation_loss": 0.77, "vegetation_gain": 0.93},
}
CONTEXT = {"place": "Test town", "before_image_date": "2023-06-05", "after_image_date": "2025-07-01"}


@pytest.fixture
def no_keys(monkeypatch):
    """Pretend no API keys exist (and ignore any real .env file)."""
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(report, "load_dotenv", lambda: None)


# ---------------------------------------------------------------- fakes
class FakeGemini:
    """Looks like google.genai.Client() just enough, and records the request."""

    def __init__(self, text="# Report\nAbout 30 ha became built-up.", error=None):
        self.request = None

        def generate_content(**kwargs):
            self.request = kwargs
            if error:
                raise error
            return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason="STOP")])

        self.models = SimpleNamespace(generate_content=generate_content)


class FakeClaude:
    """Looks like anthropic.Anthropic() just enough, and records the request."""

    def __init__(self, text="# Report\nAbout 30 ha became built-up.", stop_reason="end_turn"):
        self.request = None
        reply = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        )

        def create(**kwargs):
            self.request = kwargs
            return reply

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


# ---------------------------------------------------------------- prompt
def test_prompt_contains_the_numbers_and_readable_names():
    prompt = build_prompt(STATS, CONTEXT)
    assert "29.65" in prompt and "Test town" in prompt
    assert "New built-up / bare ground" in prompt


# ---------------------------------------------------------------- Gemini
def test_gemini_returns_text_and_sends_our_instructions():
    client = FakeGemini()
    assert write_with_gemini(STATS, CONTEXT, client=client) == "# Report\nAbout 30 ha became built-up."
    assert client.request["config"].system_instruction == report.SYSTEM_PROMPT
    assert "29.65" in client.request["contents"]


def test_gemini_free_limit_gives_a_clear_message():
    limit = genai_errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted"}})
    with pytest.raises(ReportUnavailable, match="free-tier limit"):
        write_with_gemini(STATS, CONTEXT, client=FakeGemini(error=limit))


def test_gemini_bad_key_gives_a_clear_message():
    bad = genai_errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid."}})
    with pytest.raises(ReportUnavailable, match="key was rejected"):
        write_with_gemini(STATS, CONTEXT, client=FakeGemini(error=bad))


def test_gemini_network_problem_is_caught():
    with pytest.raises(ReportUnavailable, match="Could not reach Gemini"):
        write_with_gemini(STATS, CONTEXT, client=FakeGemini(error=ConnectionError("offline")))


def test_gemini_empty_answer_is_reported():
    with pytest.raises(ReportUnavailable, match="no text"):
        write_with_gemini(STATS, CONTEXT, client=FakeGemini(text=""))


# ---------------------------------------------------------------- Claude
def test_claude_returns_text_and_ignores_thinking():
    client = FakeClaude()
    assert write_with_claude(STATS, CONTEXT, client=client) == "# Report\nAbout 30 ha became built-up."
    assert client.request["system"] == report.SYSTEM_PROMPT


def test_claude_refusal_is_reported():
    with pytest.raises(ReportUnavailable, match="declined"):
        write_with_claude(STATS, CONTEXT, client=FakeClaude(stop_reason="refusal"))


# ---------------------------------------------------------------- basic report
def test_basic_report_has_numbers_largest_first_and_flags_noise():
    text = basic_report(STATS, CONTEXT)
    assert text.startswith("# Land change in Test town: 2023-06-05 to 2025-07-01")
    assert text.index("New built-up") < text.index("Vegetation gain") < text.index("Water loss")
    assert "Water gain: 1.2 ha (0.05%) (minor, may be noise)" in text
    assert "98.6%" in text


def test_basic_report_is_deterministic():
    assert basic_report(STATS, CONTEXT) == basic_report(STATS, CONTEXT)


# ---------------------------------------------------------------- choosing who writes
def test_no_keys_means_basic_report_with_a_note(no_keys):
    result = write_report(STATS, CONTEXT)
    assert result.source == "basic (no AI)"
    assert "GEMINI_API_KEY" in result.note


def test_gemini_is_preferred_when_both_keys_exist(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    assert choose_provider("auto") == "gemini"


def test_claude_is_used_when_only_its_key_exists(no_keys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    assert choose_provider("auto") == "claude"


def test_ai_failure_falls_back_to_basic_report(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    def broken(stats, context):
        raise ReportUnavailable("Gemini free-tier limit reached.")

    monkeypatch.setitem(report.WRITERS, "gemini", broken)
    result = write_report(STATS, CONTEXT)
    assert result.source == "basic (no AI)"
    assert "free-tier limit" in result.note
    assert "New built-up" in result.text


def test_ai_success_is_labelled_with_the_model(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setitem(report.WRITERS, "gemini", lambda stats, context: "# AI text")
    result = write_report(STATS, CONTEXT)
    assert result.text == "# AI text" and result.note is None
    assert result.source == f"Google Gemini · {report.config.GEMINI_MODEL}"


def test_gemini_is_asked_to_think_little_for_speed():
    client = FakeGemini()
    write_with_gemini(STATS, CONTEXT, client=client)
    assert client.request["config"].thinking_config.thinking_level.name == report.config.GEMINI_THINKING_LEVEL.upper()


def test_gemini_timeout_gives_a_clear_message():
    class ReadTimeout(Exception):  # same name as the HTTP library's timeout error
        pass

    with pytest.raises(ReportUnavailable, match="did not answer within"):
        write_with_gemini(STATS, CONTEXT, client=FakeGemini(error=ReadTimeout()))


def test_prompt_separates_visible_from_unchanged():
    """The clear % (visible) and the unchanged % must both be given, clearly named and different."""
    prompt = build_prompt(STATS, CONTEXT)
    assert '"pct_of_area_visible_on_both_dates": 98.6' in prompt
    assert '"pct_of_visible_area_unchanged": 97.0' in prompt  # (2490.61 - 74.9) / 2490.61


# ---------------------------------------------------------------- any OpenAI-compatible company
class FakeOpenAI:
    """Looks like openai.OpenAI() just enough, and records the request."""

    def __init__(self, text="# Report\nAbout 30 ha.", error=None, finish_reason="stop"):
        self.request = None

        def create(**kwargs):
            self.request = kwargs
            if error:
                raise error
            message = SimpleNamespace(content=text)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def openai_error(cls, status):
    """A real openai error object, built the way the library builds it."""
    import httpx2 as httpx  # the HTTP library the openai SDK (3.x) is built on
    import openai

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    if cls in (openai.APIConnectionError, openai.APITimeoutError):
        return cls(request=request)
    return cls("error", response=httpx.Response(status, request=request), body=None)


def test_other_ai_sends_system_and_user_messages(no_keys, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-model")
    client = FakeOpenAI()
    assert write_with_openai_compatible(STATS, CONTEXT, client=client) == "# Report\nAbout 30 ha."
    roles = [m["role"] for m in client.request["messages"]]
    assert roles == ["system", "user"] and client.request["model"] == "some-model"
    assert "29.65" in client.request["messages"][1]["content"]


@pytest.mark.parametrize(
    ("error_class", "status", "message"),
    [
        ("AuthenticationError", 401, "key was rejected"),
        ("NotFoundError", 404, "does not know the model"),
        ("RateLimitError", 429, "rate limit"),
        ("APITimeoutError", None, "did not answer within"),
        ("APIConnectionError", None, "Could not reach"),
    ],
)
def test_other_ai_errors_give_clear_messages(no_keys, monkeypatch, error_class, status, message):
    import openai

    monkeypatch.setenv("LLM_MODEL", "some-model")
    error = openai_error(getattr(openai, error_class), status)
    with pytest.raises(ReportUnavailable, match=message):
        write_with_openai_compatible(STATS, CONTEXT, client=FakeOpenAI(error=error))


def test_other_ai_empty_answer_is_reported(no_keys, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-model")
    with pytest.raises(ReportUnavailable, match="no text"):
        write_with_openai_compatible(STATS, CONTEXT, client=FakeOpenAI(text=None, finish_reason="length"))


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "OpenAI"),
        ("https://api.mistral.ai/v1", "mistral"),
        ("https://api.groq.com/openai/v1", "groq"),
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("http://localhost:11434/v1", "local model"),
        ("https://generativelanguage.googleapis.com/v1beta/openai/", "googleapis"),
    ],
)
def test_company_name_comes_from_the_address(no_keys, monkeypatch, base_url, expected):
    if base_url:
        monkeypatch.setenv("LLM_BASE_URL", base_url)
    assert report.other_ai_name() == expected


def test_other_ai_needs_a_model_and_a_key_or_address(no_keys, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake")
    assert not report.other_ai_configured()  # no model yet
    monkeypatch.setenv("LLM_MODEL", "some-model")
    assert report.other_ai_configured()


def test_local_ollama_needs_no_key(no_keys, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "some-local-model")
    assert report.available_providers() == ["openai_compatible"]
    assert report.describe_provider("openai_compatible") == "local model (OpenAI-compatible) · some-local-model"


def test_all_set_up_providers_are_listed_in_order(no_keys, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    assert report.available_providers() == ["gemini", "openai_compatible"]
    assert choose_provider("openai_compatible") == "openai_compatible"
    assert choose_provider("claude") is None  # asked for, but not set up
