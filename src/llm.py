"""
SF Crowd Voting Simulator — LLM provider (OpenRouter)
=====================================================
Central place for talking to the LLM. The app now calls models through
`OpenRouter <https://openrouter.ai>`_, which exposes an OpenAI-compatible
Chat Completions API at a single base URL. Point the OpenAI SDK at that base
URL with an ``OPENROUTER_API_KEY`` and pick any model in the registry below.

Why a shared module: run_scenario, generate_population (agent memory) and
analyze all need a client and a JSON-returning completion. Keeping the client
construction, the model registry, and the reasoning-effort call in one file
means the OpenRouter conventions live in exactly one place.

OpenRouter specifics (differ from a raw OpenAI o-series call):
- Client: ``OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)``
- Model ids are provider-namespaced slugs, e.g. ``openai/gpt-5.5``.
- Reasoning is a **unified** parameter that works across providers:
  ``extra_body={"reasoning": {"effort": "low"|"medium"|"high"}}``. OpenRouter
  maps the effort to each model's native reasoning budget, so the same call
  works whether the model is from OpenAI, Anthropic, Google, DeepSeek, or xAI.
  (This replaces the OpenAI-only ``reasoning_effort`` kwarg.)
- Role: we send a ``system`` message (universally supported) rather than the
  OpenAI-only ``developer`` role.
- ``response_format={"type": "json_object"}`` is honored by every model in the
  registry below; prompts also instruct "respond only with valid JSON" so
  parsing stays robust even if a model ignores the flag.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Optional attribution headers OpenRouter shows on its dashboard/leaderboards.
_APP_TITLE = "SF Crowd Voting Simulator"
_APP_REFERER = "https://github.com/crowd-sim"

# ---------------------------------------------------------------------------
# Model registry — curated reasoning models available on OpenRouter.
# Each entry: {"id": <OpenRouter slug>, "label": <shown in the UI dropdown>}.
# Ordered by roleplay/persona quality (best first) — this app asks each model to
# stay in character as an SF resident, so persona fidelity matters more than raw
# coding/benchmark scores. Edit freely; ids must be valid OpenRouter slugs.
# ---------------------------------------------------------------------------
MODELS = [
    {"id": "anthropic/claude-sonnet-5", "label": "Claude Sonnet 5"},
    {"id": "mistralai/mistral-medium-3-5", "label": "Mistral Medium 3.5"},
    {"id": "deepseek/deepseek-v4-flash", "label": "DeepSeek V4 Flash"},
]
MODEL_IDS = {m["id"] for m in MODELS}

# Default model — overridable via OPENROUTER_MODEL. Falls back to the first
# registry entry if the env var names something not in the registry.
_env_default = os.environ.get("OPENROUTER_MODEL", "").strip()
DEFAULT_MODEL = _env_default if _env_default in MODEL_IDS else MODELS[0]["id"]


def api_key() -> str | None:
    """Return the configured OpenRouter API key, or None."""
    return os.environ.get("OPENROUTER_API_KEY")


def resolve_model(model: str | None) -> str:
    """Return a valid registry model id, defaulting when unknown/blank."""
    if model and model in MODEL_IDS:
        return model
    return DEFAULT_MODEL


def build_client():
    """
    Build an OpenAI SDK client pointed at OpenRouter, or None when no key is set
    (callers treat None as "LLM unavailable" and fall back where they can).
    """
    key = api_key()
    if not key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers={
                "HTTP-Referer": _APP_REFERER,
                "X-Title": _APP_TITLE,
            },
        )
    except Exception:
        return None


def require_client():
    """Like build_client() but raises if OPENROUTER_API_KEY is missing/unusable."""
    if not api_key():
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Get a key at https://openrouter.ai/keys, then: "
            "export OPENROUTER_API_KEY=sk-or-..."
        )
    client = build_client()
    if client is None:
        raise EnvironmentError(
            "Failed to initialize the OpenRouter client. Is the 'openai' package installed?"
        )
    return client


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_text(content: str | None) -> str:
    """
    Pull a parseable JSON object out of a model's message content.

    Providers on OpenRouter are inconsistent even when asked for a JSON object:
    some return bare JSON, some wrap it in a ```json fence, and some prepend a
    sentence of prose. This normalizes all three to a bare ``{...}`` string.
    Raises ValueError when there is no JSON to parse (e.g. empty content, which
    reasoning models emit when the answer never left the reasoning channel).
    """
    if not content or not content.strip():
        raise ValueError("model returned empty content")
    text = _FENCE_RE.sub("", content.strip())
    # Fall back to the outermost brace span when prose brackets the JSON.
    if not text.lstrip().startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object found in content: {content[:120]!r}")
        text = text[start : end + 1]
    return text


def chat_json(client, model: str, system_prompt: str, user_prompt: str, effort: str = "medium") -> str:
    """
    One JSON-returning chat completion through OpenRouter.

    Sends a system + user message, requests a JSON object, and drives reasoning
    with the unified OpenRouter ``reasoning.effort`` control. Returns a bare
    JSON-object string — fences and surrounding prose are stripped, so callers
    can ``json.loads`` the result directly across every provider in the
    registry. Raises on API/network errors, or ValueError when the model
    returns nothing parseable (empty content, no JSON object).
    """
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        extra_body={"reasoning": {"effort": effort}},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    choice = response.choices[0]
    text = _extract_json_text(choice.message.content)
    # Validate here so a truncated/garbled object fails at the source with a
    # clear message (and the model's finish_reason) rather than deep in a caller.
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        # Some providers emit literal newlines/tabs inside string values (common
        # with multi-line narrative fields like delivery experiences). Strict
        # JSON forbids these; retry leniently and re-serialize so callers get
        # clean, strictly-parseable JSON back rather than the raw control chars.
        try:
            parsed = json.loads(text, strict=False)
        except json.JSONDecodeError:
            finish = getattr(choice, "finish_reason", None)
            raise ValueError(
                f"model returned invalid JSON (finish_reason={finish!r}): {exc}"
            ) from exc
        return json.dumps(parsed)
    return text
