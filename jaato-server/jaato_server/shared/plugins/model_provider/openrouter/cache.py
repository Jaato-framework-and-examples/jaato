"""OpenRouter prompt-caching helpers.

OpenRouter (https://openrouter.ai/docs/features/prompt-caching) exposes
prompt caching in two flavours depending on the upstream:

- **Automatic** (OpenAI, DeepSeek, Grok, Moonshot, ...): the upstream
  caches stable prefixes on its own; the client just sends standard
  OpenAI-shaped requests.  Savings are realised regardless of any
  client-side annotation.
- **Explicit** (Anthropic Claude, Google Gemini Pro/Flash): the client
  must stamp ``cache_control: {"type": "ephemeral"}`` breakpoints onto
  message / tool content blocks.  Without them, no caching occurs.

This module:
  1. Detects whether the active model needs explicit breakpoints
     (``model_supports_explicit_cache``).
  2. Resolves the ``cache_prompt`` profile knob — ``"auto"`` / True /
     False — into a definite on/off decision (``resolve_cache_active``).
  3. Builds the canonical breakpoint dict, with optional 1-hour TTL
     (``make_cache_control``).

The actual placement (system block, last tool, response usage parsing)
is in :mod:`converters` and :mod:`provider`.  Placement is hard-coded
to two breakpoints: end of system instruction + end of last tool
definition.  Both are stable across an agent's session and capture the
vast majority of the cacheable prefix.  A third history breakpoint —
analogous to BP3 in ``cache_anthropic`` — is intentionally out of scope
for this initial implementation; it requires budget-aware placement
that the OpenAI-shaped wire makes awkward.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Model-id substrings that indicate the upstream requires *explicit*
# ``cache_control`` breakpoints.  Models that fall outside this list
# either cache automatically (OpenAI, DeepSeek, Grok, ...) or don't
# support caching at all — either way, stamping breakpoints is a no-op
# at best and an error at worst, so we skip it.
#
# Match is case-insensitive substring on the OpenRouter model id (e.g.
# ``anthropic/claude-3.5-sonnet``).  Google's caching only kicks in on
# Gemini 1.5+ and 2.5+ models, so we match those families explicitly
# rather than a bare ``google/`` prefix.
EXPLICIT_CACHE_MODEL_HINTS = (
    "anthropic/",
    "google/gemini-1.5-",
    "google/gemini-2.5-",
    "google/gemini-3",
)


def model_supports_explicit_cache(model: Optional[str]) -> bool:
    """Return True if ``model`` is an upstream that needs explicit breakpoints.

    A return value of False means one of two things:
      - The upstream caches automatically (savings happen regardless).
      - The upstream doesn't support caching at all.

    Either way the request should be sent without ``cache_control``
    annotations.  Response-side parsing of ``prompt_tokens_details``
    /``cache_discount`` happens unconditionally.
    """
    if not model:
        return False
    lowered = model.lower()
    return any(hint in lowered for hint in EXPLICIT_CACHE_MODEL_HINTS)


def resolve_cache_active(setting: Any, model: Optional[str]) -> bool:
    """Resolve the ``cache_prompt`` knob against the active model.

    Setting semantics:
      - ``"auto"`` / None     → caching is active iff the model needs
        explicit breakpoints (``anthropic/*``, ``google/gemini-1.5-*``,
        ``google/gemini-2.5-*``, ``google/gemini-3*``).
      - truthy (True / "on" / "true" / "1") → force on regardless of
        model; useful when OpenRouter adds a new explicit-cache vendor
        before this module learns about it.
      - falsy (False / "off" / "false" / "0") → force off; the user
        knows their workload doesn't benefit (e.g. always-fresh prompts)
        and wants to skip the extra wire overhead.
    """
    if setting is None or (isinstance(setting, str) and setting.lower() == "auto"):
        return model_supports_explicit_cache(model)
    if isinstance(setting, str):
        lowered = setting.lower()
        if lowered in ("on", "true", "yes", "1"):
            return True
        if lowered in ("off", "false", "no", "0"):
            return False
    return bool(setting)


def make_cache_control(ttl: str = "5m") -> Dict[str, str]:
    """Build the OpenRouter / Anthropic ``cache_control`` dict.

    OpenRouter forwards the Anthropic shape verbatim:

        {"type": "ephemeral"}            # default 5-minute TTL
        {"type": "ephemeral", "ttl": "1h"}  # extended 1-hour TTL

    The 1-hour TTL doubles the cache-write premium (2x vs 1.25x) but
    avoids cache misses during long agentic sessions.  Anthropic also
    requires the ``extended-cache-ttl-2025-04-11`` beta header for the
    1-hour variant when going direct; via OpenRouter the gateway
    handles the beta dance, so the client just sets the ttl.
    """
    if ttl == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}
