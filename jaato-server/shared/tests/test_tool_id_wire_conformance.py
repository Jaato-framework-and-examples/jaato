"""Guard: no UNHASHED tool name ever reaches the LLM, in any provider.

Tool names are hashed to opaque ids at the provider boundary (``name_to_id``)
so the model can't derive meaning from the name string.  Two surfaces carry a
tool name to the wire:

1. the ``tools`` array (every tool's ``name``) — the primary surface;
2. ``tool_choice`` (forcing a specific tool) — the surface the verbatim-forward
   bug landed on.

**Surface 1 is guarded behaviorally** (the strong check): each provider's
tool-schema converter is run with a sentinel name and the serialized wire is
scanned — the human name MUST be absent and its hash present.  Dep-safe: the
converter files use only ``jaato_sdk``/``shared`` imports, so they
``importlib``-load without the vendor SDK (same technique as
``test_provider_capability_conformance``).

**Surface 2 is guarded structurally**: every provider that forwards a
``tool_choice`` must route it through ``tool_choice_to_wire``.  The providers
not yet migrated are pinned in ``_TOOL_CHOICE_MAPPING_PENDING`` (the tools
array is already hashed for them; only the narrower tool_choice forward is
pending — closed per-provider, the rest folding into the shared OpenAI-compat
base refactor).  The allowlist can only SHRINK.
"""

import json

import pytest

from jaato_sdk.plugins.model_provider.types import ToolSchema
from shared.tool_id_map import name_to_id

# Reuse the capability-conformance infra (provider->converter file map + the
# dep-safe importlib loader) so the wire guards share one source of truth.
from shared.tests.test_provider_capability_conformance import (  # noqa: E402
    PROVIDER_DIR,
    _CONVERTERS,
    _load_converter,
)
from shared.tests.test_provider_capabilities import _provider_dirs  # noqa: E402

_SENTINEL = "WIRE_LEAK_SENTINEL_humanToolName_42"


def _tool_schema_fn(message_fn: str) -> str:
    """The tool-schema converter fn that pairs with a message converter."""
    return {
        "message_to_anthropic": "tool_schemas_to_anthropic",
        # chrome_ai has no wire tools array — its "tools surface" is the
        # prompt-injected section, rendered (and hashed) by this fn.
        "message_to_prompt_api": "tool_schemas_to_prompt",
    }.get(message_fn, "tool_schemas_to_openai")


# ----------------------------------------------- surface 1: tools array (behavioral)

@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_tools_array_never_leaks_human_tool_names(provider):
    """A sentinel tool name must NOT appear on the wire — only its hash."""
    relpath, message_fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, _tool_schema_fn(message_fn))
    wire = convert([ToolSchema(
        name=_SENTINEL, description="sentinel",
        parameters={"type": "object", "properties": {}})])
    blob = json.dumps(wire, default=str)
    assert _SENTINEL not in blob, (
        f"{provider}: the human tool name {_SENTINEL!r} leaked to the wire — "
        "the tool-schema converter must hash names via name_to_id."
    )
    assert name_to_id(_SENTINEL) in blob, (
        f"{provider}: the hashed tool id is missing from the wire."
    )


# --------------------------------------------- surface 2: tool_choice (structural)

# Providers whose provider.py forwards a tool_choice to the wire.
_TOOL_CHOICE_FORWARDERS = {"nebius", "openrouter", "vllm", "tensorrt_llm",
                           "triton", "anthropic"}

# Of those, the ones not YET routing through tool_choice_to_wire.  nebius +
# tensorrt_llm are migrated onto the OpenAICompat base (which maps); the rest
# fold in as they migrate.  SHRINK-ONLY.
_TOOL_CHOICE_MAPPING_PENDING = {"anthropic"}


def _provider_src(provider: str) -> str:
    p = PROVIDER_DIR / provider / "provider.py"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _maps_tool_choice(provider: str) -> bool:
    """Does this forwarder route tool_choice through the wire-id mapper?

    True if it calls ``tool_choice_to_wire`` directly, OR subclasses
    ``OpenAICompatProvider`` — whose shared ``_apply_api_params`` maps it for
    the whole fleet.  As providers migrate onto the base, the inheritance arm
    is what lets them leave ``_TOOL_CHOICE_MAPPING_PENDING``.
    """
    src = _provider_src(provider)
    return (
        "tool_choice_to_wire" in src
        or "OpenAICompatProvider" in src
        or "OpenAICompatLocalHostProvider" in src   # IS-A OpenAICompatProvider
    )


def test_tool_choice_forwarders_route_through_the_mapper():
    """Every tool_choice forwarder maps the name (unless explicitly pending)."""
    offenders = [
        p for p in sorted(_TOOL_CHOICE_FORWARDERS)
        if not _maps_tool_choice(p)
        and p not in _TOOL_CHOICE_MAPPING_PENDING
    ]
    assert offenders == [], (
        f"providers forward a name-bearing tool_choice without mapping it "
        f"through tool_choice_to_wire (directly or via OpenAICompatProvider): "
        f"{offenders}.  A tool_choice naming a tool must use the hashed wire "
        "id, or the upstream rejects it."
    )


def test_tool_choice_pending_list_only_shrinks():
    """A provider that now maps tool_choice must be removed from PENDING."""
    stale = [p for p in sorted(_TOOL_CHOICE_MAPPING_PENDING)
             if _maps_tool_choice(p)]
    assert stale == [], (
        f"these providers now route tool_choice through the mapper — drop them "
        f"from _TOOL_CHOICE_MAPPING_PENDING: {stale}"
    )


def test_no_unlisted_tool_choice_forwarder():
    """A NEW provider that forwards tool_choice must be added to the list.

    Soft signal (a ``["tool_choice"]`` wire-kwargs assignment); forces a
    conscious decision to map the name rather than silently leaking it.
    """
    unlisted = [
        p for p in _provider_dirs()
        if p not in _TOOL_CHOICE_FORWARDERS
        and '"tool_choice"]' in _provider_src(p)
    ]
    assert unlisted == [], (
        f"new tool_choice forwarder(s) {unlisted} — add to _TOOL_CHOICE_FORWARDERS "
        "and route through tool_choice_to_wire (or add to PENDING)."
    )
