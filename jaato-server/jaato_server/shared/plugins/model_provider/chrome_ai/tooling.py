"""Compatibility shim — the tool-call machinery moved to ``_prose_tools``.

The text-encoded tool protocol born with this provider (prompt-injected
schemas + fenced ``tool_call`` parsing) was hoisted to
``shared/plugins/model_provider/_prose_tools.py`` so OpenAI-compat
providers can reuse it behind the opt-in ``prose_tool_calls`` model
quirk.  This module re-exports the parser under its historical location
for the provider and its tests; new code should import from
``_prose_tools`` directly.
"""

from .._prose_tools import parse_tool_calls

__all__ = ["parse_tool_calls"]
