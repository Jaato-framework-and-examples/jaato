"""Which MCP tool input schemas a provider refuses at the top level (#1631).

An MCP server authors its own tools' ``inputSchema``.  Any valid JSON Schema
is valid MCP, but a function-calling API accepts a narrower shape for the
TOP level of a tool's parameters: OpenAI answers
``Invalid schema for function ...: schema must have type 'object' and not
have 'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the top level``, and
Anthropic refuses ``oneOf``/``anyOf``/``allOf`` at the top level of
``input_schema`` too.

Every tool's schema is sent on every request, so one such tool fails EVERY
turn, including turns that never choose it -- the agent dies before its
first word.  :func:`top_level_schema_refusal` names the shape; the plugin
drops the tool from what it offers and announces it at WARNING, the bargain
it already makes for a server it cannot connect.

The list is conservative and provider-agnostic: only shapes a strict
validator rejects at the TOP level.  Nested combinators are left alone (they
are accepted inside ``properties``), and a schema with no ``type`` at all is
left alone (MCP requires ``object``, and the providers seen in practice
accept a ``properties``-only schema).
"""

from typing import Any, Optional

#: Keywords a strict function-schema validator refuses at the top level.
TOP_LEVEL_REFUSED_KEYWORDS = ("oneOf", "anyOf", "allOf", "enum", "const", "not")


def top_level_schema_refusal(schema: Any) -> Optional[str]:
    """Return why a provider would refuse ``schema`` as tool parameters.

    ``None`` means the top level is acceptable (or absent: a tool with no
    input schema takes no arguments and is sent as such).  A string names the
    offending shape in words fit for a log line.
    """
    if schema is None:
        return None
    if not isinstance(schema, dict):
        return f"input schema is a {type(schema).__name__}, not an object schema"
    present = [k for k in TOP_LEVEL_REFUSED_KEYWORDS if k in schema]
    if present:
        return (
            "input schema has " + "/".join(f"'{k}'" for k in present)
            + " at the top level"
        )
    declared = schema.get("type")
    if declared is not None and declared != "object":
        return f"input schema has top-level type {declared!r}, not 'object'"
    return None
