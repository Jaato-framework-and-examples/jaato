"""Deterministic name-to-ID mapping for model-facing tool and category names.

Tools and categories are identified to the model by short hash-derived IDs
(e.g., ``t_a3f2b1c0`` for tools, ``c_7e4d9f12`` for categories) instead of
their human-readable names. This prevents the model from deriving semantic
meaning from the name string and forces it to rely on the description.

The mapping is a **pure function of the name** — the same tool always
produces the same ID regardless of session, plugin configuration, or load
order. This guarantees telemetry stability across deployments.

User-facing surfaces (TUI, events, logs, permissions) always show the
real name. The ID exists only at the provider boundary.

The full name is hashed — including any variant suffix like ``-stream``
(server 0.6.65+).  Earlier versions special-cased ``:stream`` and emitted
``t_<hash>:stream``, but the colon failed strict upstream tool-name regexes
(Anthropic / Bedrock / OpenAI all enforce ``^[a-zA-Z0-9_-]{1,128}$``); the
hyphenated suffix is universally regex-safe and the base/variant
relationship is now expressed via the registry's ``-stream`` naming
convention rather than via shared ID prefixes.
"""

import hashlib
import re
from typing import Any, Dict


_reverse: Dict[str, str] = {}

# A complete model-facing id: t_<8 hex> (tool) or c_<8 hex> (category), as a
# whole token (markdown backticks / spaces / punctuation are non-word, so the
# word boundaries match ``t_xxxxxxxx`` whether bare or wrapped).
_ID_RE = re.compile(r"\b[tc]_[0-9a-f]{8}\b")

# A trailing fragment that could be an INCOMPLETE id prefix at a stream-chunk
# boundary (``t`` / ``t_`` / ``t_<0-7 hex>``) — held back so a split id is never
# emitted half-scrubbed.  A COMPLETE id (8 hex) does NOT match (the {0,7} cap),
# so it is emitted and scrubbed rather than held.
_PARTIAL_TAIL_RE = re.compile(r"[tc](_[0-9a-f]{0,7})?$")


def scrub_tool_ids(text: str) -> str:
    """Replace known tool/category ids in free text with their real names.

    Unknown ids (never issued / hallucinated) pass through unchanged via
    :func:`id_to_name`, so coincidental ``t_########`` hex strings are safe.
    This is the user-facing-surface guard for the contract in this module's
    docstring: model NARRATION that mentions an id must show the real name.
    """
    return _ID_RE.sub(lambda m: id_to_name(m.group(0)), text)


class StreamScrubber:
    """Stateful :func:`scrub_tool_ids` for STREAMED model text.

    ``feed(chunk)`` scrubs complete ids and HOLDS a trailing incomplete-id
    fragment (so an id split across chunks — ``...t_3f2`` | ``9f306...`` — is not
    emitted half-way); ``flush()`` returns the held remainder at turn end.  For
    non-streaming text (the whole part in one ``feed``) the held fragment is
    normally empty.  Cost: a chunk ending in a lone ``t``/``c`` is delayed one
    chunk (it could begin an id) — cosmetic, and required for airtight scrubbing.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        s = self._buf + text
        m = _PARTIAL_TAIL_RE.search(s)
        hold = len(s) - m.start() if m else 0
        emit, self._buf = s[: len(s) - hold], s[len(s) - hold :]
        return scrub_tool_ids(emit)

    def flush(self) -> str:
        out = scrub_tool_ids(self._buf)
        self._buf = ""
        return out


def name_to_id(name: str, prefix: str = "t") -> str:
    """Derive a stable, opaque ID from a human-readable name.

    Uses the first 8 hex characters of the SHA-256 hash of the *full* name,
    giving ~4 billion possible values — collision probability is negligible
    for <10K names.

    Args:
        name: Human-readable tool or category name.
        prefix: ID prefix (``"t"`` for tools, ``"c"`` for categories).

    Returns:
        Stable ID string, e.g. ``"t_a3f2b1c0"``.  Always conforms to
        ``^[a-zA-Z0-9_-]{1,128}$`` so it passes every strict upstream
        tool-name validator.
    """
    h = hashlib.sha256(name.encode()).hexdigest()[:8]
    tool_id = f"{prefix}_{h}"
    _reverse[tool_id] = name
    return tool_id


def id_to_name(id_str: str) -> str:
    """Resolve an ID back to the original human-readable name.

    Returns the input unchanged if the ID is not recognized (e.g., when
    the model hallucinates a tool ID that was never issued).

    Args:
        id_str: The hash-derived ID from the model's response.

    Returns:
        Original name, or the input if not found.
    """
    return _reverse.get(id_str, id_str)


def tool_choice_to_wire(tool_choice: Any) -> Any:
    """Map a ``tool_choice`` that NAMES a tool to its wire id.

    Tool names are hashed to opaque ids on the wire (:func:`name_to_id`), so a
    ``tool_choice`` that forces a specific tool must reference the same wire id
    — otherwise the upstream rejects it ("Tool '<human>' not found in tools
    list").  This closes the boundary gap where the ``tools`` array was hashed
    but a name-bearing ``tool_choice`` was forwarded verbatim.

    Handles both wire shapes:

    - **OpenAI**: ``{"type": "function", "function": {"name": "<human>"}}``
    - **Anthropic**: ``{"type": "tool", "name": "<human>"}``

    String forms (``"auto"`` / ``"required"`` / ``"none"`` / ``"any"``) and any
    shape without a string tool name pass through unchanged.  Returns a NEW
    dict when remapping (never mutates the input).
    """
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            # OpenAI: {"type":"function","function":{"name": ...}}
            return {
                **tool_choice,
                "function": {**fn, "name": name_to_id(fn["name"])},
            }
        if isinstance(tool_choice.get("name"), str):
            # Anthropic: {"type":"tool","name": ...}
            return {**tool_choice, "name": name_to_id(tool_choice["name"])}
    return tool_choice
