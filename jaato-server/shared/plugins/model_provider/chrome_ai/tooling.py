"""Tool-call parsing for the Chrome built-in AI provider.

The Prompt API's native ``tools`` option is not available on stable Chrome
(mid-2026), and its execute-callback design runs tools inside the page —
the opposite of jaato's model, where the framework executes tools between
``complete()`` turns.  So, like ``claude_cli``'s passthrough mode, tool
calling is text-encoded: schemas are rendered into the system prompt
(``converters.tool_schemas_to_prompt``, with names hashed to opaque wire
ids per the framework contract) and the model's ``tool_call`` fenced
blocks are parsed back out here.

Small on-device models frequently hallucinate tool ids (both known
community bridges report this for names).  Parsed calls are emitted
regardless — after ``id_to_name`` mapping, an unknown id passes through
unchanged and jaato's executor returns a structured unknown-tool error
the model can recover from, a better corrective signal than silently
discarding the attempt.
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from .converters import TOOL_CALL_FENCE

logger = logging.getLogger(__name__)

#: Matches one ``tool_call`` fenced block; group 1 is the JSON body.
_FENCE_RE = re.compile(
    r"```" + TOOL_CALL_FENCE + r"[ \t]*\n(.*?)\n?```",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
    """Extract ``tool_call`` fenced blocks from model output.

    Returns ``(clean_text, calls)`` where ``clean_text`` is the output
    with successfully-parsed blocks removed, and ``calls`` is a list of
    ``(name, arguments)`` tuples in emission order.  ``name`` is whatever
    the model emitted — normally a hashed wire id; the provider maps it
    back to the human name via ``tool_id_map.id_to_name``.

    A block whose body is a JSON array yields one call per element.
    Malformed blocks (invalid JSON, or no ``name`` key) are left in the
    text verbatim — the model's words reach the user unaltered and the
    failure is visible instead of vanishing — and logged at WARNING.
    """
    calls: List[Tuple[str, Dict[str, Any]]] = []

    def _consume(match: "re.Match[str]") -> str:
        body = match.group(1).strip()
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            logger.warning("chrome_ai: unparseable tool_call block left in text: %r",
                           body[:200])
            return match.group(0)
        entries = parsed if isinstance(parsed, list) else [parsed]
        parsed_calls: List[Tuple[str, Dict[str, Any]]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("name"):
                logger.warning("chrome_ai: tool_call entry missing 'name', "
                               "left in text: %r", str(entry)[:200])
                return match.group(0)
            args = entry.get("arguments", entry.get("args", {}))
            if not isinstance(args, dict):
                args = {"value": args}
            parsed_calls.append((str(entry["name"]), args))
        calls.extend(parsed_calls)
        return ""

    clean = _FENCE_RE.sub(_consume, text)
    # Collapse the whitespace holes left by removed blocks.
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, calls
