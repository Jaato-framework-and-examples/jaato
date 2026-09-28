"""Structured notices an enrichment plugin attaches to a tool result, for the CLIENT (1.31).

Tool-result enrichment rewrites what the MODEL reads.  Before this seam,
nothing a plugin found reached a client except the formatted
``source="enrichment"`` line, which has no fields a client can key on.  So
a client that wanted to act on the finding (the web coder offering to bind
a missing toolchain) had to detect the same thing a second time, and the
two detections could disagree.

A plugin now returns, beside its rewritten result::

    ToolResultEnrichmentResult(result=..., metadata={
        "client_notice": {"kind": "toolchain_offer", "data": {...}},
    })

(or a list of such dicts), and the session emits one
``ToolResultEnrichedEvent`` per valid notice once the result is built.  The
framework does not interpret ``kind`` or ``data``; this module only decides
what is well-formed:

- ``kind`` matches ``[a-z][a-z0-9_]{0,63}``;
- ``data`` is a dict that serialises as JSON in at most
  :data:`MAX_NOTICE_BYTES` bytes.

Anything else is dropped with a WARNING naming the plugin, never passed on
half-checked: the event is a wire contract, and a plugin's mistake must not
become a client's parsing problem.  Stdlib only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: The enrichment-metadata key a plugin puts its notice under.
NOTICE_KEY = "client_notice"

#: The most bytes one notice's ``data`` may serialise to.
MAX_NOTICE_BYTES = 4096

_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def client_notices(combined_metadata: Optional[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """The valid notices in a result's enrichment metadata, as ``(plugin, kind, data)``.

    ``combined_metadata`` is what ``PluginRegistry.enrich_tool_result``
    returns: ``{plugin_name: that plugin's metadata}``.  Order follows the
    enrichment chain, then each plugin's own list.
    """
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for plugin, meta in (combined_metadata or {}).items():
        if not isinstance(meta, dict) or NOTICE_KEY not in meta:
            continue
        raw = meta[NOTICE_KEY]
        for notice in raw if isinstance(raw, list) else [raw]:
            parsed = _parse(str(plugin), notice)
            if parsed is not None:
                out.append((str(plugin), *parsed))
    return out


def _parse(plugin: str, notice: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not isinstance(notice, dict):
        logger.warning("enrichment plugin %r: client_notice is not a mapping; dropped", plugin)
        return None
    kind = notice.get("kind")
    data = notice.get("data", {})
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        logger.warning("enrichment plugin %r: client_notice kind %r is not [a-z][a-z0-9_]*; dropped", plugin, kind)
        return None
    if not isinstance(data, dict):
        logger.warning("enrichment plugin %r: client_notice %s data is not a mapping; dropped", plugin, kind)
        return None
    try:
        size = len(json.dumps(data, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError):
        logger.warning("enrichment plugin %r: client_notice %s data is not JSON; dropped", plugin, kind)
        return None
    if size > MAX_NOTICE_BYTES:
        logger.warning("enrichment plugin %r: client_notice %s data is %d bytes (max %d); dropped", plugin, kind, size, MAX_NOTICE_BYTES)
        return None
    return kind, data
