"""Filesystem paths contributed by a ``jaato.premium`` entry point.

Stdlib-only, with no jaato imports, so the modules that need a premium path
while resolving CONFIGURATION (``plugins.subagent.config`` discovering
profiles, ``scaffold.explain`` listing instruction layers,
``server.apparmor`` granting a read) can ask without importing
``jaato_runtime``.  That module was the previous home of the function, and
its import graph is the whole runtime: provider loading, token accounting
(``google.api_core``), telemetry.  A caller that only wanted a directory
name paid for all of it (#1267).

``jaato_runtime`` re-exports both names, so existing imports and the tests
that monkeypatch ``jaato_runtime._get_premium_content_path`` keep working.
The cache is the same dict object in both places.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Cache for premium content paths — resolved once per entry-point name.
_premium_content_cache: Dict[str, Optional[str]] = {}


def _get_premium_content_path(name: str) -> Optional[str]:
    """Return a filesystem path provided by a ``jaato.premium`` entry point.

    Premium content entry points (``instructions``, ``profiles``, etc.)
    return a directory path where the premium package stores its content
    files.  Results are cached for the lifetime of the process.

    Args:
        name: The entry-point name within the ``jaato.premium`` group
            (e.g. ``"instructions"``, ``"profiles"``).

    Returns:
        Absolute path string, or ``None`` if no provider is registered.
    """
    if name in _premium_content_cache:
        return _premium_content_cache[name]

    result = None
    matches = importlib.metadata.entry_points().select(
        group="jaato.premium", name=name)

    for ep in matches:
        try:
            provider_fn = ep.load()
            result = provider_fn()
            break
        except Exception:
            logger.warning("Failed to load premium content path '%s'", name, exc_info=True)

    _premium_content_cache[name] = result
    return result
