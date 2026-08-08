"""Report active ``JAATO_*`` env overrides — the live-state complement to
``jaato-scaffold explain env``.

The framework's operator-facing knobs live in the ``JAATO_`` namespace and are
absent from the environment unless explicitly set, so scanning for set
``JAATO_*`` vars is exactly the "what have I overridden from the defaults?"
view.  Surfaced at daemon startup (so you can't forget what you turned on) and
via ``--status`` (on-demand, reading the running daemon's own environment).

The *full* catalog of every env var the framework reads (set or not, with
defaults) lives in ``jaato-scaffold explain env``; this module is intentionally
the small live-state view, not a re-implementation of that catalog.

Secret-looking values are redacted by name so a startup log / status dump never
prints a token or key.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Redact the VALUE when the NAME looks like a credential (mirrors the
# secret_scrub name patterns).  Applies to e.g. JAATO_WS_TOKEN,
# JAATO_<provider>_API_KEY.
_SECRET_NAME_RE = re.compile(
    r"(TOKEN|_KEY\b|APIKEY|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE)


def _redact(name: str, value: str) -> str:
    return "<set>" if _SECRET_NAME_RE.search(name) else value


def active_overrides(environ: Dict[str, str]) -> List[Tuple[str, str]]:
    """Sorted ``(name, display_value)`` for set ``JAATO_*`` vars; secrets
    redacted to ``<set>``."""
    return [
        (k, _redact(k, environ[k]))
        for k in sorted(environ)
        if k.startswith("JAATO_")
    ]


def format_overrides(environ: Dict[str, str]) -> str:
    """One-line summary of active ``JAATO_*`` overrides for a log/stdout line."""
    items = active_overrides(environ)
    if not items:
        return ("Active JAATO_* env overrides: none (all defaults; see "
                "`jaato-scaffold explain env` for the full catalog)")
    body = "  ".join(f"{k}={v}" for k, v in items)
    return f"Active JAATO_* env overrides: {body}"


def read_proc_environ(pid: int) -> Dict[str, str]:
    """Best-effort read of ``/proc/<pid>/environ`` into a dict.

    Returns ``{}`` on any error (wrong platform, cross-user perms, dead pid) —
    the caller degrades to "unavailable" rather than failing.
    """
    out: Dict[str, str] = {}
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return out
    for chunk in raw.split(b"\x00"):
        if b"=" in chunk:
            key, _, value = chunk.partition(b"=")
            out[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return out
