"""Path-redaction pass for tool-result error tracebacks (§3.1).

Cross-tenant info-leak mitigation per parent design §4.8: when a
runner-side executor raises, the captured traceback may carry
absolute filesystem paths to the workspace_root or to the operator's
``~/.jaato/`` directory.  If that traceback is forwarded into a
daemon log shared across tenants, or surfaced in an event delivered
to a different operator, those paths leak.

This module provides :func:`sanitize_traceback`, a single-pass regex
filter that replaces:

1. ``<workspace_root>/...`` → ``<WORKSPACE>/...`` — tenant-specific.
2. ``<HOME>/.jaato/...`` → ``<HOME>/.jaato/...`` — operator-side.

Order is load-bearing: workspace_root may itself live under
``~/.jaato/`` (e.g. ``~/.jaato/sandbox/workspace-A/``), so the
workspace-pass must run FIRST so the more-specific path wins.

The regex uses ``(?<!\\w)`` lookbehind so ordinary words containing
``/home`` as a substring (or any other token-internal path-like
match) don't get false-positive-replaced.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


def sanitize_traceback(
    text: str,
    workspace_root: Optional[str] = None,
    *,
    home: Optional[str] = None,
) -> str:
    """Redact tenant + operator absolute paths from *text*.

    Args:
        text: The raw traceback (or any string carrying paths).
        workspace_root: Per-session workspace root.  Replaced with
            ``<WORKSPACE>``.  Pass ``None`` to skip the
            workspace-pass (e.g. when the runner has no workspace
            attached).
        home: Override the operator's home directory; defaults to
            ``Path.home()``.  Test injection point.

    Returns:
        The redacted text.  When *text* is empty or both replacement
        targets resolve to empty, returns *text* unchanged.
    """
    if not text:
        return text

    out = text

    # Pass 1: workspace_root.  Runs FIRST so a workspace under
    # ~/.jaato/ doesn't get prematurely caught by Pass 2.
    if workspace_root:
        ws = os.path.realpath(os.path.abspath(workspace_root))
        if ws and ws != "/":
            # Match the workspace path followed by either ``/`` or
            # end-of-token (so ``<WORKSPACE>`` shows up cleanly even
            # for the root itself).  ``(?<!\w)`` prevents matches
            # inside words like ``/var/cachehome/usr/...``.
            pattern = re.compile(
                r"(?<!\w)" + re.escape(ws) + r"(?=/|$|[^\w])"
            )
            out = pattern.sub("<WORKSPACE>", out)

    # Pass 2: ~/.jaato/.  Operator-side path.
    home_path = home if home is not None else str(Path.home())
    if home_path and home_path != "/":
        jaato_root = os.path.realpath(
            os.path.abspath(os.path.join(home_path, ".jaato"))
        )
        if jaato_root and jaato_root != "/":
            pattern = re.compile(
                r"(?<!\w)" + re.escape(jaato_root) + r"(?=/|$|[^\w])"
            )
            out = pattern.sub("<HOME>/.jaato", out)

    return out
