"""Validation helper for ``subagent_id`` in confused-deputy contexts.

Phase 4 §4.3.5 hoists this out of ``AppArmorManager`` so the sub-
AppArmor profile (§4.3.4) and sub-cgroup (§4.3.5) validators share
a single implementation.  Divergence between the two would be a
latent confused-deputy bug — both surfaces accept the same
runner-supplied id and feed it into kernel-visible names.

When ``subagent_id`` flows from runner-side caller into kernel-
visible identifiers (AppArmor profile names, cgroup directory
names), strict allow-list validation is required to prevent:

- Profile-file injection (newlines, quote escaping).
- Filesystem path traversal (``..``, ``/``).
- DoS via giant filenames.
- Collision with internal prefixes / separators (``//``,
  ``__sub_``).

Strict allow-list ``[A-Za-z0-9_-]`` with length cap 64.  REJECT on
violation (no collapse-to-underscore — that masks supervisor bugs
rather than surfacing them).
"""

from __future__ import annotations

import re
from typing import Any, Optional


# Length cap.  Keeps filenames readable + profile names below typical
# kernel buffer limits + bounds injection scope.
SUBAGENT_ID_MAX_LEN = 64


# Strict allow-list — only alphanumerics, hyphens, underscores.  Reject
# (not collapse) on violation per Audit 6 in
# docs/design/phase4_implementation_audits.md.
_SUBAGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_subagent_id(subagent_id: Any) -> Optional[str]:
    """Validate ``subagent_id`` for confused-deputy contexts.

    Phase 4 §4.3.4 introduced this for sub-AppArmor profile names;
    §4.3.5 hoisted it to a module-level helper for re-use across
    sub-cgroup directory names + future supervisor-side surfaces.

    Args:
        subagent_id: The candidate id.  Should be a ``str`` matching
            ``[A-Za-z0-9_-]+`` with length 1..64.

    Returns:
        ``None`` if valid; an error message string if invalid.
        Returning a string lets callers surface a precise diagnostic
        to the runner-side caller via the stage envelope instead of
        crashing.
    """
    if not isinstance(subagent_id, str):
        return (
            f"subagent_id must be a str, got "
            f"{type(subagent_id).__name__}"
        )
    if not subagent_id:
        return "subagent_id is empty"
    if len(subagent_id) > SUBAGENT_ID_MAX_LEN:
        return (
            f"subagent_id length {len(subagent_id)} exceeds cap "
            f"of {SUBAGENT_ID_MAX_LEN}"
        )
    if not _SUBAGENT_ID_PATTERN.match(subagent_id):
        return (
            f"subagent_id {subagent_id!r} contains characters "
            f"outside [A-Za-z0-9_-]"
        )
    return None
