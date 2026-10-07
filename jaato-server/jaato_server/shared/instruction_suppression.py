"""Canonical representation for ``suppress_base_instructions``.

Historically ``suppress_base_instructions`` was a plain ``bool`` that dropped
only the disk BASE layer (``.jaato/instructions/*.md``) from the assembled
system prompt, while the framework prompt constants (task-completion /
verification, parallel / batching, turn-summary / SUMMARY-FORMAT — plus their
jaato-premium overrides) and the untrusted-content security boundary were
appended UNCONDITIONALLY (``jaato_runtime.get_system_instructions`` steps 6-7).

That was surprising: profiles set ``suppress_base_instructions: true`` intending
to drop *everything the framework injects*, but ~1000 tokens of constants
survived — a real problem on tiny-context models (Gemini Nano ~9k), where the
SUMMARY-FORMAT block even made the model pad every reply with
``**Actions**: None. **Findings**: None. …``.

This module makes the knob granular. It accepts:

* ``bool`` — ``True`` suppresses ``{disk, constants}`` (the security boundary is
  KEPT: it is the indirect-prompt-injection defense for web_fetch / MCP tool
  results, and silently dropping it on a convenience flag is a footgun; the
  ``disclosure`` piece is kept for the same reason -- it is the Article 50(1)
  "you are talking to an AI" posture). ``False`` suppresses nothing.
* a mapping over the piece names (``{"disk": true, "constants": true,
  "security": false}``) — absent key = keep that piece.
* a list/set of piece names to suppress (wire form), or the token ``"all"``.

The canonical INTERNAL form is a ``frozenset`` of the suppressed piece names.
An empty set means "suppress nothing", so existing truthiness checks
(``if suppress:`` / ``not suppress``) keep working unchanged.

No hardcoded fallback: ``True``/``"all"`` map to explicit, enumerated member
sets defined here; every other value is normalized deterministically or rejected
loudly.
"""
from __future__ import annotations

from typing import Any, FrozenSet, List, Mapping

# The framework-originated instruction pieces a session can suppress.
PIECE_DISK = "disk"            # .jaato/instructions/*.md base layer
PIECE_CONSTANTS = "constants"  # task-completion + parallel + turn-summary (OSS or premium)
PIECE_SECURITY = "security"    # untrusted-content boundary (injection defense)
PIECE_DISCLOSURE = "disclosure"  # "this is an AI system" (EU AI Act, Art. 50(1))

SUPPRESSION_PIECES: FrozenSet[str] = frozenset(
    {PIECE_DISK, PIECE_CONSTANTS, PIECE_SECURITY, PIECE_DISCLOSURE}
)

# What the blanket ``true`` suppresses.  Security and disclosure are
# deliberately EXCLUDED — each is dropped only when named explicitly (dict
# ``security: true`` / list ``[..., "disclosure"]`` / the ``"all"`` token).
# Security is the indirect-prompt-injection defense; disclosure is a legal
# posture (Regulation (EU) 2024/1689, Art. 50(1): a person must be told they
# are interacting with an AI system), and neither is the kind of thing a
# convenience flag written to save tokens should silently remove.
_TRUE_PIECES: FrozenSet[str] = frozenset({PIECE_DISK, PIECE_CONSTANTS})

#: The pieces whose removal is a POSTURE change rather than a token saving,
#: and is therefore announced at WARNING by the session that applies it --
#: the rule ``scrub_secret_env: none`` and ``--ws-unsafe-no-auth`` follow.
ANNOUNCED_PIECES: FrozenSet[str] = frozenset({PIECE_SECURITY, PIECE_DISCLOSURE})

# Token that expands to every piece (list/set form), for callers who really do
# want to drop the security boundary too.
_ALL_TOKEN = "all"


def normalize_suppression(value: Any) -> FrozenSet[str]:
    """Normalize a ``suppress_base_instructions`` value to a canonical set.

    Args:
        value: ``None`` / ``bool`` / mapping / iterable of piece names /
            ``frozenset`` (idempotent).

    Returns:
        A ``frozenset`` of suppressed piece names (subset of
        :data:`SUPPRESSION_PIECES`).  Empty means "suppress nothing".

    Raises:
        ValueError: on an unknown piece name (mapping key or list item) or an
            unsupported value type — fail loud rather than silently keeping a
            layer the author meant to drop.
    """
    if value is None or value is False:
        return frozenset()
    if value is True:
        return _TRUE_PIECES

    # Mapping: {piece: truthy}.  Absent key = keep.
    if isinstance(value, Mapping):
        unknown = set(value.keys()) - SUPPRESSION_PIECES
        if unknown:
            raise ValueError(
                f"suppress_base_instructions: unknown piece(s) "
                f"{sorted(unknown)!r}; valid pieces are "
                f"{sorted(SUPPRESSION_PIECES)!r}"
            )
        return frozenset(k for k, v in value.items() if bool(v))

    # Iterable of piece names (wire form / set form).  Reject strings here —
    # a bare string would iterate into characters; callers pass a real list.
    if isinstance(value, (list, tuple, set, frozenset)):
        out: set = set()
        for item in value:
            if item == _ALL_TOKEN:
                return frozenset(SUPPRESSION_PIECES)
            if item not in SUPPRESSION_PIECES:
                raise ValueError(
                    f"suppress_base_instructions: unknown piece "
                    f"{item!r}; valid pieces are "
                    f"{sorted(SUPPRESSION_PIECES)!r} (or {_ALL_TOKEN!r})"
                )
            out.add(item)
        return frozenset(out)

    raise ValueError(
        "suppress_base_instructions must be a bool, a mapping over "
        f"{sorted(SUPPRESSION_PIECES)!r}, or a list of those piece names; "
        f"got {type(value).__name__}"
    )


def suppression_to_wire(pieces: FrozenSet[str]) -> List[str]:
    """Serialize a canonical suppression set to a stable, JSON-friendly list.

    Sorted for deterministic output (byte-identical wire / disk snapshots).
    ``normalize_suppression`` reads this list form back.
    """
    return sorted(pieces)
