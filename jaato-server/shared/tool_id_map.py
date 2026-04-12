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
"""

import hashlib
from typing import Dict


_reverse: Dict[str, str] = {}


def name_to_id(name: str, prefix: str = "t") -> str:
    """Derive a stable, opaque ID from a human-readable name.

    Uses the first 8 hex characters of the SHA-256 hash, giving ~4 billion
    possible values — collision probability is negligible for <10K names.

    Tool-name suffixes (e.g. ``:stream``) are preserved through the mapping:
    ``grep_content:stream`` → ``t_<hash-of-grep_content>:stream``.  This
    ensures the model sees a consistent relationship between base tools and
    their variants.

    Args:
        name: Human-readable tool or category name.
        prefix: ID prefix (``"t"`` for tools, ``"c"`` for categories).

    Returns:
        Stable ID string, e.g. ``"t_a3f2b1c0"`` or ``"t_a3f2b1c0:stream"``.
    """
    # Preserve suffixes (e.g. ":stream") so the model-facing ID keeps the
    # same base hash as the unsuffixed tool.
    suffix = ""
    base_name = name
    colon = name.rfind(":")
    if colon > 0:
        suffix = name[colon:]   # e.g. ":stream"
        base_name = name[:colon]

    h = hashlib.sha256(base_name.encode()).hexdigest()[:8]
    base_id = f"{prefix}_{h}"
    _reverse[base_id] = base_name
    return f"{base_id}{suffix}"


def id_to_name(id_str: str) -> str:
    """Resolve an ID back to the original human-readable name.

    Handles suffixes transparently: ``t_a3f2b1c0:stream`` is resolved by
    looking up ``t_a3f2b1c0`` and re-appending ``:stream``.

    Returns the input unchanged if the ID is not recognized (e.g., when
    the model hallucinates a tool ID that was never issued).

    Args:
        id_str: The hash-derived ID from the model's response.

    Returns:
        Original name, or the input if not found.
    """
    # Fast path: direct lookup (covers all unsuffixed IDs).
    resolved = _reverse.get(id_str)
    if resolved is not None:
        return resolved

    # Suffix-aware lookup: strip suffix, resolve base, re-append.
    colon = id_str.rfind(":")
    if colon > 0:
        base_id = id_str[:colon]
        suffix = id_str[colon:]
        base_name = _reverse.get(base_id)
        if base_name is not None:
            return f"{base_name}{suffix}"

    return id_str
