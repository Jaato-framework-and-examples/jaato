"""Read the ONE thing about an arm's bound profile that no event reports.

Everything else the per-arm report needs about the configuration comes
off the wire: ``SessionInfoEvent`` names the model and provider the
daemon actually bound, which beats reading a file and hoping the two
agree.  The per-arm **budget ceiling** has no such witness — the daemon
enforces ``budget_control`` without ever announcing it — so this module
reads it from the profile the arm ran under.

WHY THIS IS NOT A SECOND PROFILE LOADER
=======================================

It reads exactly one field.  ``model`` and ``provider`` are deliberately
NOT resolved here even though they sit in the same file: a resolver that
answered those questions too would be a second implementation of
profile binding, and the moment it disagreed with the daemon the report
would describe an arm that never ran.  The runner takes them from the
session instead.

The two rules it does implement are the framework's, restated with the
framework's own reasons:

* **Set-directory first.**  ``<config_root>/profiles/<set>/<name>`` wins
  over ``<config_root>/profiles/<name>``; this mirrors
  ``discover_profiles``, which scans the set overlay before the regular
  tier and lets first-scanned win.  It is what makes the sweep's model
  axis an axis at all.
* **Limits merge min-wins.**  A child may only ever TIGHTEN an inherited
  ceiling — the direction ``shared.budget_control.merge_limits`` takes,
  because child-replaces-parent on a resource ceiling is an escape
  hatch.  Getting this backwards would print a ceiling larger than the
  one the daemon enforced, which is the one error a budget column must
  not make.

Unknown is reported as ``None``, never as ``{}``: "this profile declares
no ceiling" and "this engine could not find the profile" are different
facts, and a report that renders them the same invites the reader to
conclude an arm was unbudgeted when it was merely unresolved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

#: Extensions ``_scan_profiles_dir`` accepts, in its own precedence order.
_SUFFIXES = (".yaml", ".yml", ".json")

#: Depth bound for the ``inherits:`` walk.  A cycle in a task's own
#: profiles is the daemon's error to report — this module must merely
#: refuse to hang on it.
_MAX_DEPTH = 16


def resolve_budget_ceiling(config_root: Path, profile: str,
                           profile_set: Optional[str]) -> Optional[Dict[str, float]]:
    """The per-arm ``budget_control.limits`` this arm ran under.

    Args:
        config_root: The task's read-only ``.jaato/`` tree.
        profile: Profile name from ``harness.profile``.
        profile_set: The arm's set — the sweep's model axis, written into
            the workspace ``.env`` as ``JAATO_PROFILE_SET``.

    Returns:
        Dimension -> ceiling, merged min-wins across the ``inherits:``
        chain.  ``None`` when the profile could not be found or declares
        no ``budget_control`` — see the module docstring on why that is
        not ``{}``.
    """
    document = _load(config_root, profile, profile_set)
    if document is None:
        return None
    limits = _limits(config_root, profile_set, document, depth=0)
    return limits or None


def _limits(config_root: Path, profile_set: Optional[str],
            document: Dict[str, Any], depth: int) -> Dict[str, float]:
    """Merge this document's limits over its parents', min-wins.

    Parents are folded in declaration order and the child applied last,
    exactly as ``_merge_budget_control`` does — the order is immaterial
    to a min, and preserving it keeps the correspondence readable.
    """
    merged: Dict[str, float] = {}
    if depth < _MAX_DEPTH:
        for parent in _inherits(document):
            parent_doc = _load(config_root, parent, profile_set)
            if parent_doc is not None:
                merged = _merge(merged, _limits(config_root, profile_set,
                                                parent_doc, depth + 1))
    return _merge(merged, _own_limits(document))


def _merge(parent: Dict[str, float], child: Dict[str, float]) -> Dict[str, float]:
    """Min-wins over the union — ``shared.budget_control.merge_limits``.

    An absent dimension is unbounded, so whichever layer declares one
    wins outright; where both declare, the tighter number is what the
    daemon enforced.
    """
    merged = dict(parent)
    for dimension, value in child.items():
        merged[dimension] = (min(merged[dimension], value)
                             if dimension in merged else value)
    return merged


def _own_limits(document: Dict[str, Any]) -> Dict[str, float]:
    """This document's own ``budget_control.limits``, numbers only.

    A non-numeric ceiling is dropped rather than coerced: the daemon
    would have rejected the profile, and inventing a number here would
    put a figure in the report that bounded nothing.
    """
    block = document.get("budget_control")
    if not isinstance(block, dict):
        return {}
    limits = block.get("limits")
    if not isinstance(limits, dict):
        return {}
    out: Dict[str, float] = {}
    for dimension, value in limits.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(dimension)] = float(value)
    return out


def _inherits(document: Dict[str, Any]) -> List[str]:
    """Parent names, accepting both spellings the loader accepts."""
    value = document.get("inherits")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def _load(config_root: Path, name: str,
          profile_set: Optional[str]) -> Optional[Dict[str, Any]]:
    """Find and parse one profile document, set directory first.

    Returns ``None`` for absent and for unparseable alike.  The
    distinction matters to whoever wrote the file, but not here: the
    daemon reports parse errors (``SessionProfilesEvent.parse_errors``)
    and an arm whose profile does not parse never runs, so this module's
    only job is to avoid claiming a ceiling it did not read.
    """
    profiles = Path(config_root) / "profiles"
    directories = ([profiles / profile_set] if profile_set else []) + [profiles]
    for directory in directories:
        for suffix in _SUFFIXES:
            candidate = directory / f"{name}{suffix}"
            if not candidate.is_file():
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
                data = (json.loads(text) if suffix == ".json"
                        else yaml.safe_load(text))
            except (OSError, ValueError, yaml.YAMLError):
                return None
            return data if isinstance(data, dict) else None
    return None
