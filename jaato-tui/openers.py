"""External-program openers for files in the workspace panel.

Loads a JSON config that maps fnmatch glob patterns to one or more
**actions**.  Each action names a different launcher for the same file —
e.g. ``raw`` for the editor and ``diff`` for a diff viewer.  The
workspace panel uses :func:`resolve_opener` to choose which command to
launch, picking the action based on the keybinding the user pressed
(default Enter → ``raw``, default ``d`` → ``diff``).

Config search order (project entries override user entries per-action):

    1. ``~/.jaato/openers.json`` (user)
    2. ``.jaato/openers.json`` (project)

Example config (new dict-per-pattern shape):

    {
        "*.md":  { "raw": "glow -p", "diff": "git diff HEAD --" },
        "*.png": { "raw": "chafa" },
        "*":     { "raw": "$EDITOR", "diff": "git diff HEAD --" }
    }

For backward compatibility, a bare string is treated as the ``raw``
action only.  These two pattern entries are equivalent::

    "*.md": "glow -p"
    "*.md": { "raw": "glow -p" }

Pattern matching uses :mod:`fnmatch` against the file's basename first,
falling back to the full relative path if no basename pattern matches.
When multiple patterns match, the longest pattern wins (most specific);
on a tie a basename match beats a path match.

**Action fallthrough.**  If the most-specific matching pattern doesn't
define the requested action, the resolver walks the next-most-specific
match and so on.  This means you don't have to redeclare ``diff`` on
every pattern that overrides ``raw`` — a catch-all ``"*"`` entry can
supply the default for any action.

If no pattern defines the requested action, ``raw`` falls back to the
default editor (``$EDITOR`` → ``$VISUAL`` → ``vi``) and other actions
return an empty argv (the caller should treat that as a no-op).

Commands may reference ``$EDITOR`` and ``$VISUAL`` as placeholders;
both resolve to the user's default editor.  Other ``$VAR`` references
are expanded via :func:`os.path.expandvars`.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

from editor_utils import get_editor

logger = logging.getLogger(__name__)

# Action names recognised in config files.  Unknown action names are
# accepted (and stored) so future actions can be configured without a
# code change, but only these are wired to keybindings today.
ACTION_RAW = "raw"
ACTION_DIFF = "diff"

OpenerSpec = Dict[str, Dict[str, str]]
"""Loaded openers map: ``pattern → {action_name: command_string}``."""


def _normalize_pattern_value(
    pattern: str,
    value: object,
    source_path: str,
) -> Optional[Dict[str, str]]:
    """Coerce a raw config value into ``{action: command}``.

    Bare strings are treated as the ``raw`` action.  Dicts are validated
    entry-by-entry; non-string commands and non-string action keys are
    skipped with a warning.  Returns ``None`` if the value can't be
    interpreted at all (e.g. number, list).
    """
    if isinstance(value, str):
        return {ACTION_RAW: value}
    if isinstance(value, dict):
        actions: Dict[str, str] = {}
        for action_key, cmd in value.items():
            if not isinstance(action_key, str) or not isinstance(cmd, str):
                logger.warning(
                    "Openers file %s: pattern %r action %r → non-string entry, skipped",
                    source_path,
                    pattern,
                    action_key,
                )
                continue
            actions[action_key] = cmd
        return actions
    logger.warning(
        "Openers file %s: pattern %r → expected string or object, got %s",
        source_path,
        pattern,
        type(value).__name__,
    )
    return None


def load_openers(
    project_path: str = ".jaato/openers.json",
    user_path: Optional[str] = None,
) -> OpenerSpec:
    """Load openers config from project + user JSON files.

    Project entries override user entries **per action** when patterns
    conflict — a user-level ``raw`` for ``*.md`` survives a project that
    only declares ``diff`` for the same pattern.  Keys starting with
    ``_`` are ignored (reserved for comments/metadata).  Malformed
    entries log a warning and are skipped.

    Args:
        project_path: Project-level config path.
        user_path: User-level config path; defaults to
            ``~/.jaato/openers.json``.

    Returns:
        Dict mapping fnmatch patterns to ``{action: command}`` dicts.
        May be empty.
    """
    if user_path is None:
        user_path = str(Path.home() / ".jaato" / "openers.json")

    merged: OpenerSpec = {}
    for path in (user_path, project_path):  # project loaded last → wins
        cfg_path = Path(path)
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in openers file %s: %s", path, e)
            continue
        except OSError as e:
            logger.warning("Could not read openers file %s: %s", path, e)
            continue

        if not isinstance(data, dict):
            logger.warning(
                "Openers file %s: expected object, got %s",
                path,
                type(data).__name__,
            )
            continue

        for pattern, value in data.items():
            if pattern.startswith("_"):
                continue
            actions = _normalize_pattern_value(pattern, value, path)
            if not actions:
                continue
            # Per-action merge: project overrides user only for actions
            # the project actually declares.
            merged.setdefault(pattern, {}).update(actions)

        logger.info("Loaded openers from %s", path)

    return merged


def resolve_opener(
    rel_path: str,
    openers: OpenerSpec,
    action: str = ACTION_RAW,
) -> List[str]:
    """Resolve which argv to launch when opening *rel_path* with *action*.

    Each configured pattern is tested against both the basename
    (``foo.md``) and the full path (``docs/foo.md``).  Matches are
    ranked most-specific first by ``(pattern_length, basename_priority)``;
    the resolver walks them in that order and picks the first pattern
    that defines the requested action.  This means a more-specific
    pattern that lacks the action transparently falls through to the
    next, so a catch-all ``"*"`` entry can supply defaults.

    ``$EDITOR`` and ``$VISUAL`` inside the command string both resolve to
    :func:`editor_utils.get_editor`.  Other ``$VAR`` references are
    expanded via :func:`os.path.expandvars`.

    Fallback when no matching pattern defines the action:

    * ``raw`` → default editor (``$EDITOR`` → ``$VISUAL`` → ``vi``)
    * any other action → empty list (caller should treat as no-op)

    Args:
        rel_path: Path of the file to open.
        openers: Mapping of fnmatch patterns to ``{action: command}``.
        action: Which action to resolve (e.g. ``"raw"``, ``"diff"``).

    Returns:
        Argv list (without the file path appended).  May be empty for
        non-``raw`` actions when nothing matches.
    """
    basename = os.path.basename(rel_path)

    # Score: (pattern_length, basename_priority).  Higher wins.
    matches: List[Tuple[Tuple[int, int], str]] = []
    for pattern in openers.keys():
        if fnmatch.fnmatch(basename, pattern):
            score = (len(pattern), 1)
        elif fnmatch.fnmatch(rel_path, pattern):
            score = (len(pattern), 0)
        else:
            continue
        matches.append((score, pattern))

    matches.sort(key=lambda m: m[0], reverse=True)

    cmd: Optional[str] = None
    for _, pattern in matches:
        actions = openers.get(pattern) or {}
        if action in actions:
            cmd = actions[action]
            break

    if cmd is None:
        if action != ACTION_RAW:
            return []
        cmd = get_editor()

    editor = get_editor()
    cmd = cmd.replace("$EDITOR", editor).replace("$VISUAL", editor)
    cmd = os.path.expandvars(cmd)

    argv = shlex.split(cmd)
    if not argv:
        if action != ACTION_RAW:
            return []
        argv = [editor]
    return argv
