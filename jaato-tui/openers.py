"""External-program openers for files in the workspace panel.

Loads a JSON config that maps fnmatch glob patterns to shell commands.
The workspace panel uses :func:`resolve_opener` to choose which command
to launch when the user opens a file (default keybinding: Enter).

Config search order (project entries override user entries):

    1. ``~/.jaato/openers.json`` (user)
    2. ``.jaato/openers.json`` (project)

Example config:

    {
        "*.md":       "glow -p",
        "*.markdown": "glow -p",
        "*.png":      "chafa",
        "*.jpg":      "chafa",
        "*":          "$EDITOR"
    }

Pattern matching uses :mod:`fnmatch` against the file's basename first,
falling back to the full relative path if no basename pattern matches.
When multiple patterns match, the longest pattern wins (most specific);
on a tie a basename match beats a path match.  If no pattern matches,
the default editor is used (``$EDITOR`` → ``$VISUAL`` → ``vi``).

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
from typing import Dict, List, Optional, Tuple

from editor_utils import get_editor

logger = logging.getLogger(__name__)


def load_openers(
    project_path: str = ".jaato/openers.json",
    user_path: Optional[str] = None,
) -> Dict[str, str]:
    """Load openers config from project + user JSON files.

    Project entries override user entries when patterns conflict.  Keys
    starting with ``_`` are ignored (reserved for comments/metadata).
    Malformed files log a warning and are skipped.

    Args:
        project_path: Project-level config path.
        user_path: User-level config path; defaults to
            ``~/.jaato/openers.json``.

    Returns:
        Dict mapping fnmatch patterns to command strings (may be empty).
    """
    if user_path is None:
        user_path = str(Path.home() / ".jaato" / "openers.json")

    merged: Dict[str, str] = {}
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

        for pattern, cmd in data.items():
            if pattern.startswith("_"):
                continue
            if not isinstance(cmd, str):
                logger.warning(
                    "Openers file %s: pattern %r → non-string command, skipped",
                    path,
                    pattern,
                )
                continue
            merged[pattern] = cmd

        logger.info("Loaded openers from %s", path)

    return merged


def resolve_opener(
    rel_path: str,
    openers: Dict[str, str],
) -> List[str]:
    """Resolve which argv to launch when opening *rel_path*.

    Each configured pattern is tested against both the basename
    (``foo.md``) and the full path (``docs/foo.md``).  The longest
    matching pattern wins; on a tie a basename match beats a path match.
    ``$EDITOR`` and ``$VISUAL`` inside the command string both resolve to
    :func:`editor_utils.get_editor`.  Other ``$VAR`` references are
    expanded via :func:`os.path.expandvars`.  Falls back to the default
    editor if no pattern matches.

    Args:
        rel_path: Path of the file to open.
        openers: Mapping of fnmatch patterns to command strings.

    Returns:
        Argv list (without the file path appended).
    """
    basename = os.path.basename(rel_path)

    # Score: (pattern_length, basename_priority).  Higher wins.
    best: Optional[Tuple[Tuple[int, int], str]] = None
    for pattern in openers.keys():
        if fnmatch.fnmatch(basename, pattern):
            score = (len(pattern), 1)
        elif fnmatch.fnmatch(rel_path, pattern):
            score = (len(pattern), 0)
        else:
            continue
        if best is None or score > best[0]:
            best = (score, pattern)

    cmd = openers[best[1]] if best is not None else get_editor()

    editor = get_editor()
    cmd = cmd.replace("$EDITOR", editor).replace("$VISUAL", editor)
    cmd = os.path.expandvars(cmd)

    argv = shlex.split(cmd)
    if not argv:
        argv = [editor]
    return argv
