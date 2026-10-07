"""A pointer at a repository's own agent guidance (#1347).

A checkout often carries guidance written for agents or contributors:
``AGENTS.md``, ``CLAUDE.md``, ``CONTRIBUTING.md``,
``.github/copilot-instructions.md``, ``.cursor/rules``.  When the session's
workspace ROOT holds any of them, the framework adds ONE line to the system
instructions naming them, so the model reads the relevant one with
``readFile`` before working.

Rules this module holds to:

* **A pointer, never the contents.**  The files are never read.  Their text
  would enter the trusted region of the prompt (a third party's repository is
  a prompt-injection route), would be stale after the next ``git pull``, and
  may be large.  Read through ``readFile`` it is ordinary tool output.
* **The root only.**  The tree is never walked.  Guidance in cloned
  subdirectories is the web coder's managed file
  (``.jaato/instructions/30-repo-guidance.md``), which this module defers to.
* **No double pointer.**  A file in an instructions directory whose first
  line carries a ``jaato-managed: repo-guidance`` marker already names the
  paths it lists in backticks; a root-level name it lists (exactly
  ``AGENTS.md``, the root-relative path) is skipped here.  All skipped means
  nothing is emitted.
* **Best effort.**  Any ``OSError`` is swallowed: a missing pointer is a lost
  hint, a raised one would refuse the session.

The caller (``JaatoRuntime.get_base_system_instructions``) appends the line to
the ``disk`` base layer, so ``suppress_base_instructions: true`` or
``{disk: true}`` drops it together with ``.jaato/instructions/``.

Stdlib only, no jaato imports.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

#: Root-relative guidance paths, in the order the pointer names them.
GUIDANCE_NAMES = (
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    ".github/copilot-instructions.md",
    ".cursor/rules",
)

#: The marker the web coder writes on the first line of its managed file.
MANAGED_MARKER = "jaato-managed: repo-guidance"

# A managed file is small; a bound keeps a stray large file from being read whole.
_MANAGED_READ_LIMIT = 64 * 1024
_BACKTICKED = re.compile(r"`([^`\n]+)`")


def _present_names(root: Path) -> List[str]:
    """The guidance names present at *root*, in ``GUIDANCE_NAMES`` order.

    ``.cursor/rules`` may be a file or a directory; a directory is rendered
    with a trailing ``/`` so the model does not ``readFile`` it.  Every other
    name must be a regular file (symlinks followed).
    """
    found: List[str] = []
    for name in GUIDANCE_NAMES:
        path = root / name
        try:
            if name == ".cursor/rules" and path.is_dir():
                found.append(name + "/")
            elif path.is_file():
                found.append(name)
        except OSError:
            continue
    return found


def _managed_listing(path: Path) -> Set[str]:
    """Backticked paths in a managed repo-guidance file, else an empty set."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(_MANAGED_READ_LIMIT)
    except OSError:
        return set()
    first_line = text.split("\n", 1)[0]
    if MANAGED_MARKER not in first_line:
        return set()
    return {m.strip().removeprefix("./") for m in _BACKTICKED.findall(text)}


def managed_names(instructions_dirs: Iterable[Path]) -> Set[str]:
    """Every path named by a managed repo-guidance file in *instructions_dirs*."""
    named: Set[str] = set()
    for directory in instructions_dirs:
        try:
            files = sorted(Path(directory).glob("*.md"))
        except OSError:
            continue
        for path in files:
            named |= _managed_listing(path)
    return named


def _join(names: List[str]) -> str:
    quoted = [f"`{n}`" for n in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def repo_guidance_pointer(
    workspace_root: Optional[Path],
    instructions_dirs: Optional[Iterable[Path]] = None,
) -> Optional[str]:
    """The pointer line for *workspace_root*, or ``None`` when there is none.

    Args:
        workspace_root: The session's workspace root.  ``None`` or a path that
            is not a directory yields ``None``.
        instructions_dirs: Directories searched for a managed repo-guidance
            file.  Defaults to ``<workspace_root>/.jaato/instructions``.

    Returns:
        One sentence naming the present guidance files, minus those a managed
        file already names; ``None`` when nothing is left to name.  Never the
        files' contents.
    """
    if workspace_root is None:
        return None
    root = Path(workspace_root)
    try:
        if not root.is_dir():
            return None
    except OSError:
        return None
    present = _present_names(root)
    if not present:
        return None
    if instructions_dirs is None:
        instructions_dirs = [root / ".jaato" / "instructions"]
    skipped = managed_names(instructions_dirs)
    names = [n for n in present if n.rstrip("/") not in skipped and n not in skipped]
    if not names:
        return None
    what = "it" if len(names) == 1 else "the relevant one"
    return (
        f"This repository's own guidance is in {_join(names)} (at the "
        f"workspace root); read {what} with readFile before working."
    )


__all__ = [
    "GUIDANCE_NAMES",
    "MANAGED_MARKER",
    "managed_names",
    "repo_guidance_pointer",
]
