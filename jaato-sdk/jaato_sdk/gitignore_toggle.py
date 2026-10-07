"""Toggle one exact entry in a ``.gitignore`` — the workspace panel's ``i`` key.

The TUI's workspace panel lets the user add the entry under the cursor to
the workspace's ``.gitignore`` (and remove it again with the same key).  A
remote client — the web coding UI — cannot write that file itself, so the
daemon serves the same edit as ``workspace.ignore <path>``.  Both callers
must agree on what "the same edit" is, so the text transform lives here,
in the protocol package both already import, rather than being written
twice and drifting.

Semantics (unchanged from the TUI's original implementation):

- the pattern is the panel's entry id, written verbatim — a directory
  keeps its trailing ``/``, a file is its workspace-relative path;
- removal is **exact-match only**: a line equal to the pattern after
  stripping is dropped, every occurrence; a partial match, a glob that
  happens to cover the path, and a commented line are never touched;
- adding appends one line, making sure the existing content ends with a
  newline first, so the file is never left with two entries on one line.

The function is pure — it takes the file's text and returns the new text
— so it is exercised without a filesystem and cannot be affected by how
either caller reads or writes the file.
"""
from __future__ import annotations

from typing import Optional, Tuple


def validate_ignore_pattern(pattern: str) -> Optional[str]:
    """Return why ``pattern`` is not a workspace entry, or ``None`` if it is.

    The verb toggles *an entry the panel shows*, not an arbitrary
    ``.gitignore`` line, so the accepted shape is a workspace-relative
    path.  Refused, each with the reason the caller is handed:

    - empty or whitespace-only — nothing to write;
    - a line break or NUL — a second line smuggled into the file;
    - an absolute path — the panel's sandbox-monitored entries are
      absolute and lie outside the tree ``.gitignore`` covers, which is
      the same no-op the TUI has always made of them;
    - a leading ``#`` or ``!`` — git would read the line as a comment or a
      negation, so the "entry" would not be ignored and the toggle would
      report a state the file does not have.
    """
    if not pattern or not pattern.strip():
        return "empty pattern"
    if any(ch in pattern for ch in ("\n", "\r", "\0")):
        return "pattern contains a line break or NUL"
    if pattern.startswith("/") or pattern.startswith("\\") or (
            len(pattern) > 1 and pattern[1] == ":" and pattern[0].isalpha()):
        return (
            "absolute paths are not addressable via the workspace .gitignore "
            "(sandbox-monitored entries lie outside it)"
        )
    if pattern.lstrip().startswith(("#", "!")):
        return "a leading '#' or '!' would be read as a comment or a negation"
    return None


def toggle_gitignore_pattern(existing: str, pattern: str) -> Tuple[str, bool]:
    """Add ``pattern`` to ``existing`` if absent, else remove every exact match.

    Args:
        existing: The current ``.gitignore`` text (``""`` when there is no
            file yet).
        pattern: The entry to toggle, already validated by
            :func:`validate_ignore_pattern`.

    Returns:
        ``(new_text, ignored)`` where ``ignored`` is the entry's state
        AFTER the toggle: ``True`` when the line was added, ``False`` when
        it was removed.
    """
    lines = existing.splitlines()
    stripped = [ln.strip() for ln in lines]

    if pattern in stripped:
        new_lines = [ln for ln, s in zip(lines, stripped) if s != pattern]
        new_content = "\n".join(new_lines)
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"
        return new_content, False

    if existing and not existing.endswith("\n"):
        existing += "\n"
    return existing + pattern + "\n", True
