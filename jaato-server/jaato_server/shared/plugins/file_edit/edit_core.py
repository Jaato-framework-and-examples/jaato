"""Core logic for targeted search-and-replace edits.

Provides the shared ``apply_edit()`` function used by both ``updateFile``
(targeted mode) and ``multiFileEdit`` (edit action).

The algorithm:
1. Reject degenerate requests that cannot describe any replacement.
2. Build an *anchor* string: ``prologue + old + epilogue``
3. Verify the anchor appears exactly once in the file content.
4. Replace only the ``old`` portion with ``new``, preserving prologue/epilogue.

Raises ``MalformedEditError`` when the request itself is degenerate,
``EditNotFoundError`` when the anchor is absent and ``AmbiguousEditError``
when the anchor matches more than once.

The three exceptions are kept distinct because their *remedies* differ, and
a caller handed the remedy for a failure it did not have will loop on it:

* ``AmbiguousEditError`` — narrow the match (add or extend anchors).
* ``EditNotFoundError`` — fix the search text, or the anchors' adjacency.
* ``MalformedEditError`` — neither of the above can help; the call itself
  has to change (#814).
"""

# How many occurrence line numbers a diagnostic names before summarising
# the rest as a count.  Three is enough to show a pattern without turning
# the error into a listing.
_MAX_REPORTED_OCCURRENCES = 3


class EditNotFoundError(Exception):
    """The search text (with optional prologue/epilogue) was not found in the file."""


class AmbiguousEditError(Exception):
    """The search text (with optional prologue/epilogue) matched multiple locations."""


class MalformedEditError(Exception):
    """The edit request is degenerate — it cannot describe any replacement.

    Raised at the input boundary, before any matching is attempted, so a
    degenerate call is never diagnosed as a *match* failure.  An empty
    ``old`` occurs between every pair of characters in the file, so it used
    to fall through to ``AmbiguousEditError`` and be reported as "matched
    ``len(content) + 1`` times, add anchors" — advice no anchor can
    satisfy, which callers nonetheless keep following because it reads like
    progress is possible (#814).
    """


def apply_edit(
    file_content: str,
    old: str,
    new: str,
    prologue: str | None = None,
    epilogue: str | None = None,
) -> str:
    """Find ``old`` in *file_content* and replace it with ``new``.

    When *prologue* and/or *epilogue* are provided they are concatenated
    around ``old`` to form a wider *anchor* that must appear exactly once.
    Only the ``old`` portion of the matched anchor is replaced with
    ``new``; prologue and epilogue text is preserved in the output.

    Because the anchor is built by direct concatenation, *prologue* and
    *epilogue* must be **byte-adjacent** to ``old`` — landmarks that merely
    occur elsewhere in the file can never match.  When they don't, the
    raised ``EditNotFoundError`` says so and names where each piece
    actually occurs, rather than only echoing the three pieces back.

    Args:
        file_content: The full text of the file.
        old: The text fragment to find and replace.  Must be non-empty and
            differ from *new*.
        new: The replacement text.
        prologue: Optional text that must appear immediately before *old*.
            Used for disambiguation when *old* alone is ambiguous.
        epilogue: Optional text that must appear immediately after *old*.
            Used for disambiguation when *old* alone is ambiguous.

    Returns:
        The updated file content with the single replacement applied.

    Raises:
        MalformedEditError: If the request is degenerate — *old* is empty,
            or *old* and *new* are identical — so no match result could
            make it a valid edit.
        EditNotFoundError: If the anchor (prologue+old+epilogue) is not
            found anywhere in *file_content*.
        AmbiguousEditError: If the anchor appears more than once.
    """
    _reject_degenerate_edit(old, new)

    prologue_text = prologue or ""
    epilogue_text = epilogue or ""

    anchor = prologue_text + old + epilogue_text
    replacement = prologue_text + new + epilogue_text

    count = file_content.count(anchor)

    if count == 0:
        raise EditNotFoundError(
            _not_found_message(file_content, old, prologue_text, epilogue_text)
        )

    if count > 1:
        if prologue_text or epilogue_text:
            msg = (
                f"Search text matched {count} times even with context anchors. "
                f"Extend prologue/epilogue outward — more literal adjacent lines "
                f"copied verbatim from the file (blank lines included) — until the "
                f"match is unique.\n"
                f"{_pieces_dump(old, prologue_text, epilogue_text)}"
            )
        else:
            msg = (
                f"Search text matched {count} times. Add 'prologue'/'epilogue' — "
                f"the literal lines immediately before/after 'old', copied verbatim "
                f"from the file (blank lines included), extended until the match is "
                f"unique: {_truncate(old, 120)!r}"
            )
        raise AmbiguousEditError(msg)

    return file_content.replace(anchor, replacement, 1)


def _reject_degenerate_edit(old: str, new: str) -> None:
    """Refuse edit requests that no file content could make valid.

    These are input errors, not match failures: an empty ``old`` names no
    text to replace, and ``old == new`` names no change.  Both are caught
    here so the caller is told what is actually wrong instead of being
    handed match-narrowing advice it can never act on (#814).

    Raises:
        MalformedEditError: For either degenerate case.
    """
    if not old:
        raise MalformedEditError(
            "'old' is empty, so it names no text to replace — an empty string "
            "occurs between every pair of characters in the file, and no "
            "'prologue'/'epilogue' can narrow that to one site. Supply the "
            "exact text to replace, or to rewrite the file's whole contents "
            "omit 'old' and pass the full text instead."
        )

    if old == new:
        raise MalformedEditError(
            "'old' and 'new' are identical, so this edit would change nothing. "
            "Put the replacement text in 'new'. (If the change is already "
            "applied, no call is needed.)"
        )


def _not_found_message(
    file_content: str,
    old: str,
    prologue_text: str,
    epilogue_text: str,
) -> str:
    """Explain *why* ``prologue + old + epilogue`` was not found.

    With no anchors in play there is only one thing to say.  With anchors,
    the file content is right here, so one extra count separates the cases
    a caller has to tell apart (#813):

    * ``old`` absent entirely → the anchors are irrelevant; the search text
      is wrong.
    * ``old`` present but the concatenation absent → the anchors are not
      byte-adjacent to it (or not verbatim).  Naming which side failed, and
      where that piece actually sits, is what breaks the loop in which a
      caller refines the same wrong thing.
    """
    if not (prologue_text or epilogue_text):
        return f"Search text not found: {_truncate(old, 120)!r}"

    old_total = file_content.count(old)

    if old_total == 0:
        return (
            "Search text not found (with context anchors) — and 'old' does not "
            "occur in the file on its own either, so the anchors are not the "
            "problem. Fix 'old' first: copy it verbatim from the file.\n"
            f"{_pieces_dump(old, prologue_text, epilogue_text)}"
        )

    parts = [
        f"Search text not found with context anchors, but 'old' occurs "
        f"{_format_occurrences(file_content, old, old_total)}. "
        f"'prologue'/'epilogue' must be the text IMMEDIATELY adjacent to "
        f"'old': they are joined to it verbatim "
        f"('prologue'+'old'+'epilogue') and that whole string must be a "
        f"substring of the file, so landmarks further away can never match."
    ]
    parts.extend(_anchor_details(file_content, old, prologue_text, epilogue_text))
    parts.append(_pieces_dump(old, prologue_text, epilogue_text))
    return "\n".join(parts)


def _anchor_details(
    file_content: str,
    old: str,
    prologue_text: str,
    epilogue_text: str,
) -> list[str]:
    """Per-anchor diagnostic lines naming which side broke adjacency."""
    details = []

    if prologue_text and (prologue_text + old) not in file_content:
        details.append(_describe_anchor("prologue", prologue_text, file_content))

    if epilogue_text and (old + epilogue_text) not in file_content:
        details.append(_describe_anchor("epilogue", epilogue_text, file_content))

    if not details:
        # Each anchor is adjacent to *some* occurrence of 'old' — just not
        # to the same one, so their concatenation still isn't a substring.
        details.append(
            "  'prologue' and 'epilogue' are each adjacent to a different "
            "occurrence of 'old'. Anchor both sides of the same one."
        )

    return details


def _describe_anchor(label: str, text: str, file_content: str) -> str:
    """One diagnostic line for an anchor that is not adjacent to ``old``."""
    total = file_content.count(text)
    if total == 0:
        return (
            f"  '{label}' does not occur in the file at all — copy it verbatim, "
            f"including indentation and blank lines."
        )
    return (
        f"  '{label}' occurs {_format_occurrences(file_content, text, total)}, "
        f"but not immediately adjacent to any occurrence of 'old'."
    )


def _format_occurrences(file_content: str, needle: str, total: int) -> str:
    """Render where *needle* occurs as ``"once, at line 34"`` / ``"3 times, at lines ..."``."""
    lines = _occurrence_lines(file_content, needle)
    shown = ", ".join(str(n) for n in lines)
    if total == 1:
        return f"once, at line {shown}"
    if total > len(lines):
        return f"{total} times, at lines {shown} and {total - len(lines)} more"
    return f"{total} times, at lines {shown}"


def _occurrence_lines(file_content: str, needle: str) -> list[int]:
    """1-based line numbers of the first few occurrences of *needle*.

    Occurrences are counted non-overlapping, matching ``str.count()``, so
    the numbers agree with the totals quoted alongside them.
    """
    lines = []
    start = 0
    while len(lines) < _MAX_REPORTED_OCCURRENCES:
        index = file_content.find(needle, start)
        if index < 0:
            break
        lines.append(file_content.count("\n", 0, index) + 1)
        start = index + len(needle)
    return lines


def _pieces_dump(old: str, prologue_text: str, epilogue_text: str) -> str:
    """The three search pieces, truncated, for verbatim comparison."""
    return (
        f"  prologue: {_truncate(prologue_text, 80)!r}\n"
        f"  old:      {_truncate(old, 80)!r}\n"
        f"  epilogue: {_truncate(epilogue_text, 80)!r}"
    )


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* for display in error messages."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
