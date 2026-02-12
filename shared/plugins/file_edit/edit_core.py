"""Core logic for targeted search-and-replace edits.

Provides the shared ``apply_edit()`` function used by both ``updateFile``
(targeted mode) and ``multiFileEdit`` (edit action).

The algorithm:
1. Build an *anchor* string: ``prologue + old + epilogue``
2. Verify the anchor appears exactly once in the file content.
3. Replace only the ``old`` portion with ``new``, preserving prologue/epilogue.

Raises ``EditNotFoundError`` when the anchor is absent and
``AmbiguousEditError`` when the anchor matches more than once.
"""


class EditNotFoundError(Exception):
    """The search text (with optional prologue/epilogue) was not found in the file."""


class AmbiguousEditError(Exception):
    """The search text (with optional prologue/epilogue) matched multiple locations."""


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

    Args:
        file_content: The full text of the file.
        old: The text fragment to find and replace.
        new: The replacement text.
        prologue: Optional text that must appear immediately before *old*.
            Used for disambiguation when *old* alone is ambiguous.
        epilogue: Optional text that must appear immediately after *old*.
            Used for disambiguation when *old* alone is ambiguous.

    Returns:
        The updated file content with the single replacement applied.

    Raises:
        EditNotFoundError: If the anchor (prologue+old+epilogue) is not
            found anywhere in *file_content*.
        AmbiguousEditError: If the anchor appears more than once.
    """
    prologue_text = prologue or ""
    epilogue_text = epilogue or ""

    anchor = prologue_text + old + epilogue_text
    replacement = prologue_text + new + epilogue_text

    count = file_content.count(anchor)

    if count == 0:
        # Build a helpful error message
        if prologue or epilogue:
            msg = (
                f"Search text not found (with context anchors).\n"
                f"  prologue: {_truncate(prologue_text, 80)!r}\n"
                f"  old:      {_truncate(old, 80)!r}\n"
                f"  epilogue: {_truncate(epilogue_text, 80)!r}"
            )
        else:
            msg = f"Search text not found: {_truncate(old, 120)!r}"
        raise EditNotFoundError(msg)

    if count > 1:
        if prologue or epilogue:
            msg = (
                f"Search text matched {count} times even with context anchors. "
                f"Provide more specific prologue/epilogue to disambiguate.\n"
                f"  prologue: {_truncate(prologue_text, 80)!r}\n"
                f"  old:      {_truncate(old, 80)!r}\n"
                f"  epilogue: {_truncate(epilogue_text, 80)!r}"
            )
        else:
            msg = (
                f"Search text matched {count} times. "
                f"Use prologue/epilogue to disambiguate: {_truncate(old, 120)!r}"
            )
        raise AmbiguousEditError(msg)

    return file_content.replace(anchor, replacement, 1)


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* for display in error messages."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
