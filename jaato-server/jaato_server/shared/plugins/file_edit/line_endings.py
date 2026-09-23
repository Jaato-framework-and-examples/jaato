"""Preserve a file's line endings across a ``file_edit`` write.

A targeted ``old``/``new`` edit must change the lines it was asked to change
and nothing else.  Before this module it changed every line ending in the
file as well: reads went through Python's universal-newline translation, so
a CRLF file arrived as LF and was written back that way, turning a one-line
edit into a whole-file rewrite in which the real change is unfindable
(jaato #805).

The fix is a round trip with an explicit middle:

``load()``
    decodes the file, remembers the ending it actually uses, and hands the
    caller LF-only text.  Matching, diffing and everything the model sees
    stays LF, so ``old``/``new`` fragments written with ``\\n`` keep working
    exactly as they do today.

``ending_for()``
    decides what the write should produce — the repository's own setting
    first (see :mod:`.git_eol`), then the file's existing ending, then LF
    for a genuinely new file with nothing to preserve.

``restore()``
    puts that ending back on the way out.

Callers must write the result with ``newline=""`` so no layer underneath
translates a second time; :func:`shared.plugins.path_safety.write_text_verified`
takes that argument.

Normalisation is still available — it is just no longer a side effect of
editing one line.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

from ..path_safety import read_bytes_verified, read_text_verified
from .git_eol import CRLF, LF, GitEolResolver

CR = "\r"

__all__ = [
    "CR",
    "CRLF",
    "LF",
    "LineEndingPolicy",
    "detect_line_ending",
    "normalize",
    "restore",
]


def detect_line_ending(text: str) -> Optional[str]:
    """Return the ending *text* predominantly uses, or ``None`` if it has none.

    Ties break towards CRLF, then LF: a file that has been edited into a
    mixed state is more likely to have started as CRLF (nothing adds CR to
    an LF file by accident, while every LF-only editor strips them), so
    keeping CRLF is the choice that shrinks the diff.

    Args:
        text: File content with its endings intact — i.e. *not* read through
            universal-newline translation, which erases the distinction.

    Returns:
        ``"\\r\\n"``, ``"\\n"``, ``"\\r"``, or ``None`` for a file with no
        line ending at all (empty, or a single unterminated line).
    """
    crlf = text.count(CRLF)
    cr = text.count(CR) - crlf
    lf = text.count(LF) - crlf
    if crlf >= lf and crlf >= cr and crlf > 0:
        return CRLF
    if lf >= cr and lf > 0:
        return LF
    return CR if cr > 0 else None


def normalize(text: str) -> str:
    """Convert every line ending in *text* to LF."""
    return text.replace(CRLF, LF).replace(CR, LF)


def restore(text: str, ending: str) -> str:
    """Re-apply *ending* to LF-normalised *text*.

    *text* is normalised first, so passing content that still holds CRLF —
    a full-replacement body straight from the model, say — is safe and
    idempotent rather than producing ``\\r\\r\\n``.
    """
    normalized = normalize(text)
    return normalized if ending == LF else normalized.replace(LF, ending)


class LineEndingPolicy:
    """Chooses the line ending each ``file_edit`` write should produce.

    Holds a :class:`~.git_eol.GitEolResolver`, whose caches make the
    repository lookup cheap enough to run on every write.  One policy object
    is shared by every write path in the plugin (``updateFile``,
    ``writeNewFile``, ``multiFileEdit``, ``findAndReplace``) so they cannot
    drift apart.

    The object is stateless with respect to any single file: nothing is
    remembered between :meth:`load` and the eventual write, because the two
    are separated by a permission prompt and a backup, and a file can change
    underneath in between.  Callers pass the content they read back into
    :meth:`ending_for`.
    """

    #: Bytes sampled by :meth:`ending_for_file` when it has to sniff a file
    #: it is not otherwise reading.  The dominant ending of the first 64 KiB
    #: is the file's ending in every realistic case, and the sample is only
    #: taken when the repository has expressed no preference at all.
    SNIFF_BYTES = 64 * 1024

    def __init__(self, git_resolver: Optional[GitEolResolver] = None) -> None:
        self._git = git_resolver if git_resolver is not None else GitEolResolver()

    def load(
        self,
        path: Path,
        *,
        validate: Optional[Callable[[str], bool]] = None,
        errors: str = "strict",
    ) -> Tuple[str, Optional[str]]:
        """Read *path* without translating its endings.

        Args:
            path: File to read.
            validate: Sandbox callback, forwarded to
                :func:`~shared.plugins.path_safety.read_text_verified`.
            errors: Decoding error policy.

        Returns:
            ``(content, ending)`` where *content* is LF-normalised and
            *ending* is what the file actually used, or ``None`` when it
            holds no line ending to preserve.

        Raises:
            OSError: on ordinary read failures, as the underlying helper does.
        """
        raw = read_text_verified(path, validate=validate, errors=errors, newline="")
        return normalize(raw), detect_line_ending(raw)

    def ending_for(self, path: Path, existing: Optional[str] = None) -> str:
        """Decide the ending to write at *path*.

        Args:
            path: Destination, used to locate the enclosing repository.
            existing: The ending the file currently uses, as returned by
                :meth:`load`.  ``None`` for a new file, or for one with no
                line ending yet.

        Returns:
            The ending to write.  A repository setting wins outright — it is
            what the checkout will hold anyway, so writing anything else just
            queues up a diff.  Otherwise the file keeps its own ending, and a
            file with no precedent gets LF.
        """
        from_git = self._git.ending_for(path)
        if from_git is not None:
            return from_git
        return existing if existing else LF

    def ending_for_file(
        self,
        path: Path,
        *,
        validate: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Decide the ending for a file whose content the caller has not read.

        Used by the whole-file write paths, which replace the content
        outright and so never load it.  The repository setting is consulted
        first as usual; only when it has none is a bounded prefix of the file
        sampled to recover its existing convention.  A path that does not
        exist, or cannot be read, resolves to LF.
        """
        from_git = self._git.ending_for(path)
        if from_git is not None:
            return from_git
        return self._sniff(path, validate=validate) or LF

    def _sniff(
        self,
        path: Path,
        *,
        validate: Optional[Callable[[str], bool]] = None,
    ) -> Optional[str]:
        """Detect an existing file's ending from its first :attr:`SNIFF_BYTES`."""
        try:
            raw = read_bytes_verified(path, validate=validate, max_bytes=self.SNIFF_BYTES)
        except (OSError, ValueError):
            return None
        # A sample can end mid-CRLF; that lone CR would otherwise be counted
        # as a CR-only ending.
        text = raw.decode("utf-8", errors="replace").rstrip(CR)
        return detect_line_ending(text)

    def encode(self, text: str, path: Path, existing: Optional[str] = None) -> str:
        """Apply :meth:`ending_for` to *text* — the whole write-side round trip."""
        return restore(text, self.ending_for(path, existing))
