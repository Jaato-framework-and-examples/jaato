"""Sentence-coherence tag matching, shared across plugins.

Both the ``memory`` and ``references`` plugins surface items based on
tag matches against arbitrary text (user prompts, tool results, file
contents).  The right matching algorithm is the same in both cases:

- Split text into sentence/line segments.
- A tag matches if its full string appears verbatim in any segment, or
  if **all** of its sub-tokens (≥3 chars, split on non-alphanumerics)
  co-occur in the same segment.
- Segments over a size cap fall back to verbatim-only matching, so
  large structured dumps (compact JSON, dense CLI output) don't
  trivially satisfy multi-component coherence by chance.

This module hosts the segmentation and coherence primitives so both
plugins evolve together.
"""

import re
from typing import List, Set, Tuple


# Sentence/segment splitter — see _segment_word_sets for rationale.
_SEGMENT_SPLIT_RE = re.compile(r"\n+|[.!?]+\s+")

# Maximum raw character length for a segment to be trusted for component
# coherence.  Larger segments fall back to verbatim-only.
SEGMENT_MAX_CHARS = 250

# Minimum component length — components shorter than this are dropped
# to avoid noise (e.g. ``v2`` in ``plan-v2``).  Tags whose components
# are all too short still match via the verbatim path.
COMPONENT_MIN_CHARS = 3


def tag_components(tag: str) -> List[str]:
    """Split a tag into its ≥3-char sub-tokens (lowercased).

    Splits on **typographical separators** (``-``, ``_``, ``:``) only.
    Dots are treated as **structural qualifiers** (think ``java.util``,
    ``spring.boot``) and are NOT separators — a tag like ``spring.boot``
    keeps as a single component, which routes it to the strict-boundary
    matching path so it doesn't accidentally match inside longer dotted
    names like ``org.spring.boot.autoconfigure``.

    ``workspace-baseline`` → ``["workspace", "baseline"]``
    ``agent:main`` → ``["agent", "main"]``
    ``circuit_breaker`` → ``["circuit", "breaker"]``
    ``api`` → ``["api"]``
    ``spring.boot`` → ``["spring.boot"]``  (dot kept; treated as atomic)
    ``plan-v2`` → ``["plan"]``  (``v2`` dropped as too short)
    ``gc`` → ``[]``  (no usable components — verbatim path only)
    """
    parts = re.split(r"[-_:\s]+", tag.lower())
    return [p for p in parts if len(p) >= COMPONENT_MIN_CHARS]


def text_segments(text: str) -> List[Tuple[str, Set[str]]]:
    """Split text into ``(raw_segment, word_set)`` pairs.

    Segments are sentence-or-line scoped.  Each word set includes
    whole tokens (with internal hyphens/underscores/colons/dots) AND
    alphanumeric sub-words, so a segment mentioning
    ``workspace-baseline`` contributes ``workspace-baseline``,
    ``workspace``, ``baseline``.

    Empty/whitespace-only segments are dropped.
    """
    result: List[Tuple[str, Set[str]]] = []
    for seg in _SEGMENT_SPLIT_RE.split(text):
        seg = seg.strip()
        if not seg:
            continue
        seg_lower = seg.lower()
        # Whole tokens preserve internal separators (`-` `_` `:` `.`)
        # so a segment containing ``circuit-breaker`` contributes that
        # whole string for verbatim matching.
        whole = set(re.findall(r"[a-z0-9][a-z0-9\-_:.]*[a-z0-9]", seg_lower))
        # Sub-words decompose on the same typographical separators that
        # ``tag_components`` splits on (``-`` ``_`` ``:``), so a tag
        # ``circuit-breaker`` whose components are ``circuit``+``breaker``
        # matches a segment containing ``circuit_breaker``.  Dots and
        # slashes are NOT separators here — tokens like ``java.util``
        # or ``path/to/file`` stay whole, blocking spurious component
        # matches inside qualified names and paths.
        sub = set(re.findall(r"[a-z0-9]+", seg_lower))
        result.append((seg, whole | sub))
    return result


# Strict word-boundary regex for single-component tags.  Disallows
# adjacent alphanumerics, hyphens, underscores, dots, and slashes —
# so tag ``java`` matches "the Java SDK" but NOT "java.util.concurrent",
# "java/lang" or "myjava".  This guards against tags accidentally
# matching inside dotted package names, file paths, and identifiers.
_STRICT_BOUNDARY = r"(?<![a-zA-Z0-9_./\-])"
_STRICT_BOUNDARY_AFTER = r"(?![a-zA-Z0-9_./\-])"


def tag_coherent_in_segments(
    tag: str,
    segments: List[Tuple[str, Set[str]]],
) -> bool:
    """Return True when *tag* is topically present in any segment.

    Matching rules differ by tag shape:

    **Single-component tags** (``java``, ``gpg``, ``gc``, ``api``):
    use **strict word-boundary regex** matching against the raw
    segment text.  Adjacent alphanumerics, hyphens, underscores,
    dots, and slashes block the match — so ``java`` matches
    "the Java SDK" but NOT "java.util.concurrent" or "myjava".

    **Multi-component tags** (``circuit-breaker``,
    ``workspace-baseline``):
    the full tag string appears verbatim in any segment OR all of
    its ≥3-char components co-occur in the same segment.  Hyphens
    make these unlikely to appear inside dotted names, so the
    component path is safe.  Segments larger than
    ``SEGMENT_MAX_CHARS`` fall back to verbatim-only matching.

    **Tags with no usable components** (too short — e.g. ``gc``):
    treated as single-component and matched via strict regex.
    """
    components = tag_components(tag)

    # Single-component (or no usable components) tags: strict regex.
    # Avoids false positives like ``java`` matching ``java.util``.
    if len(components) <= 1:
        try:
            pattern = re.compile(
                _STRICT_BOUNDARY + re.escape(tag.lower()) + _STRICT_BOUNDARY_AFTER
            )
        except re.error:
            return False
        for raw, _words in segments:
            if pattern.search(raw.lower()):
                return True
        return False

    # Multi-component tag: whole-tag verbatim OR component coherence.
    tag_lower = tag.lower()
    for _raw, words in segments:
        if tag_lower in words:
            return True

    for raw, words in segments:
        if len(raw) > SEGMENT_MAX_CHARS:
            continue
        if all(c in words for c in components):
            return True
    return False
