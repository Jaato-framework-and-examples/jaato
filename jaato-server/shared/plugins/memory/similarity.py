"""Near-duplicate detection for ``store_memory`` (#973).

``store_memory`` had no notion of "I already know this".  A voice agent's
greeting turn re-annotated the memory inventory it had been handed at
session creation and wrote **the same three facts 83 times in 179
seconds** — 30, 29 and 24 near-identical copies — each one answered with
a bare ``{"status": "success"}``.  All three were already in
``curated.jsonl``, promoted correctly by the curator in the previous
session, before the first of the 83 was written.

Nothing here rejects anything.  A store still succeeds; the plugin
merely says, in the result the model reads, that the content it just
wrote looks like something already stored.  That asymmetry is the whole
reason this module can afford a heuristic at all:

- a **false positive** costs one advisory field the model may ignore;
- a **false negative** costs exactly what happens today — nothing new.

Rejection is available (``reject_duplicates``) and is off by default,
because a similarity threshold that is wrong in the rejecting direction
silently discards real memories, which for this plugin is the worst
available failure.

THE MEASURE
-----------

Token **overlap coefficient** over ``description + content``:

    |A ∩ B| / min(|A|, |B|)

Overlap rather than symmetric Jaccard because a rephrasing characteristically
*adds* words — "ha trabajado durante dos años en X" becomes "ha compartido que
durante dos años trabajó en X".  Jaccard charges that addition to the
similarity score; the overlap coefficient asks the question actually worth
asking, which is whether the shorter text is already contained in the longer
one.  A terse restatement of a longer existing memory scoring 1.0 is the
correct answer, not a pathology — it adds nothing the store does not hold.

No stopword list.  :data:`~.indexer.STOPWORDS` is English-only and the
incident was in Spanish, so applying it there would strip nothing while
appearing to; not applying it anywhere keeps the measure language-agnostic
and keeps the floor below honest rather than accidentally low for one
language.

WHY 0.75
--------

Function words are counted, so two *unrelated* memories written in the same
language share a floor of shared tokens.  Measured over same-language,
same-register pairs (Spanish and English memories about different subjects,
including pairs sharing a subject — "El usuario prefiere…" against "El
usuario ha trabajado…"), that floor tops out at **0.30**.  Rephrasings in the
style the issue quotes score **0.90–0.92**.  The default sits between the two
with a wide margin on each side: roughly 2.5x the observed noise floor and
comfortably under the observed duplicate band, so neither a slightly chattier
noise pair nor a more heavily reworded duplicate crosses it.

Stated as a limitation rather than a result: the duplicate figures come from
reconstructions of the incident's strings in the style the report quotes, not
from the strings themselves — #973 truncates all three — so the separation
above is a measurement of the *measure*, not a fit to the incident's data.
The margin is wide enough that this is not load-bearing, and nothing is
rejected on it by default.

NO TAG GATE
-----------

An earlier shape of this required the two memories to share a tag before
comparing text, reusing the relatedness notion ``search_by_tags`` already
implements.  It is cheaper and it was dropped: a model rephrasing a fact
plausibly re-tags it too, and a tag gate turns that into a silent false
negative in exactly the case the feature exists for.  With nothing being
rejected, recall is the property worth buying.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple


#: Default text-similarity score at or above which two memories are reported
#: as near-duplicates.  See the module docstring for the measurement behind
#: the number.  Overridable per deployment via
#: ``plugin_configs.memory.duplicate_threshold``.
DEFAULT_DUPLICATE_THRESHOLD = 0.75

#: Minimum distinct tokens the SHORTER of two texts must carry before a
#: similarity verdict is offered at all.  The overlap coefficient reports 1.0
#: for any two-word text contained in a longer one, which is noise rather than
#: a duplicate claim; below this floor the answer is "no opinion".
MIN_COMPARABLE_TOKENS = 4

#: Minimum length of a token that counts.  Single characters carry no topical
#: signal and inflate the shared-token count between unrelated texts.
MIN_TOKEN_CHARS = 2

#: Unicode word runs, underscores excluded so ``oauth_pkce_flow`` contributes
#: its parts.  ``[^\W_]`` is "word character but not underscore"; it keeps
#: accented letters, which an ASCII-only pattern would split ``años`` on.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class DuplicateMatch:
    """One already-stored memory a new one closely resembles.

    Attributes:
        memory_id: The id of the memory already in the store.  This is what
            reaches the model as ``duplicate_of``, so it must be usable
            directly as a ``retrieve_memories(ids=[…])`` argument.
        description: That memory's own description, so the model can tell
            whether the match is the fact it meant without a second call.
        similarity: The measured score, in ``[0.0, 1.0]``.
        source: Which pool the match came from — ``"session"`` for something
            this same plugin instance stored (the runaway-loop case) or
            ``"curated"`` for the curator-vetted store.  The two mean
            different things to a reader: "you wrote this minutes ago" versus
            "this was reviewed and kept".
    """

    memory_id: str
    description: str
    similarity: float
    source: str


def content_tokens(*parts: Optional[str]) -> Set[str]:
    """Tokenise memory text into the set the similarity measure compares.

    Args:
        *parts: Text fragments — normally a memory's ``description`` and
            ``content``.  ``None`` and empty fragments are skipped, so a
            caller need not pre-filter optional fields.

    Returns:
        Lowercased distinct tokens of at least :data:`MIN_TOKEN_CHARS`
        characters.  Order is not meaningful; the measure is set-based.
    """
    tokens: Set[str] = set()
    for part in parts:
        if not part:
            continue
        for token in _TOKEN_RE.findall(part.lower()):
            if len(token) >= MIN_TOKEN_CHARS:
                tokens.add(token)
    return tokens


def similarity(left: Set[str], right: Set[str]) -> float:
    """Overlap coefficient between two token sets.

    Args:
        left: Token set of one text.
        right: Token set of the other.

    Returns:
        ``|left ∩ right| / min(|left|, |right|)`` in ``[0.0, 1.0]``, or
        ``0.0`` when either side is below :data:`MIN_COMPARABLE_TOKENS` —
        too little text to make a containment claim about.  Returning a
        score rather than raising keeps the caller branch-free.
    """
    smaller = min(len(left), len(right))
    if smaller < MIN_COMPARABLE_TOKENS:
        return 0.0
    return len(left & right) / smaller


def find_near_duplicate(
    candidate_tokens: Set[str],
    pools: Sequence[Tuple[str, Iterable]],
    *,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    exclude_ids: Optional[Set[str]] = None,
) -> Optional[DuplicateMatch]:
    """Find the stored memory a candidate most closely resembles.

    Scans every pool in full and returns the **best** match rather than the
    first one over the threshold: with several near-duplicates on file, the
    closest is the one worth naming, and "you already know this, as mem_x"
    is only actionable if ``mem_x`` is the right one.

    Cost is one tokenisation per pooled memory per call.  The pools are the
    curated stores (kept small by the curator, which consolidates) plus this
    session's own recent writes (bounded).  The raw queue is deliberately not
    a pool — see ``MemoryPlugin._duplicate_pools``.

    Args:
        candidate_tokens: :func:`content_tokens` of the memory about to be
            stored.
        pools: ``(source_label, memories)`` pairs.  Each memory needs ``id``,
            ``description`` and ``content`` attributes.  Iterated once.
        threshold: Minimum score to report.  Scores strictly below it yield
            ``None``.
        exclude_ids: Ids to skip — the candidate itself when it has already
            been written, and any id a nearer pool already covered, so the
            same memory is not compared twice under two source labels.

    Returns:
        The best :class:`DuplicateMatch` at or above ``threshold``, or
        ``None`` when nothing reaches it.
    """
    if not candidate_tokens:
        return None
    skip = exclude_ids or set()
    best: Optional[DuplicateMatch] = None
    for source, memories in pools:
        for mem in memories:
            if mem.id in skip:
                continue
            score = similarity(
                candidate_tokens,
                content_tokens(mem.description, mem.content),
            )
            if score < threshold:
                continue
            if best is not None and score <= best.similarity:
                continue
            best = DuplicateMatch(
                memory_id=mem.id,
                description=mem.description,
                similarity=round(score, 3),
                source=source,
            )
    return best


def resolve_threshold(raw, default: float = DEFAULT_DUPLICATE_THRESHOLD) -> float:
    """Coerce a configured ``duplicate_threshold`` into a usable float.

    Args:
        raw: The configured value — anything a YAML profile can produce,
            including ``None`` for "unset" and an unexpanded ``${VAR}``
            string.
        default: The value to fall back to.

    Returns:
        ``raw`` clamped to ``[0.0, 1.0]`` when it is numeric, else
        ``default``.  A malformed value falls back rather than raising:
        the knob tunes an advisory field, and failing a session's memory
        writes over it would be a worse outcome than ignoring it.  A
        deployment that mistypes it is told by ``jaato-scaffold validate``,
        which reads the declared type (#925).
    """
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return default


def duplicate_note(match: Optional[DuplicateMatch]) -> str:
    """Render the advisory sentence appended to ``store_memory``'s message.

    The prose matters as much as the field.  ``message`` is the **anchor**
    of ``store_memory``'s result — the wordiest string, which is what
    ``pick_anchor_field`` selects and what tool-result enrichment writes
    back to (#922) — so it is the part of the payload a model reads most
    reliably, and it is where a caller that reads no schema still sees this.

    Args:
        match: The near-duplicate found, or ``None``.

    Returns:
        The sentence to append, or ``""`` when there is nothing to say.
        Returning a string for the ``None`` case keeps the call site free
        of a branch it would otherwise need on every store.
    """
    if match is None:
        return ""
    where = (
        "you already stored this in THIS session"
        if match.source == "session"
        else "this is already in the curated store"
    )
    return (
        f" — NOTE: near-duplicate (similarity {match.similarity:.2f}) of "
        f"{match.memory_id}: {match.description!r}. You already know this: "
        f"{where}. Do not store it again; retrieve it instead."
    )


def duplicate_fields(match: Optional[DuplicateMatch]) -> dict:
    """Render the near-duplicate result fields, or nothing at all.

    Args:
        match: The near-duplicate found, or ``None``.

    Returns:
        ``{"duplicate_of", "duplicate_similarity", "duplicate_source"}`` when
        a match was found, else ``{}``.  The keys are **absent** rather than
        ``None`` on a clean store so that ``"duplicate_of" in result`` is the
        whole test, and so a store that duplicates nothing is byte-identical
        to the shape every existing caller already handles.
    """
    if match is None:
        return {}
    return {
        "duplicate_of": match.memory_id,
        "duplicate_similarity": match.similarity,
        "duplicate_source": match.source,
    }


def duplicate_telemetry(match: Optional[DuplicateMatch]) -> dict:
    """Render the near-duplicate span attributes for a store.

    Unlike :func:`duplicate_fields`, these keys are ALWAYS present: a span
    attribute that appears only on duplicate stores cannot be charted as a
    rate, and "no attribute" and "no duplicate" would be indistinguishable
    from a backend's side — the same failure this issue is about, one layer
    out.

    Args:
        match: The near-duplicate found, or ``None``.

    Returns:
        ``{"jaato.memory.duplicate_of", "jaato.memory.duplicate_similarity"}``
        with empty-string / ``0.0`` standing for "nothing resembled this".
    """
    return {
        "jaato.memory.duplicate_of": match.memory_id if match else "",
        "jaato.memory.duplicate_similarity": match.similarity if match else 0.0,
    }


def recent_store_window(memories: List, cap: int) -> List:
    """Trim this session's own-writes cache to its most recent ``cap``.

    Args:
        memories: The cache, oldest first.
        cap: Maximum entries to keep.

    Returns:
        The list to keep, newest-biased.  A cap exists because the cache is
        the only pool that grows without a curator draining it, and the
        plugin deliberately survives ``reset_for_next_session`` — so an
        uncapped list would grow for the life of a cascade.
    """
    if cap <= 0 or len(memories) <= cap:
        return memories
    return memories[-cap:]
