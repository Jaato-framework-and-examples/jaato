"""The results file as a CONTRACT, not as whatever ``ArmResult`` happens to hold.

A results file is read by things outside this package — today the dossier
generator in ``jaato-server/shared/scaffold/eval_results.py``, which renders an
arm's numbers into the accuracy section of an Annex IV dossier (jaato #1124).
That reader may **not** import this engine: ``jaato_eval`` imports ``jaato_sdk``
and nothing else from the tree, and a consumer that imported the producer would
make the rule run backwards.  So the two sides are joined by a declared format
rather than by a shared object, and this module is the producer's half of it.
``docs/eval-results.md`` is the declaration both halves are written against.

Two fields exist for the reader and for nobody in this package:

``results_version``
    Stamped on every record.  A reader that cannot recognise it must refuse the
    file **by name** rather than render what it can make sense of — a dossier
    filled from a format the reader guessed at is the failure this field exists
    to prevent, and it is worse than an empty section because it looks complete.

``caveats``
    The limits of the instruments that produced THIS arm's verdicts, in the
    words of the harness that owns them.  A consumer renders them verbatim.

WHY THE CAVEAT TRAVELS IN THE FILE
==================================

The alternative is for each consumer to know that ``judge`` means "an LLM
scored this" and to write its own warning.  That is a second copy of a fact,
and the copy that rots is the one nothing executes: a consumer's hand-written
caveat outlives the limit it describes and goes on being quoted after the limit
is fixed, or misses a limit added later.  The harness measured the limit, so
the harness states it, once, and it rides beside the number it qualifies.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

#: The declared shape of a results record.  Bump when a reader that only knows
#: the previous version could be MISLED by a file written under the new one —
#: not for an added field a reader ignores.  ``docs/eval-results.md`` records
#: what each version guarantees.
RESULTS_FORMAT_VERSION = "1"

#: What each grader kind cannot tell you, in the harness's own words.
#:
#: Keyed by the ``kind`` half of a ``grader_id`` (``kind:identifier``).  A kind
#: absent from this map asserts nothing — absence is "this harness has not
#: measured a limit worth stating", never "this instrument is calibrated".
GRADER_CAVEATS: Dict[str, str] = {
    "judge": (
        "The `judge` grader is one language model scoring another's output, "
        "and it is not a calibrated instrument: before the mandatory-prefetch "
        "fix it skipped its own verification step in roughly one run in four, "
        "and nothing in this harness establishes that the remaining error rate "
        "is zero. Read a judge-graded number as an indication, never as a "
        "measurement, and never quote one without this sentence."
    ),
}


def caveats_for(grader_ids: Iterable[str]) -> List[str]:
    """Every caveat applying to the graders that produced one arm.

    De-duplicated and ordered by first appearance, so two ``judge`` rubrics on
    one arm state the judge's limit once and a reader's diff stays stable.
    """
    out: List[str] = []
    for grader_id in grader_ids:
        kind = str(grader_id).split(":", 1)[0]
        caveat = GRADER_CAVEATS.get(kind)
        if caveat and caveat not in out:
            out.append(caveat)
    return out
