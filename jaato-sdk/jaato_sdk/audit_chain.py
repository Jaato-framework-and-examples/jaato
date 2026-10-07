"""Tamper evidence on the audit record -- Art. 73(6), #507, #1120.

Article 73(6) asks that, after a serious incident, the logs used in the
investigation not have been altered; #507 asks for tamper evidence on the
permission record for the framework's own reasons.  Every audit-bearing
file here is an append-only JSONL that anyone with write access can edit
in place, and nothing detects it.  #859 gave the record IDENTITY (who
answered); this is the INTEGRITY half.

``record_keeping.integrity: sha256-chain`` turns it on.  Each record then
carries ``prev_digest`` -- the SHA-256 of the previous record's canonical
bytes -- and ``digest``, its own; the first record of a FILE chains to
:data:`GENESIS`.

**What it proves, and what it does not.**  It proves the file was not
edited in place after the fact.  It does **not** prove who wrote it: a
writer holding the file can re-chain from any point, and nothing here
stops them.  That distinction is exactly what an investigator needs to
know, so it is stated in the docstring, in ``explain audit``'s output and
in ``docs/audit-log.md`` rather than left for somebody to infer from the
absence of a signature.  Signing -- a key the daemon holds -- is the next
step and is deliberately not this module.

**Stated cost, from the design doc**: a chained file cannot be pruned
from the front, because removing a line breaks every link after it.
Retention therefore rotates whole SEGMENTS -- a new file per period, each
with its own genesis -- rather than deleting lines.

Stdlib only, and in the SDK, so a file can be verified by somebody who
has the file and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: What the first record of a FILE chains to.  A literal rather than
#: an empty string, so "this is the start of the chain" and "somebody deleted
#: the field" are different states on disk.
GENESIS = "genesis"

#: The two fields a chained record carries.  Named, because
#: :func:`canonical_bytes` must exclude ``digest`` (a record cannot
#: contain its own hash) and INCLUDE ``prev_digest`` (or the link is not
#: covered by the hash and can be rewritten freely).
DIGEST_FIELD = "digest"
PREV_DIGEST_FIELD = "prev_digest"


def canonical_bytes(record: Dict[str, Any]) -> bytes:
    """The bytes a record's digest is taken over.

    Sorted keys and a fixed separator set, because two writers producing
    the same record must agree about its digest -- and they will not if
    the serialisation carries dict ordering or whitespace.

    ``digest`` is excluded (a record cannot contain its own hash) and
    ``prev_digest`` is INCLUDED: leaving the link outside the hash would
    let anyone re-point a record at a different predecessor without
    breaking anything, which is the whole property being bought.

    Values that JSON cannot represent are stringified rather than
    raising: a digest is a fact about what was written, and the writer
    already wrote it.
    """
    body = {k: v for k, v in record.items() if k != DIGEST_FIELD}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def digest_of(record: Dict[str, Any]) -> str:
    """The SHA-256 of ``record``'s canonical bytes, as hex."""
    return hashlib.sha256(canonical_bytes(record)).hexdigest()


def chain(record: Dict[str, Any], prev_digest: Optional[str]) -> Dict[str, Any]:
    """Return ``record`` with its two chain fields stamped.

    Args:
        record: The record as its writer built it.
        prev_digest: The previous record's ``digest``, or ``None`` for
            the first record of a file (which chains to
            :data:`GENESIS`).

    Returns:
        A NEW dict.  The caller's record is not mutated, so a ledger's
        in-memory event list keeps the shape its own consumers read and
        the chain exists only in what reaches disk.
    """
    stamped = dict(record)
    stamped[PREV_DIGEST_FIELD] = prev_digest or GENESIS
    stamped[DIGEST_FIELD] = digest_of(stamped)
    return stamped


@dataclass(frozen=True)
class ChainBreak:
    """Where a chain stopped holding.

    Attributes:
        line: 1-based line number in the file.
        reason: What was wrong, in an investigator's vocabulary.
        record: The record as read, when it parsed.
    """

    line: int
    reason: str
    record: Optional[Dict[str, Any]] = None


def verify(lines: Iterable[str]) -> Tuple[bool, List[ChainBreak]]:
    """Walk a chained JSONL file and report the FIRST break and any after.

    Args:
        lines: The file's lines, in order.  Blank lines are skipped --
            a trailing newline is not a break.

    Returns:
        ``(intact, breaks)``.  ``intact`` is ``True`` only when every
        record carried both fields, each ``prev_digest`` matched its
        predecessor's ``digest``, and each ``digest`` matched the record.

    **An unchained file is reported as unchained, not as intact.**  A
    verifier that returned "fine" for a file carrying no digests would
    answer the investigator's question wrongly: they asked whether this
    file was tampered with, and the true answer is that it carries no
    evidence either way.
    """
    breaks: List[ChainBreak] = []
    previous: Optional[str] = None
    seen = 0

    for number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text:
            continue
        seen += 1
        try:
            record = json.loads(text)
        except ValueError:
            breaks.append(ChainBreak(number, "not valid JSON"))
            previous = None
            continue
        if not isinstance(record, dict):
            breaks.append(ChainBreak(number, "not a JSON object"))
            previous = None
            continue

        declared = record.get(DIGEST_FIELD)
        link = record.get(PREV_DIGEST_FIELD)
        if declared is None or link is None:
            breaks.append(ChainBreak(
                number,
                "carries no chain fields -- this file is not chained, so it "
                "is evidence of nothing either way",
                record))
            previous = None
            continue

        expected_link = previous if previous is not None else GENESIS
        if previous is not None and link != expected_link:
            breaks.append(ChainBreak(
                number,
                f"prev_digest {link[:12]}... does not match the previous "
                f"record's digest {expected_link[:12]}... -- a record before "
                f"this one was edited, removed or inserted",
                record))
        recomputed = digest_of(record)
        if recomputed != declared:
            breaks.append(ChainBreak(
                number,
                f"digest {declared[:12]}... does not match the record's own "
                f"content (recomputed {recomputed[:12]}...) -- THIS record "
                f"was edited in place",
                record))
        previous = declared

    if seen == 0:
        return False, [ChainBreak(0, "the file holds no records")]
    return not breaks, breaks
