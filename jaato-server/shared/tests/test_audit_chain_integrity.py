"""Tamper evidence on the audit record -- Art. 73(6), #507, #1120.

Article 73(6) asks that, after a serious incident, the logs used in the
investigation not have been altered.  Every audit-bearing file here is an
append-only JSONL that anyone with write access can edit in place, and
nothing detected it.  #859 gave the record IDENTITY (who answered); this
is the INTEGRITY half.

Six properties, each attached to a way it could silently stop holding:

A. an in-place edit is DETECTED, and named by line;
B. an APPEND is not -- a chain that flagged its own growth would fire on
   every healthy run and stop being read;
C. `none` is byte-identical to before, so a profile that did not ask for
   this sees no change to its records;
D. **both write paths chain identically** -- `_append_to_disk` and
   `write_ledger` are two writers of one file, and `write_ledger` exists
   precisely to write what the append path did not;
E. the digest covers `prev_digest`, so a link cannot be re-pointed
   without breaking the record that carries it;
F. it says what it does NOT prove.  A chain proves the file was not
   edited in place; it does not prove who wrote it, and an investigator
   told otherwise draws a stronger conclusion than the evidence carries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_sdk.audit_chain import (
    DIGEST_FIELD,
    GENESIS,
    PREV_DIGEST_FIELD,
    canonical_bytes,
    chain,
    digest_of,
    verify,
)
from shared.plugins.subagent.config import RecordKeepingConfig
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
from shared.token_accounting import LEDGER_INTEGRITY_ENV, TokenLedger

_CHAIN = "jaato-sdk/jaato_sdk/audit_chain.py"
_LEDGER = "jaato-server/shared/token_accounting.py"
_DOCTOR = "jaato-sdk/jaato_sdk/doctor.py"

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/token_accounting.py",
        find="            self._prev_digest = self._tail_digest(path)",
        replace="            pass",
        because=(
            "the chain pointer held per instance instead of read off the "
            "file -- a restart or a second session sharing an absolute "
            "trace.ledger writes a genesis link mid-file, and the verifier "
            "reports an untouched audit log as tampered with"
        ),
        test="test_a_second_writer_continues_the_chain",
    ),
    Reversion(
        target=_CHAIN,
        find='    body = {k: v for k, v in record.items() if k != DIGEST_FIELD}',
        replace=('    body = {k: v for k, v in record.items()\n'
                 '            if k not in (DIGEST_FIELD, PREV_DIGEST_FIELD)}'),
        because=(
            "the digest must COVER prev_digest, or a record's link can be "
            "re-pointed at a different predecessor without breaking "
            "anything -- which is the whole property being bought"
        ),
        test="test_the_digest_covers_the_link",
    ),
    Reversion(
        target="jaato-server/shared/token_accounting.py",
        # Both paths go through ``_write_pending`` now, so the way to
        # separate them is to give ``write_ledger`` its own writer again
        # -- which is exactly the shape that used to break a file in the
        # middle, at the handover between the two.
        find="            self._write_pending(path, fsync=True)",
        replace=("            with open(path, \"a\", encoding=\"utf-8\") as f:\n"
                 "                for idx in range(self._flushed, len(self._events)):\n"
                 "                    f.write(json.dumps(\n"
                 "                        self._enrich(self._events[idx], idx)) + \"\\n\")\n"
                 "                f.flush()\n"
                 "            self._flushed = len(self._events)"),
        because=(
            "write_ledger flushes what the append path did not, so one "
            "file is written by both; if only one chains, the file breaks "
            "in the middle -- which reads exactly like tampering"
        ),
        test="test_both_write_paths_chain_identically",
    ),
    Reversion(
        target=_CHAIN,
        find='''        if declared is None or link is None:
            breaks.append(ChainBreak(''',
        replace='''        if False:
            breaks.append(ChainBreak(''',
        because=(
            "an UNCHAINED file must be reported as unchained, not as "
            "intact: the investigator asked whether it was tampered with, "
            "and for a file carrying no digests the true answer is that it "
            "is evidence of nothing either way"
        ),
        test="test_an_unchained_file_is_not_reported_as_intact",
    ),
]


_STAMP = {"stage": "response", "total_tokens": 10}


def _chained(records):
    """``records`` as chained JSONL lines."""
    prev = None
    lines = []
    for record in records:
        stamped = chain(record, prev)
        prev = stamped[DIGEST_FIELD]
        lines.append(json.dumps(stamped))
    return lines


# ----------------------------------------------------- A. an edit is seen

def test_an_in_place_edit_is_detected_and_named():
    lines = _chained([{"stage": "response", "total_tokens": n}
                      for n in (1, 2, 3)])
    assert verify(lines)[0] is True

    record = json.loads(lines[1])
    record["total_tokens"] = 9999
    lines[1] = json.dumps(record)

    intact, breaks = verify(lines)
    assert intact is False
    assert breaks[0].line == 2, "the break must name the LINE"
    assert "edited in place" in breaks[0].reason


def test_a_removed_record_breaks_the_link():
    lines = _chained([{"stage": "response", "n": n} for n in range(4)])
    del lines[1]
    intact, breaks = verify(lines)
    assert intact is False
    assert any("removed" in b.reason or "does not match" in b.reason
               for b in breaks)


def test_a_reordered_pair_breaks_the_link():
    lines = _chained([{"stage": "response", "n": n} for n in range(3)])
    lines[1], lines[2] = lines[2], lines[1]
    assert verify(lines)[0] is False


# ---------------------------------------------------- B. growth is not a break

def test_an_appended_record_is_not_a_break():
    """A chain that fired on its own growth would fire on every healthy
    run and stop being read."""
    lines = _chained([{"stage": "response", "n": 1}])
    tail = chain({"stage": "response", "n": 2}, json.loads(lines[0])[DIGEST_FIELD])
    assert verify(lines + [json.dumps(tail)])[0] is True


def test_a_trailing_newline_is_not_a_break():
    lines = _chained([{"stage": "response", "n": 1}]) + ["", "  "]
    assert verify(lines)[0] is True


# --------------------------------------------------------- C. none is inert

def test_none_writes_no_digest_fields(tmp_path, monkeypatch):
    monkeypatch.delenv(LEDGER_INTEGRITY_ENV, raising=False)
    target = tmp_path / "plain.jsonl"
    ledger = TokenLedger(path=str(target))
    ledger._record("response", {"total_tokens": 1})

    record = json.loads(target.read_text().strip())
    assert DIGEST_FIELD not in record
    assert PREV_DIGEST_FIELD not in record


def test_an_unrecognised_posture_is_none(tmp_path):
    # Anything but the one spelling is `none`: silently writing digests a
    # verifier does not expect is worse than not chaining, and an
    # unchained file at least SAYS it is unchained.
    target = tmp_path / "typo.jsonl"
    ledger = TokenLedger(path=str(target), integrity="sha256chain")
    ledger._record("response", {"total_tokens": 1})
    assert DIGEST_FIELD not in json.loads(target.read_text().strip())


def test_the_block_defaults_to_none():
    assert RecordKeepingConfig().chains is False
    assert RecordKeepingConfig(integrity="sha256-chain").chains is True


def test_the_posture_reaches_the_ledger_from_the_profile():
    """The env var is the TRANSPORT; `record_keeping.integrity` is home.

    The ledger is constructed before any profile is resolved, and the
    runner-side session reads the session-scoped context rather than the
    daemon object -- the same reason `trace.ledger` seeds `LEDGER_PATH`
    rather than being read directly.
    """
    import ast

    source = Path("jaato-server/server/core.py").read_text()
    assert "LEDGER_INTEGRITY_ENV" in source, (
        "nothing seeds the posture from the profile, so the block is inert")
    ast.parse(source)


# ------------------------------------------------- D. one file, two writers

def test_both_write_paths_chain_identically(tmp_path):
    """`_append_to_disk` and `write_ledger` write ONE file.

    `write_ledger` exists to flush what the append path did not, so a
    ledger whose path arrives late writes its first records through one
    and its later ones through the other.  If only one chains, the file
    breaks at the handover -- and it breaks in the middle, which reads
    exactly like tampering.
    """
    target = tmp_path / "mixed.jsonl"

    # No path yet: these stay in memory.
    ledger = TokenLedger(path="", integrity="sha256-chain")
    ledger._record("response", {"total_tokens": 1})
    ledger._record("response", {"total_tokens": 2})
    assert not target.exists()

    # write_ledger flushes them ...
    ledger.write_ledger(str(target))
    # ... and the append path continues the same chain.
    ledger._path = str(target)
    ledger._record("permission-check", {"tool": "x", "allowed": True})

    intact, breaks = verify(target.read_text().splitlines())
    assert intact is True, [b.reason for b in breaks]
    assert len(target.read_text().strip().splitlines()) == 3


def test_a_record_is_never_written_twice(tmp_path):
    target = tmp_path / "once.jsonl"
    ledger = TokenLedger(path=str(target), integrity="sha256-chain")
    ledger._record("response", {"total_tokens": 1})
    ledger.write_ledger()
    assert len(target.read_text().strip().splitlines()) == 1


def test_a_second_writer_continues_the_chain(tmp_path):
    """A restart, and a shared absolute ``trace.ledger``, are the SAME case.

    ``_prev_digest`` used to be per-instance and in memory only, so the
    second ``TokenLedger`` over one file wrote a record linked to
    ``genesis`` in the middle of it and ``verify`` reported an untouched
    file as tampered -- the mechanism accusing its own normal
    deployment.  Two sessions sharing one file is not an exotic
    configuration: it is what an ABSOLUTE ``trace.ledger`` means.

    The chain belongs to the file, so the link is read back from the
    file.
    """
    target = tmp_path / "shared.jsonl"
    first = TokenLedger(path=str(target), integrity="sha256-chain")
    first._record("response", {"total_tokens": 1})
    first._record("response", {"total_tokens": 2})

    second = TokenLedger(path=str(target), integrity="sha256-chain")
    second._record("response", {"total_tokens": 3})

    intact, breaks = verify(target.read_text().splitlines())
    assert intact is True, [b.reason for b in breaks]
    assert len(target.read_text().strip().splitlines()) == 3


def test_two_writers_interleaved_keep_one_chain(tmp_path):
    """Concurrent appenders serialise on the link rather than racing it.

    Each chained append takes the lock, re-reads the tail and writes --
    so records interleave in whatever order they arrive and every one of
    them links to the record actually before it.  Without the re-read,
    each writer chains to the last record IT wrote and every handover is
    a break.
    """
    target = tmp_path / "both.jsonl"
    a = TokenLedger(path=str(target), integrity="sha256-chain")
    b = TokenLedger(path=str(target), integrity="sha256-chain")
    for i in range(4):
        (a if i % 2 == 0 else b)._record("response", {"total_tokens": i})

    intact, breaks = verify(target.read_text().splitlines())
    assert intact is True, [b.reason for b in breaks]
    assert len(target.read_text().strip().splitlines()) == 4


def test_the_chain_still_detects_an_edit_after_a_handover(tmp_path):
    """Non-vacuity: reading the link off the file must not make verify blind.

    A fix that recovered the pointer by trusting whatever the file says
    would report every file as intact, which passes the two tests above
    for the wrong reason.
    """
    target = tmp_path / "edit.jsonl"
    TokenLedger(path=str(target), integrity="sha256-chain")._record(
        "response", {"total_tokens": 1})
    TokenLedger(path=str(target), integrity="sha256-chain")._record(
        "response", {"total_tokens": 2})

    lines = target.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["total_tokens"] = 999
    lines[0] = json.dumps(tampered)

    intact, breaks = verify(lines)
    assert intact is False
    assert any("edited in place" in b.reason for b in breaks)


def test_a_failed_write_does_not_duplicate_what_landed(tmp_path):
    """Cursors advance per line, after that line is flushed.

    Advancing for the whole batch meant a failure on line 3 of 5 left
    lines 1-2 on disk and UNRECORDED, so the retry wrote them again --
    with different digests, because the pointer had moved.  A duplicated
    record in an audit log is the thing the log exists to rule out.
    """
    target = tmp_path / "partial.jsonl"
    ledger = TokenLedger(path="", integrity="sha256-chain")
    for i in range(5):
        ledger._record("response", {"total_tokens": i})

    real_write = TokenLedger._flush_lines
    calls = {"n": 0}

    def _fail_on_the_third(self, path, *, fsync):
        class _Boom(Exception):
            pass

        with open(path, "a", encoding="utf-8") as f:
            for idx in range(self._flushed, len(self._events)):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise _Boom("disk full")
                f.write(self._line(idx) + "\n")
                f.flush()
                self._flushed = idx + 1

    ledger._path = str(target)
    TokenLedger._flush_lines = _fail_on_the_third
    try:
        ledger.write_ledger()
    finally:
        TokenLedger._flush_lines = real_write

    landed = len(target.read_text().strip().splitlines())
    assert landed == 2, "the two lines that reached disk should be recorded"
    assert ledger._flushed == 2

    # The retry writes only what is missing, and the file still verifies.
    ledger.write_ledger()
    lines = target.read_text().strip().splitlines()
    assert len(lines) == 5, "a retry must not rewrite what already landed"
    intact, breaks = verify(lines)
    assert intact is True, [b.reason for b in breaks]


def test_the_first_record_chains_to_a_named_genesis(tmp_path):
    # A literal rather than an empty string, so "segment start" and
    # "somebody deleted the field" are different states on disk.
    target = tmp_path / "g.jsonl"
    ledger = TokenLedger(path=str(target), integrity="sha256-chain")
    ledger._record("response", {"total_tokens": 1})
    assert json.loads(target.read_text().strip())[PREV_DIGEST_FIELD] == GENESIS


# ------------------------------------------------------ E. the link is covered

def test_the_digest_covers_the_link():
    record = chain(dict(_STAMP), "aaaa")
    repointed = dict(record)
    repointed[PREV_DIGEST_FIELD] = "bbbb"
    assert digest_of(repointed) != record[DIGEST_FIELD], (
        "re-pointing a record at another predecessor must break its digest")


def test_the_serialisation_is_canonical():
    # Two writers producing the same record must agree about its digest,
    # and they will not if the bytes carry dict ordering.
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})
    assert b" " not in canonical_bytes({"a": 1, "b": 2})


def test_a_record_does_not_contain_its_own_hash():
    record = chain(dict(_STAMP), None)
    assert canonical_bytes(record) == canonical_bytes(
        {k: v for k, v in record.items() if k != DIGEST_FIELD})


# --------------------------------------------- F. it says what it does not prove

def test_an_unchained_file_is_not_reported_as_intact():
    intact, breaks = verify(['{"stage": "response", "total_tokens": 1}'])
    assert intact is False
    assert "evidence of nothing either way" in breaks[0].reason


def test_an_empty_file_is_not_reported_as_intact():
    intact, breaks = verify([])
    assert intact is False
    assert "no records" in breaks[0].reason


def test_the_doctor_says_what_the_chain_does_not_prove(tmp_path):
    from jaato_sdk.doctor import PASS, WARN, check_audit_chain

    target = tmp_path / "ok.jsonl"
    target.write_text("\n".join(_chained([{"stage": "response"}])) + "\n")
    (check,) = check_audit_chain([str(target)])
    assert check.status == PASS
    assert "does NOT prove who wrote it" in check.detail, (
        "an investigator told a chain proves authorship draws a stronger "
        "conclusion than the evidence carries")

    record = json.loads(target.read_text().strip())
    record["stage"] = "tampered"
    target.write_text(json.dumps(record) + "\n")
    (broken,) = check_audit_chain([str(target)])
    assert broken.status == WARN, (
        "never FAIL: jaato-doctor is documented as usable as a CI gate, "
        "and a broken chain is a finding for a person, not a build error")


def test_the_doctor_reports_a_missing_file_rather_than_passing(tmp_path):
    from jaato_sdk.doctor import WARN, check_audit_chain

    (check,) = check_audit_chain([str(tmp_path / "absent.jsonl")])
    assert check.status == WARN
    assert "no such file" in check.detail
