"""The token ledger reaches disk -- per record, not at an exit nobody reached.

``TokenLedger`` records every model round trip (``response``: tokens, cost,
the user the session runs as) and every permission verdict
(``permission-check``: verdict, method, approver).  ``write_ledger()`` was
the only path to disk and had NO caller outside its tests, so on the daemon
path the ledger was an in-memory list that died with the runner -- a
``permission-check`` row carrying an approver existed nowhere after the
process exited.  Regulation (EU) 2024/1689 Arts. 12 and 19 ask for logs
that outlive the run; ``docs/design/eu-ai-act.md`` §4.4 names this as the
first thing to fix.

Pinned here:

A. a record is appended to disk the moment it is recorded, when a path is
   configured -- explicitly, or through ``LEDGER_PATH``, which the typed
   ``trace.ledger`` key now seeds; a relative path is one file per session
   (resolved against ``JAATO_WORKSPACE_ROOT``), the rule the two trace
   paths follow;
B. ``write_ledger`` flushes only what was not yet appended, so no record
   lands twice, and an explicit argument outranks the env var (the
   inversion the env catalog recorded);
C. a ledger that cannot be written never fails the round trip it records,
   and says so once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from shared.token_accounting import LEDGER_PATH_ENV, TokenLedger
from shared.tests.reversion import Reversion

_LEDGER = "jaato-server/shared/token_accounting.py"

REVERSIONS = [
    Reversion(
        target=_LEDGER,
        find="        self._events.append(details)\n        self._append_to_disk()",
        replace="        self._events.append(details)",
        because=(
            "a record must reach disk when it is recorded; a ledger flushed "
            "only at an exit nobody reaches is the in-memory list this fixes"
        ),
        test="test_a_record_reaches_disk_when_recorded",
    ),
]


def _lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------- A. append

def test_a_record_reaches_disk_when_recorded(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = TokenLedger(path=str(path))
    ledger._record("response", {"prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    rows = _lines(path)
    assert [r["stage"] for r in rows] == ["response"]
    assert rows[0]["event_index"] == 0
    assert rows[0]["iso_ts"].endswith("Z")
    assert rows[0]["internal_tokens"] == 0
    ledger._record("permission-check", {"tool": "writeNewFile", "allowed": False})
    assert [r["stage"] for r in _lines(path)] == ["response", "permission-check"]


def test_the_env_var_configures_it_and_a_relative_path_is_per_session(tmp_path, monkeypatch):
    monkeypatch.setenv(LEDGER_PATH_ENV, ".jaato/logs/ledger.jsonl")
    monkeypatch.setenv("JAATO_WORKSPACE_ROOT", str(tmp_path))
    ledger = TokenLedger()
    assert ledger.ledger_path() == str(tmp_path / ".jaato/logs/ledger.jsonl")
    ledger._record("response", {"total_tokens": 1})
    assert len(_lines(tmp_path / ".jaato/logs/ledger.jsonl")) == 1


def test_an_empty_env_value_means_no_ledger_file(monkeypatch):
    monkeypatch.setenv(LEDGER_PATH_ENV, "")
    assert TokenLedger().ledger_path() is None


def test_the_typed_key_seeds_the_env_var():
    from shared.plugins.subagent.config import TRACE_ENV_VARS, TraceProfileConfig
    assert TRACE_ENV_VARS["ledger"] == LEDGER_PATH_ENV
    cfg = TraceProfileConfig.from_dict({"ledger": ".jaato/logs/ledger.jsonl"})
    assert cfg.as_env() == {LEDGER_PATH_ENV: ".jaato/logs/ledger.jsonl"}


def test_the_typed_key_refuses_a_switch_like_its_siblings():
    import pytest
    from shared.plugins.subagent.config import TraceProfileConfig
    with pytest.raises(ValueError):
        TraceProfileConfig.from_dict({"ledger": "1"})


def test_the_env_catalog_records_the_typed_home():
    from shared.env_scope import AWAITING_TYPED_KEY, CATALOG
    assert CATALOG[LEDGER_PATH_ENV].typed_key == "trace.ledger"
    assert LEDGER_PATH_ENV not in AWAITING_TYPED_KEY


# ---------------------------------------------------------------- B. flush

def test_write_ledger_flushes_only_what_was_not_appended(tmp_path, monkeypatch):
    monkeypatch.delenv(LEDGER_PATH_ENV, raising=False)
    ledger = TokenLedger()                       # no path yet: in memory
    ledger._record("response", {"total_tokens": 1})
    ledger._record("response", {"total_tokens": 2})
    path = tmp_path / "late.jsonl"
    assert ledger.write_ledger(str(path)) == str(path)
    assert len(_lines(path)) == 2
    assert ledger.write_ledger(str(path)) == str(path)
    assert len(_lines(path)) == 2, "a second flush writes nothing twice"
    ledger._record("response", {"total_tokens": 3})
    ledger.write_ledger(str(path))
    assert [r["total_tokens"] for r in _lines(path)] == [1, 2, 3]


def test_a_ledger_that_appended_has_nothing_left_to_flush(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = TokenLedger(path=str(path))
    ledger._record("response", {"total_tokens": 1})
    ledger.write_ledger()
    assert len(_lines(path)) == 1


def test_an_explicit_argument_outranks_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(LEDGER_PATH_ENV, str(tmp_path / "env.jsonl"))
    ledger = TokenLedger(path=str(tmp_path / "arg.jsonl"))
    ledger._record("response", {"total_tokens": 1})
    assert (tmp_path / "arg.jsonl").exists()
    assert not (tmp_path / "env.jsonl").exists()


# ------------------------------------------------------------- C. failure

def test_an_unwritable_path_never_fails_the_record_and_is_reported_once(tmp_path, caplog):
    ledger = TokenLedger(path=str(tmp_path))     # a DIRECTORY, not a file
    with caplog.at_level(logging.WARNING, logger="shared.token_accounting"):
        ledger._record("response", {"total_tokens": 1})
        ledger._record("response", {"total_tokens": 2})
    warnings = [r for r in caplog.records if "cannot append" in r.getMessage()]
    assert len(warnings) == 1
    assert len(ledger.events()) == 2, "the records are kept in memory"
