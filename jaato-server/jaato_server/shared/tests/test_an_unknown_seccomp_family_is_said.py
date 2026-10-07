"""An unknown ``seccomp_allow`` family is said in the daemon log.

Allowing back a family that does not exist (a typo, or a name from a newer
release) leaves the filter whole, which is the safe side.  But the only
WARNING was the runner's, in the runner's log, and the daemon logged the
session's posture at INFO with no trace of the name: an operator who meant
to allow something back never learned it did not happen (SELinux run of
2026-10-05, ``--seccomp-allow nonsense``).

The posture now carries ``ignored_families``, and the daemon's posture line
says it at WARNING, naming the session and the known families.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from jaato_server.server import runner_spawn
from jaato_server.shared import seccomp_filter as sf
from jaato_server.shared.tests.reversion import Reversion

_SF = "jaato-server/jaato_server/shared/seccomp_filter.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"

REVERSIONS = [
    Reversion(
        target=_SF,
        find="        if self.ignored:\n            out[\"ignored_families\"] = list(self.ignored)\n",
        replace="",
        test="test_the_posture_names_the_ignored_families",
        because="the daemon is told only what was allowed back, never what was not",
    ),
    Reversion(
        target=_SPAWN,
        find="    if ignored:\n        logger.warning(\n",
        replace="    if False:\n        logger.warning(\n",
        test="test_the_daemon_log_warns_about_an_unknown_family",
        because="the posture carries the names and the daemon log stays INFO",
    ),
]


class _Compiled:
    instructions = 108
    libseccomp = "2.6.0"

    def install(self):  # pragma: no cover - never forked here
        pass


def test_the_posture_names_the_ignored_families(monkeypatch):
    monkeypatch.setattr(sf, "load_shipped", lambda shipped, allowed: _Compiled())
    plan = sf.plan_for_session(
        None, ["nonsense", "ptrace"], boundary_active=True,
        shipped={"stub": True}, required=False)
    posture = plan.as_dict()
    assert posture["posture"] == sf.POSTURE_FILTER
    assert posture["allowed_families"] == ["ptrace"]
    assert posture["ignored_families"] == ["nonsense"]


def test_an_absent_filter_still_names_them():
    plan = sf.plan_for_session(
        None, ["nonsense"], boundary_active=True, shipped=None, required=False)
    assert plan.as_dict()["ignored_families"] == ["nonsense"]


def test_a_known_family_adds_nothing(monkeypatch):
    monkeypatch.setattr(sf, "load_shipped", lambda shipped, allowed: _Compiled())
    plan = sf.plan_for_session(None, ["ptrace"], boundary_active=True,
                               shipped={"stub": True}, required=False)
    assert "ignored_families" not in plan.as_dict()


def _daemon_records(posture, caplog):
    server = SimpleNamespace(note_seccomp_posture=lambda _p: None)
    with caplog.at_level(logging.DEBUG, logger=runner_spawn.logger.name):
        runner_spawn._note_seccomp_posture(server, {"seccomp": posture}, "s-unk")
    return [r for r in caplog.records if r.name == runner_spawn.logger.name]


def test_the_daemon_log_warns_about_an_unknown_family(caplog):
    records = _daemon_records({"posture": "filter", "libseccomp": "2.6.0",
                               "ignored_families": ["nonsense"]}, caplog)
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert warnings, [(r.levelname, r.getMessage()) for r in records]
    assert "s-unk" in warnings[0] and "nonsense" in warnings[0]
    assert "ptrace" in warnings[0]  # the known families are listed


def test_a_clean_filter_stays_info(caplog):
    records = _daemon_records({"posture": "filter", "libseccomp": "2.6.0",
                               "allowed_families": ["ptrace"]}, caplog)
    assert records and all(r.levelno == logging.INFO for r in records)
