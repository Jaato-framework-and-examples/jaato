"""Every ``JAATO_*`` env var the tree reads says what it is in ``explain env``.

``jaato-scaffold explain env`` takes a var's one-line description from an
``# env: ...`` comment trailing its read line
(``introspect._env_doc_comments``).  Nothing required the comment, so a
var added without one rendered as a bare name: the scaffold coverage
audit (docs/design/scaffold-coverage-audit.md) found
``JAATO_APPARMOR_PROFILE_GRACE_SECONDS`` (#1502) and
``JAATO_LOCK_HOLD_WARN_MS`` (#1454) in that state, beside fifteen older
ones.  ``env_scope.py`` held a note for each, which ``explain env`` shows
only for session-scoped vars without a typed key, so a host knob an
operator sets reached the listing with nothing beside it.

The rule is checked on the SCAN, the same source ``explain env`` renders,
so a var is described exactly when the listing describes it.  It covers
the framework's own names (``JAATO_*`` and ``LEDGER_PATH``); ambient
host variables (``PATH``, ``TERM``, ``HF_HOME`` ...) are the host's, not
knobs this tree defines.
"""

from __future__ import annotations

from jaato_server.shared.scaffold.introspect import env_vars
from jaato_server.shared.tests.reversion import Reversion

#: Framework-defined names outside the ``JAATO_`` prefix.
_FRAMEWORK_NAMES_WITHOUT_PREFIX = frozenset({"LEDGER_PATH"})


def _framework_owned(name: str) -> bool:
    return name.startswith("JAATO_") or name in _FRAMEWORK_NAMES_WITHOUT_PREFIX


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/lock_profile.py",
        find='os.environ.get("JAATO_LOCK_HOLD_WARN_MS", "").strip()  # env: ',
        replace='os.environ.get("JAATO_LOCK_HOLD_WARN_MS", "").strip()  # ',
        test="test_every_framework_env_var_has_a_description",
        because="the lock threshold reaches explain env as a bare name, "
                "the state #1454 shipped in",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/introspect.py",
        find="                    desc = doc_comments.get(lineno)\n",
        replace="                    desc = None\n",
        test="test_every_framework_env_var_has_a_description",
        because="the scan stops reading # env: comments, so every var in "
                "explain env loses its description at once",
    ),
]


def test_every_framework_env_var_has_a_description():
    missing = sorted(
        name for name, var in env_vars().items()
        if _framework_owned(name) and not var.description
    )
    assert not missing, (
        "these env vars render in `jaato-scaffold explain env` with no "
        "description; add a trailing `# env: <one line>` comment on the "
        "line that reads each (the comment must START the comment token, "
        f"and be on the read call's first line): {missing}"
    )


def test_the_scan_still_finds_framework_vars():
    """A scan that found nothing would satisfy the rule above vacuously."""
    owned = [n for n in env_vars() if _framework_owned(n)]
    assert len(owned) > 100, owned
    assert "JAATO_LOCK_HOLD_WARN_MS" in owned
