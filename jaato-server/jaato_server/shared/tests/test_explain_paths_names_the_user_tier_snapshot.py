"""``explain paths`` says that a runner reads ``~/.jaato`` from a snapshot.

#1465 moved the user tier onto the session envelope: the daemon reads
``permissions.json``, ``system_instructions.md`` and the rest at spawn,
and the runner reads that copy.  ``explain paths`` -- the topic an author
reads to learn where a file is read from -- kept describing ``~/.jaato``
as read live, so an edit made mid-session looked like a bug instead of a
next-session change.  The scaffold coverage audit
(docs/design/scaffold-coverage-audit.md) found it; the lines are derived
from ``shared/user_tier.py``, and this holds that they are rendered.
"""

from __future__ import annotations

from jaato_server.shared import user_tier
from jaato_server.shared.scaffold import explain
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/explain.py",
        find="        *_user_tier_snapshot_lines(),\n",
        replace="",
        test="test_paths_names_every_shipped_user_tier_file",
        because="explain paths goes back to describing ~/.jaato as read live",
    ),
]


def _text() -> str:
    _data, text = explain.paths()
    return " ".join(text.split())


def test_paths_names_every_shipped_user_tier_file():
    text = _text()
    assert "SNAPSHOT" in text
    missing = [n for n in (*user_tier.SHIPPED_FILES, *user_tier.SHIPPED_DIRS)
               if n not in text]
    assert not missing, missing


def test_paths_says_an_edit_reaches_the_next_session():
    assert "NEXT session" in _text()
