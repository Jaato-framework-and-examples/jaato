"""``explain paths`` names the WORKSPACE reactor tier, and derives it.

Three surfaces of this one package already knew that reactor rules are a
workspace-tier authored asset:

* ``shared/scaffold/gitignore.py`` — ``AUTHORED`` carries
  ``reactors.json`` with ``confined=True``, so ``new gitignore`` re-includes
  it and ``validate`` checks that it is committable;
* ``server/apparmor.py`` — the template ``audit deny``s
  ``{workspace_path}/.jaato/reactors.json`` to the confined runner, which is
  a rule about a file that only exists because the workspace tier does;
* the reactor engine itself, which loads ``<workspace>/.jaato/reactors.json``
  on every session.

``explain paths`` — the one surface an agent reads before it writes a file —
named the HOME copy alone.  The reported cost is exact: a session asked to add
a reactor was told the declaration file lives at ``~/.jaato/reactors/``, could
not write there (the daemon's home is not the reader's on a service install),
and had nothing to tell it a workspace tier existed.  It went looking in the
source.

So this guards the agreement rather than the sentence: ``paths`` READS
``AUTHORED``, which is why it cannot fall behind it again.
"""

import re
from pathlib import Path

from jaato_server.shared.scaffold import explain as _explain
from jaato_server.shared.scaffold import gitignore as _gi
from jaato_server.shared.tests.reversion import Reversion

_EXPLAIN = "jaato-server/jaato_server/shared/scaffold/explain.py"
_APPARMOR_PY = (Path(__file__).resolve().parents[2] / "server" / "apparmor.py")

REVERSIONS = [
    Reversion(
        target=_EXPLAIN,
        find='''        f"    {path:<36} {why}"
        for path, why in _authored_workspace_paths("reactors.json")''',
        replace='''        f"    {path:<36} {why}"
        for path, why in []''',
        test="test_explain_paths_names_the_workspace_reactor_tier",
        because="`explain paths` names the HOME reactor tier alone again, so "
                "a reader is sent to a path the daemon's user owns and told "
                "nothing about the one they can write",
    ),
    Reversion(
        target=_EXPLAIN,
        find='''    from . import gitignore as _gi
    by_path = {e.path: e for e in _gi.AUTHORED}
    return [(f".jaato/{n}", by_path[n].why) for n in names if n in by_path]''',
        replace='''    return [(".jaato/reactors.json", "reactor rules")]''',
        test="test_the_workspace_rows_are_read_from_the_authored_set",
        because="the workspace rows are a second hand-typed copy again, so "
                "`paths` and the AUTHORED set can drift the way they did",
    ),
]


def _paths_text() -> str:
    return _explain.paths()[1]


def test_explain_paths_names_the_workspace_reactor_tier():
    """The reported blocker: the tier a reader can actually write."""
    text = _paths_text()
    assert ".jaato/reactors.json" in text, (
        "`explain paths` does not name <workspace>/.jaato/reactors.json -- "
        "the AUTHORED set and the AppArmor template both declare it, and this "
        "is the surface an agent reads before it writes the file."
    )
    # ... in the PER-SESSION block, not only in the daemon-global one, which
    # already named a ~/.jaato reactor path and was the whole confusion.
    per_session = text.split("<workspace>/   — PER-SESSION", 1)
    assert len(per_session) == 2, "the per-session block is gone from `paths`"
    assert ".jaato/reactors.json" in per_session[1]


def test_the_workspace_rows_are_read_from_the_authored_set(monkeypatch):
    """Derived, not re-typed: a new AUTHORED entry is renderable at once."""
    sentinel = _gi.AuthoredEntry("guard-probe.json", "a probe entry",
                                 confined=True)
    monkeypatch.setattr(_gi, "AUTHORED", _gi.AUTHORED + (sentinel,))
    rows = _explain._authored_workspace_paths("guard-probe.json")
    assert rows == [(".jaato/guard-probe.json", "a probe entry")], (
        "_authored_workspace_paths does not read gitignore.AUTHORED -- it is "
        "a second declaration, so `paths` can fall behind it again."
    )


def test_a_name_the_authored_set_does_not_carry_is_dropped_not_guessed():
    """Absence renders as nothing, never as a plausible invented row."""
    assert _explain._authored_workspace_paths("no-such-file.json") == []


def test_the_reactor_row_is_in_the_json_too():
    """An agent reads `--json`; a fix only in the prose is half a fix."""
    data = _explain.paths()[0]
    assert ".jaato/reactors.json" in data["per_session"]["holds"]


def test_the_home_tier_names_both_spellings_the_engine_reads():
    """``~/.jaato/reactors.json`` was missing beside ``reactors/<name>.json``.

    The engine's home tier is a fragments DIRECTORY and a single FILE, merged
    in that order.  `paths` named the directory alone, so the file an operator
    is most likely to hand-edit -- and the only one that is hot-reloaded --
    appeared in no jaato surface at all.
    """
    holds = _explain.paths()[0]["daemon_global"]["holds"]
    assert "reactors.json" in holds
    assert "reactors/<name>.json" in holds


def test_the_apparmor_template_still_declares_the_file_paths_now_names():
    """The cross-surface agreement, asserted in the direction that can rot.

    If the template stops denying it, this row in `paths` is describing a
    workspace file nothing protects -- worth knowing at the moment it changes,
    rather than the next time someone reads the two and disagrees.
    """
    template = _APPARMOR_PY.read_text(encoding="utf-8")
    assert re.search(r"\{workspace_path\}/\.jaato/reactors\.json", template), (
        "server/apparmor.py no longer names {workspace_path}/.jaato/"
        "reactors.json; `explain paths` and gitignore.AUTHORED still do."
    )
    assert any(e.path == "reactors.json" for e in _gi.AUTHORED), (
        "gitignore.AUTHORED no longer carries reactors.json; `explain paths` "
        "derives its workspace row from it and would silently lose the row."
    )
