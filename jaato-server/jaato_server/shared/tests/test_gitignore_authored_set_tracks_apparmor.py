"""Guard: the ``.gitignore`` authored set and the AppArmor template agree.

The framework states which ``.jaato/`` subpaths are USER-AUTHORED in one
other place: the ``audit deny {workspace_path}/.jaato/<x> wlk`` rules in
``server/apparmor.py``, which stop a confined runner rewriting them.  The
``.gitignore`` block ``jaato-scaffold new`` writes re-includes the same
subpaths so they can be committed.  Two declarations of one fact drift
unless something compares them, so this guard reads the template's rules
out of its source and checks them against
``shared.scaffold.gitignore.AUTHORED`` in BOTH directions:

* every subpath the template write-denies is in the authored set, flagged
  ``confined=True`` — a new protected directory becomes committable the
  release it is protected, not when someone remembers;
* every entry flagged ``confined=True`` is in the template — the flag is a
  claim about the template, and a claim nobody checks is decoration.

Read from the template's SOURCE rather than by rendering a profile: the
question is what the author declared, and rendering would need a
workspace, a venv path and the fragment tiers to exist.
"""

from __future__ import annotations

import re
from pathlib import Path

from jaato_server.shared.scaffold import gitignore as G
from jaato_server.shared.tests.reversion import Reversion

REPO = Path(__file__).resolve().parents[4]
APPARMOR = REPO / "jaato-server" / "jaato_server" / "server" / "apparmor.py"

#: Put the defect back: drop one protected subpath from the authored set.
#: The block then ignores ``reactors.json`` — a file the template says a
#: confined runner may not even write — which is exactly the drift the
#: guard exists to notice.
REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/scaffold/gitignore.py",
        find='    AuthoredEntry("reactors.json", "reactor rules", confined=True),\n',
        replace='    # reactors.json: dropped\n',
        test="test_every_apparmor_write_denied_subpath_is_authored",
        because="an AppArmor-protected (user-authored) .jaato/ subpath that "
                "the .gitignore block would leave ignored",
    ),
]

_DENY = re.compile(r'audit deny "?\{workspace_path\}/\.jaato/(\S+?)"?\s+wlk,')


def _template_denied() -> set:
    """Every ``.jaato/`` subpath the template write-denies, verbatim."""
    return set(_DENY.findall(APPARMOR.read_text(encoding="utf-8")))


def _confined_state(raw: str) -> bool:
    """Whether a write-denied subpath is declared RUNTIME STATE
    (``G.CONFINED_STATE``) rather than an authored asset."""
    return any(raw.startswith(path) for path, _ in G.CONFINED_STATE)


def _template_authored() -> set:
    """The ``.jaato/`` subpaths the template write-denies, as AUTHORED
    spells them: a ``x/**`` or ``x/*/`` rule is the directory ``x/``, a
    file is itself.  Declared runtime state is left out: it is denied
    for a different reason and is never committed."""
    out = set()
    for raw in _template_denied():
        if _confined_state(raw):
            continue
        head = raw.split("/", 1)[0]
        out.add(head + "/" if "/" in raw else head)
    return out


def test_the_template_still_declares_authored_subpaths():
    """The regex must find the rules, or the guard below is vacuous."""
    found = _template_authored()
    assert {"profiles/", "agents/", "scripts/", "reactors.json"} <= found, found


def test_every_apparmor_write_denied_subpath_is_authored():
    authored = {e.path: e for e in G.AUTHORED}
    for sub in sorted(_template_authored()):
        assert sub in authored, (
            f"server/apparmor.py write-denies .jaato/{sub} for a confined "
            f"runner (user-authored config) but shared/scaffold/gitignore.py "
            f"does not re-include it — a workspace scaffolded today cannot "
            f"commit it.  Add AuthoredEntry({sub!r}, <what it holds>, "
            f"confined=True)")
        assert authored[sub].confined, (
            f".jaato/{sub} is write-denied by the template; mark its "
            f"AuthoredEntry confined=True")


def test_every_confined_flag_is_backed_by_the_template():
    template = _template_authored()
    for entry in G.AUTHORED:
        if entry.confined:
            assert entry.path in template, (
                f"AuthoredEntry({entry.path!r}) claims confined=True but "
                f"server/apparmor.py has no write-deny for it")


def test_authored_entries_are_well_formed():
    """Directories carry their trailing slash (so the rendered rule is
    directory-only, as git reads it); nothing is spelled with a leading
    ``.jaato/``; every entry says what it holds; no duplicates."""
    paths = [e.path for e in G.AUTHORED]
    assert len(paths) == len(set(paths))
    for e in G.AUTHORED:
        assert not e.path.startswith((".jaato", "/", "!"))
        assert e.why.strip(), e.path
        if "." not in e.path.rsplit("/", 1)[-1]:
            assert e.path.endswith("/"), f"{e.path}: a directory needs its /"


def test_no_authored_entry_is_a_credential_file():
    """``<provider>_auth.json`` is where a stored key lands.  The block's
    default direction (ignored unless named) is what keeps every such file
    out of git; naming one here would undo it."""
    for e in G.AUTHORED:
        assert not e.path.endswith("_auth.json"), e.path
        assert "token" not in e.path, e.path


def test_every_confined_state_entry_is_backed_by_the_template():
    """``CONFINED_STATE`` exempts subpaths from the check above, so an
    entry must name a deny the template really has.  An exemption nothing
    backs would be a hole in the guard."""
    denied = _template_denied()
    for path, why in G.CONFINED_STATE:
        assert why.strip(), path
        assert path.endswith("/"), f"{path}: a directory needs its /"
        assert any(raw.startswith(path) for raw in denied), (
            f"G.CONFINED_STATE names .jaato/{path} but server/apparmor.py "
            f"has no write-deny under it")


def test_confined_state_is_never_reincluded():
    """Runtime state stays ignored: a CONFINED_STATE entry must not be
    (or sit under) an AUTHORED entry, or the block would commit it."""
    for path, _ in G.CONFINED_STATE:
        for e in G.AUTHORED:
            assert not path.startswith(e.path), (path, e.path)
