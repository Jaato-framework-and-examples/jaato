"""One target key, and a harness the manifest's author is sure about.

Two changes, and they are independent of each other except that the second
is only safe once the first has settled the schema.

ONE KEY, TWO FORMS.  ``target`` is a string when a harness uses one relative
path at both scopes, or ``{"user": ..., "workspace": ...}`` when they differ.
The rejected alternative was a ``target`` key plus ``user_target`` /
``workspace_target`` companions, which costs three things: a reader has to
know which of the three are alternatives and which are siblings, every
``listing()`` row carries two nulls whose position flips per integration, and
-- the sharp one -- an author who declares one scope and forgets the other got
a SILENTLY wrong path for the missing one.  Measured before this change: a
manifest declaring only ``user_target`` resolved its workspace install to
``/ws/.jaato-integration-<name>`` and installed a real payload there.

That fallback claimed in its own docstring that "the caller reports it".  No
caller did.  The string ``.jaato-integration-`` occurred exactly once in the
tree -- at the site that built it -- with no reader anywhere.  So it is gone,
and an unusable manifest raises.

DETECT.  ``jaato-doctor`` warned once per shipped-but-unapplied integration
with no test of whether that harness exists on the machine.  With one
integration shipped that was invisible; the moment a second exists, every
user of the first gets a warning they cannot clear -- the only way to satisfy
it is to install a skill for a tool they do not use.

The fix is a manifest key, because only the author of an integration knows
how to tell their harness is installed.  Three properties carry it:

  * ``None`` (no ``detect`` declared) is NOT ``False``.  It means "nobody
    asserted anything", and those still warn exactly as before.  Suppression
    only ever follows an assertion.
  * the skip is gated on ``state == "absent"``, so the most a heuristic may
    do is withhold an optional suggestion.  A copy that EXISTS is reported
    whatever detection says.
  * a ``detect.paths`` entry that is an ancestor of the integration's own
    target is refused, because jaato creates those.  Measured: on a host with
    neither harness, installing only our skills brings ``~/.claude``,
    ``~/.claude/skills``, ``~/.pi`` and ``~/.pi/agent`` into existence.  That
    is the one detect error a machine can check; the rest rests on the
    author's knowledge and is reviewed by reading ``detect.why``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_server.shared.scaffold import integrations as I


def _ship(root: Path, name: str, manifest: dict) -> None:
    """Make ``name`` a shipped integration under a fake source root."""
    d = root / name
    (d / "payload").mkdir(parents=True)
    (d / "payload" / "SKILL.md").write_text("payload\n", encoding="utf-8")
    (d / "integration.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture()
def shipped(tmp_path, monkeypatch):
    """A source root this test owns, so manifests can be malformed on purpose."""
    root = tmp_path / "integrations"
    root.mkdir()
    monkeypatch.setattr(I, "_source_root", lambda: root)
    return root


# --------------------------------------------------------------- one key

def test_a_string_target_serves_both_scopes(shipped):
    """The common case stays a one-liner -- and claude-code must not move."""
    _ship(shipped, "same", {"tool": "S", "target": ".x/skills/jaato-sdk"})
    assert I.resolve_targets("same") == (".x/skills/jaato-sdk",
                                         ".x/skills/jaato-sdk")


def test_an_object_target_serves_each_scope(shipped):
    """The case Pi actually has: ~/.pi/agent/skills vs <ws>/.pi/skills."""
    _ship(shipped, "split", {"tool": "S", "target": {
        "user": ".pi/agent/skills/jaato-sdk",
        "workspace": ".pi/skills/jaato-sdk"}})
    assert I.resolve_targets("split") == (".pi/agent/skills/jaato-sdk",
                                          ".pi/skills/jaato-sdk")


def test_the_real_claude_code_manifest_still_resolves_where_it_always_did():
    """Not a tautology: this is the regression the schema change risks.

    Uses the SHIPPED manifest, not a fixture, because what must not move is
    the path on real machines that already have a copy installed there.
    """
    assert I.resolve_targets("claude-code") == (".claude/skills/jaato-sdk",
                                                ".claude/skills/jaato-sdk")
    assert I.target_dir("claude-code", user=True, workspace=None) == \
        Path.home() / ".claude/skills/jaato-sdk"
    assert I.target_dir("claude-code", user=False, workspace="/ws") == \
        Path("/ws/.claude/skills/jaato-sdk")


def test_a_half_declared_object_target_is_refused_not_guessed(shipped):
    """THE defect this schema replaces.

    Before: this installed to ``/ws/.jaato-integration-half`` and said
    nothing.  A wrong path that looks plausible is worse than an error.
    """
    _ship(shipped, "half", {"tool": "H", "target": {"user": ".h/skills/x"}})
    with pytest.raises(I.IntegrationManifestError) as e:
        I.resolve_targets("half")
    assert "workspace" in str(e.value)


def test_a_missing_target_is_refused(shipped):
    _ship(shipped, "none", {"tool": "N"})
    with pytest.raises(I.IntegrationManifestError):
        I.resolve_targets("none")


def test_an_absolute_target_is_refused(shipped):
    """Targets are joined onto $HOME or the workspace, so an absolute path
    would escape the scope the operator asked for."""
    _ship(shipped, "abs", {"tool": "A", "target": "/etc/skills/x"})
    with pytest.raises(I.IntegrationManifestError):
        I.resolve_targets("abs")


def test_nothing_constructs_the_guessed_fallback_path_any_more():
    """It had no reader, and its docstring claimed a caller reported it.

    Checks for the CONSTRUCTION rather than the mention: the prose above
    names the old path deliberately, and a test that forbade the name would
    forbid explaining why it went.
    """
    src = Path(I.__file__).read_text(encoding="utf-8")
    assert 'f".jaato-integration-' not in src
    assert "f'.jaato-integration-" not in src


def test_one_unusable_manifest_does_not_take_down_the_listing(shipped):
    """`integration` with no name is how an operator FINDS OUT something is
    wrong, so it must survive the thing being wrong."""
    _ship(shipped, "good", {"tool": "G", "target": ".g/skills/x"})
    _ship(shipped, "broken", {"tool": "B"})
    data, text = I.listing()
    states = {r["name"]: r["state"] for r in data["integrations"]}
    assert states["good"] != "invalid"
    assert states["broken"] == "invalid"
    assert "broken" in text


def test_listing_rows_carry_one_shape_with_no_nulls(shipped):
    """The companion-keys schema emitted three keys, two null per row, and
    which two flipped per integration -- so every consumer handled both."""
    _ship(shipped, "same", {"tool": "S", "target": ".x/skills/y"})
    _ship(shipped, "split", {"tool": "S", "target": {
        "user": ".p/agent/skills/y", "workspace": ".p/skills/y"}})
    rows = {r["name"]: r for r in I.listing()[0]["integrations"]}
    for name in ("same", "split"):
        assert rows[name]["user_target"] and rows[name]["workspace_target"]
        assert "target" not in rows[name]


# ----------------------------------------------------------------- detect

def test_no_detect_means_unknown_not_absent(shipped):
    """The load-bearing distinction: suppression follows an ASSERTION."""
    _ship(shipped, "quiet", {"tool": "Q", "target": ".q/skills/x"})
    assert I.harness_present("quiet") is None


def test_an_empty_detect_is_also_unknown(shipped):
    _ship(shipped, "empty", {"tool": "E", "target": ".e/skills/x",
                             "detect": {}})
    assert I.harness_present("empty") is None


def test_detect_is_false_when_nothing_it_names_is_there(shipped, monkeypatch):
    _ship(shipped, "gone", {"tool": "G", "target": ".g/skills/x",
                            "detect": {"commands": ["definitely-not-a-binary"],
                                       "paths": ["/nonexistent/harness"]}})
    assert I.harness_present("gone") is False


def test_detect_is_true_on_a_path_the_harness_owns(shipped, tmp_path):
    marker = tmp_path / "harness-state"
    marker.mkdir()
    _ship(shipped, "here", {"tool": "H", "target": ".h/skills/x",
                            "detect": {"paths": [str(marker)]}})
    assert I.harness_present("here") is True


def test_detect_is_true_on_a_command(shipped):
    _ship(shipped, "cmd", {"tool": "C", "target": ".c/skills/x",
                           "detect": {"commands": ["sh"]}})
    assert I.harness_present("cmd") is True


def test_a_detect_path_we_create_ourselves_is_refused(shipped):
    """The mistake this guard exists for, and the one I made first.

    ``~/.pi`` looks like evidence of Pi and is created by installing OUR
    skill under ``~/.pi/agent/skills``.  It would answer "the harness is
    here" on a machine that has never had it -- silently restoring the noise
    the key exists to remove.
    """
    _ship(shipped, "self", {"tool": "S", "target": ".s/agent/skills/jaato-sdk",
                            "detect": {"paths": ["~/.s"]}})
    problems = I.manifest_detect_problems("self")
    assert problems and "ancestor" in problems[0]


def test_a_harness_owned_path_passes_the_same_guard(shipped):
    _ship(shipped, "ok", {"tool": "O", "target": ".o/skills/jaato-sdk",
                          "detect": {"paths": ["~/.o/projects"]}})
    assert I.manifest_detect_problems("ok") == []


def test_every_shipped_manifest_is_usable_and_its_detect_is_sound():
    """The guard that binds the two halves to what we actually ship."""
    for name in I.available():
        I.resolve_targets(name)                       # raises if unusable
        assert I.manifest_detect_problems(name) == [], name


def test_a_shipped_detect_explains_itself():
    """`why` is how a reviewer who does not use the harness can judge the
    claim -- the position we are in for any integration but our own."""
    for name in I.available():
        detect = I.manifest(name).get("detect")
        if detect:
            assert detect.get("why"), f"{name} declares detect with no why"


# ---------------------------------------------------------------------------
# Reversions -- the meta-suite (test_every_guard_detects_its_own_reversion)
# discovers this list by name and asserts each makes the NAMED test fail.
# `test` is the nodeid WITHIN this module and `replace` must still COMPILE.
# ---------------------------------------------------------------------------
from jaato_server.shared.tests.reversion import (  # noqa: E402
    Reversion,
)

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/integrations.py",
        find='        missing = [k for k in ("user", "workspace") if not target.get(k)]\n',
        replace='        missing = []\n',
        because=(
            "the half-declared object target stops being refused and "
            "resolves the missing scope to a plausible wrong path again -- "
            "the silent install this schema replaced"
        ),
        test="test_a_half_declared_object_target_is_refused_not_guessed",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/integrations.py",
        find='    detect = manifest(name).get("detect")\n'
             "    if not isinstance(detect, dict):\n"
             "        return None\n",
        replace='    detect = manifest(name).get("detect")\n'
                "    if not isinstance(detect, dict):\n"
                "        return False\n",
        because=(
            "'nobody declared detect' collapses into 'the harness is "
            "absent', so an integration that asserted nothing is silently "
            "dropped from jaato-doctor -- absence of evidence read as "
            "evidence of absence"
        ),
        test="test_no_detect_means_unknown_not_absent",
    ),
]
