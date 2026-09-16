"""A warning nobody can clear is noise, and it grows per integration.

``check_integrations`` walks every integration this build ships and WARNs on
``absent``, with no test of whether that harness exists on the machine.  With
exactly one integration shipped that was invisible.  The moment a second one
exists, every user of the first is told to install a skill for a tool they do
not use, and the only way to satisfy it is to create that tool's config
directory.

Measured on a real machine before the fix -- one with Claude Code installed
and correctly integrated -- adding a second integration produced a PASS for
the one in use and a permanent WARN for the one that was not.

The fix reads a ``detect`` block the integration's own author declares, and
three properties are what make a heuristic acceptable here.  Each has a test
below, because each is a way the change could go wrong:

  * ``None`` != ``False``.  An integration declaring no ``detect`` asserted
    nothing, so it still warns.  Suppression follows an assertion only.
  * the skip is gated on ``absent``.  A copy that EXISTS is reported whatever
    detection says, so drift stays visible on a machine whose harness was
    removed -- detection can only ever withhold a suggestion.
  * an unusable manifest is reported, not raised.  ``jaato-doctor`` is what
    an operator runs when something is wrong; it must survive that.
"""
from __future__ import annotations

import json

import pytest

from jaato_sdk.doctor import PASS, WARN, check_integrations
from shared.scaffold import integrations as I


def _ship(root, name, manifest):
    d = root / name
    (d / "payload").mkdir(parents=True)
    (d / "payload" / "SKILL.md").write_text("payload\n", encoding="utf-8")
    (d / "integration.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture()
def shipped(tmp_path, monkeypatch):
    root = tmp_path / "integrations"
    root.mkdir()
    monkeypatch.setattr(I, "_source_root", lambda: root)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return root


def _named(checks, name):
    return [c for c in checks if c.name == f"integration ({name})"]


def test_an_absent_integration_for_a_harness_you_do_not_have_is_silent(shipped):
    """The defect: this WARNed, and nothing the operator did could clear it."""
    _ship(shipped, "notinstalled", {
        "tool": "NotInstalled", "target": ".ni/skills/jaato-sdk",
        "detect": {"commands": ["definitely-not-a-real-binary"],
                   "paths": ["/nonexistent/harness/state"],
                   "why": "test"}})
    assert I.harness_present("notinstalled") is False
    assert _named(check_integrations(), "notinstalled") == []


def test_an_absent_integration_for_a_harness_you_DO_have_still_nudges(shipped):
    """The suggestion is the feature; only the noise is the bug."""
    _ship(shipped, "installed", {
        "tool": "Installed", "target": ".inst/skills/jaato-sdk",
        "detect": {"commands": ["sh"], "why": "test"}})
    checks = _named(check_integrations(), "installed")
    assert [c.status for c in checks] == [WARN]
    assert "not applied" in checks[0].detail


def test_an_integration_declaring_no_detect_still_warns(shipped):
    """`None` is not `False`.  Absence of evidence is not evidence of absence,
    and an author who was not sure must not have their integration silently
    dropped from the doctor."""
    _ship(shipped, "silent", {"tool": "Silent",
                              "target": ".s/skills/jaato-sdk"})
    assert I.harness_present("silent") is None
    checks = _named(check_integrations(), "silent")
    assert [c.status for c in checks] == [WARN]


def test_drift_is_reported_even_when_the_harness_is_undetected(shipped):
    """The property that makes the heuristic safe.

    A copy on disk that has been edited is a real problem whatever detection
    says -- the harness may have been uninstalled after the skill was
    applied.  Gating the skip on ``absent`` is what preserves this.
    """
    _ship(shipped, "drifted", {
        "tool": "Drifted", "target": ".d/skills/jaato-sdk",
        "detect": {"commands": ["definitely-not-a-real-binary"],
                   "paths": ["/nonexistent/harness/state"], "why": "test"}})
    dest = I.target_dir("drifted", user=True, workspace=None)
    I.install("drifted", dest)
    (dest / "SKILL.md").write_text("locally edited\n", encoding="utf-8")

    assert I.harness_present("drifted") is False      # still undetected
    checks = _named(check_integrations(), "drifted")
    assert [c.status for c in checks] == [WARN]


def test_an_applied_integration_passes_whether_or_not_it_is_detected(shipped):
    _ship(shipped, "applied", {
        "tool": "Applied", "target": ".a/skills/jaato-sdk",
        "detect": {"commands": ["definitely-not-a-real-binary"], "why": "t"}})
    I.install("applied", I.target_dir("applied", user=True, workspace=None))
    checks = _named(check_integrations(), "applied")
    assert [c.status for c in checks] == [PASS]


def test_an_unusable_manifest_is_reported_not_raised(shipped):
    """jaato-doctor is what you run when something is wrong."""
    _ship(shipped, "broken", {"tool": "Broken"})      # no target
    checks = _named(check_integrations(), "broken")
    assert [c.status for c in checks] == [WARN]
    assert "cannot check" in checks[0].detail


# ---------------------------------------------------------------------------
# Reversions -- see test_every_guard_detects_its_own_reversion.
# ---------------------------------------------------------------------------
from shared.tests.test_every_guard_detects_its_own_reversion import (  # noqa: E402
    Reversion,
)

REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/doctor.py",
        find='        if state == "absent" and _install.harness_present(name) is False:\n',
        replace='        if False:\n',
        because=(
            "every shipped-but-unapplied integration warns again, so a user "
            "of one harness is permanently told to install a skill for a "
            "harness they do not have -- clearable only by creating that "
            "tool's config directory"
        ),
        test="test_an_absent_integration_for_a_harness_you_do_not_have_is_silent",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/doctor.py",
        find='        if state == "absent" and _install.harness_present(name) is False:\n',
        replace='        if _install.harness_present(name) is False:\n',
        because=(
            "dropping the `absent` gate lets detection suppress a copy that "
            "EXISTS, so an edited or stale skill goes unreported on a machine "
            "whose harness was uninstalled after it was applied"
        ),
        test="test_drift_is_reported_even_when_the_harness_is_undetected",
    ),
]
