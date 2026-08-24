"""``jaato-scaffold new`` must not blame the generator for USER-tier errors.

``validate_workspace`` reports over the MERGED profile tree: the workspace set
just generated AND the inherited user tier (``~/.jaato/profiles``).  ``new``
counted every error and then printed

    ✘ scaffold emitted 1 error(s) — this is a generator bug; please report.

on a CLEAN generation, whenever the developer had any broken profile in their
home directory.  Maximally misleading: emphatic, asks for a report, and sends
the reader into the scaffold templates hunting a plugin reference that lives in
$HOME.  ``validate`` already labels findings ``[workspace]`` / ``[user]``;
``new`` never used it.

Reported from the cascade-coordination example, 2026-08-24.
"""
from types import SimpleNamespace

import pytest

from shared.scaffold.build import _report_revalidation


def _d(tier, severity="error", profile="p", code="unknown_plugin"):
    return SimpleNamespace(tier=tier, severity=severity, profile=profile,
                           code=code, message="plugin 'x' is not installed",
                           where=None)


def test_a_user_tier_error_does_not_convict_the_generator(capsys):
    rc = _report_revalidation([_d("user", profile="gen-references")])
    out = capsys.readouterr().out
    assert rc == 0, (
        "a broken profile in the developer's ~/.jaato failed the scaffold and "
        "was reported as a generator bug")
    assert "generator" not in out
    assert "valid by construction" in out


def test_a_workspace_tier_error_still_convicts(capsys):
    rc = _report_revalidation([_d("workspace")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "generator bug" in out
    assert "in the generated set" in out, "the verdict must say what it counted"


def test_a_mixed_run_counts_only_the_workspace_error(capsys):
    rc = _report_revalidation([_d("user"), _d("workspace"), _d("user")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "1 error(s) in the generated set" in out, (
        f"counted user-tier findings toward the generator verdict: {out!r}")


def test_user_tier_findings_are_surfaced_as_context_not_hidden(capsys):
    _report_revalidation([_d("user", profile="gen-references")])
    out = capsys.readouterr().out
    assert "gen-references" in out, (
        "hiding user findings trades a misleading message for a silent one")
    assert "USER tier" in out and "not attributed to the scaffold" in out


def test_findings_are_tier_labelled_like_validate(capsys):
    _report_revalidation([_d("user"), _d("workspace", severity="warning")])
    out = capsys.readouterr().out
    assert "[user]" in out and "[workspace]" in out, (
        "without the label the reader cannot tell whose fault a finding is")


def test_an_untiered_finding_is_treated_as_ours(capsys):
    """Fail closed: an unlabelled finding must not escape the verdict."""
    rc = _report_revalidation([_d(None)])
    assert rc == 1


def test_a_clean_run_is_still_clean(capsys):
    assert _report_revalidation([]) == 0
    assert "valid by construction" in capsys.readouterr().out
