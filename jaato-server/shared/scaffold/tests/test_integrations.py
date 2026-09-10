"""`jaato-scaffold integration` — provenance, and the states doctor reports.

The point of this verb is not copying files; `cp` does that. It is that every
copy carries the framework version it came from, so a stale one is DETECTABLE.
These tests are therefore mostly about `compare()` telling the four
after-install states apart — each wants a different answer from the operator,
and the historical failure was that all four looked identical on disk.
"""
import json

import pytest

from shared.scaffold import integrations as I


@pytest.fixture
def dest(tmp_path):
    return tmp_path / ".claude" / "skills" / "claude-code"


def test_the_build_ships_the_claude_code_integration():
    assert "claude-code" in I.available()


def test_install_writes_the_tree_and_a_stamp(dest):
    changed, _ = I.install("claude-code", dest)
    assert changed
    assert (dest / "SKILL.md").is_file()
    assert (dest / "references").is_dir()
    stamp = json.loads((dest / I.STAMP).read_text())
    assert stamp["integration"] == "claude-code"
    assert stamp["version"] == I.framework_version()


def test_dry_run_writes_nothing(dest):
    changed, lines = I.install("claude-code", dest, dry_run=True)
    assert not changed
    assert not dest.exists()
    assert any("SKILL.md" in l for l in lines)


def test_a_second_install_refuses_without_force(dest):
    I.install("claude-code", dest)
    changed, lines = I.install("claude-code", dest)
    assert not changed
    assert any("--force" in l for l in lines)
    changed, _ = I.install("claude-code", dest, force=True)
    assert changed


def test_unknown_integration_names_what_is_available(dest):
    changed, lines = I.install("no-such-integration", dest)
    assert not changed
    assert "claude-code" in " ".join(lines)


# --- the four states, which are the whole reason the stamp exists -----------

def test_absent(dest):
    assert I.compare("claude-code", dest)[0] == "absent"


def test_current(dest):
    I.install("claude-code", dest)
    assert I.compare("claude-code", dest)[0] == "current"


def test_stale_when_the_stamp_names_another_build(dest):
    I.install("claude-code", dest)
    f = dest / I.STAMP
    d = json.loads(f.read_text()); d["version"] = "0.0.1-old"; f.write_text(json.dumps(d))
    state, detail = I.compare("claude-code", dest)
    assert state == "stale" and "0.0.1-old" in detail


def test_edited_when_the_INSTALLED_copy_changed(dest):
    """Local edit: --force would discard it, so it must not be recommended."""
    I.install("claude-code", dest)
    (dest / "SKILL.md").write_text("edited locally\n")
    state, detail = I.compare("claude-code", dest)
    assert state == "edited"
    assert "upstream it before re-applying" in detail


def test_outdated_when_the_PAYLOAD_changed_upstream(dest, monkeypatch, tmp_path):
    """The case that produced confidently wrong advice in production.

    The payload moved at the SAME version — which happens whenever the skill is
    edited without a version bump — and the previous implementation reported it
    as a local edit, telling the operator to upstream a change they had never
    made and warning them off the `--force` that was in fact correct.
    """
    I.install("claude-code", dest)
    fake_src = tmp_path / "moved"; fake_src.mkdir()
    for f in I.payload_dir("claude-code").rglob("*"):
        if f.is_file():
            t = fake_src / f.relative_to(I.payload_dir("claude-code"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(f.read_bytes())
    (fake_src / "SKILL.md").write_text("upstream gained a paragraph\n")
    monkeypatch.setattr(I, "payload_dir", lambda n: fake_src)

    state, detail = I.compare("claude-code", dest)
    assert state == "outdated"
    assert "nothing local is lost" in detail
    # The operator must not be told to upstream a change they never made —
    # that instruction is what made the old message wrong.  ("changed upstream"
    # is fine: an adverb about the payload, not an imperative about them.)
    assert "upstream it" not in detail and "upstream them" not in detail


def test_diverged_when_both_sides_moved(dest, monkeypatch, tmp_path):
    I.install("claude-code", dest)
    (dest / "SKILL.md").write_text("edited locally\n")
    fake_src = tmp_path / "moved"; fake_src.mkdir()
    for f in I.payload_dir("claude-code").rglob("*"):
        if f.is_file():
            t = fake_src / f.relative_to(I.payload_dir("claude-code"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(f.read_bytes())
    (fake_src / "SKILL.md").write_text("upstream moved too\n")
    monkeypatch.setattr(I, "payload_dir", lambda n: fake_src)
    state, detail = I.compare("claude-code", dest)
    assert state == "diverged" and "discards the local side" in detail


def test_a_change_under_references_is_seen(dest):
    """The predecessor compared only the TOP level, so an edit confined to
    references/ — where most of the prose lives — was invisible."""
    I.install("claude-code", dest)
    ref = dest / "references" / "profiles.md"
    assert ref.is_file(), "the payload should carry references/"
    ref.write_text(ref.read_text() + "\nlocal note\n")
    assert I.compare("claude-code", dest)[0] == "edited"


def test_a_stamp_without_a_digest_does_not_guess(dest):
    """Applied before digests existed: which side moved is genuinely unknown,
    and saying so beats picking one and sounding certain."""
    import json
    I.install("claude-code", dest)
    f = dest / I.STAMP
    d = json.loads(f.read_text()); d.pop("digest"); f.write_text(json.dumps(d))
    assert I.compare("claude-code", dest)[0] == "current"      # content still matches
    (dest / "SKILL.md").write_text("something else\n")
    state, detail = I.compare("claude-code", dest)
    assert state == "diverged" and "unknown" in detail


def test_unstamped_is_reported_not_treated_as_absent(dest):
    """The historical case: hand-copied, so provenance is unknown.

    Reporting it as `absent` would invite an --force that silently discards
    whatever the copier had; reporting it as `current` would hide the drift
    this verb exists to end."""
    I.install("claude-code", dest)
    (dest / I.STAMP).unlink()
    assert I.compare("claude-code", dest)[0] == "unstamped"


def test_the_target_comes_from_the_manifest_not_from_code():
    """A Cursor integration would not write into .claude/skills — so the path
    is the integration's to declare, not this module's to assume."""
    from pathlib import Path
    m = I.manifest("claude-code")
    assert m["target"] == ".claude/skills/jaato-sdk"
    assert I.target_dir("claude-code", user=True, workspace=None) == \
        Path.home() / ".claude" / "skills" / "jaato-sdk"


def test_listing_reports_every_shipped_integration():
    data, text = I.listing()
    names = [r["name"] for r in data["integrations"]]
    assert "claude-code" in names
    assert "Claude Code" in text


# --- the CLI's promises must parse ------------------------------------------

@pytest.mark.parametrize("argv", [
    ["integration"],
    ["integration", "claude-code"],
    ["integration", "claude-code", "--user"],
    ["integration", "claude-code", "--workspace", "/tmp/x"],
    ["integration", "claude-code", "--user", "--dry-run"],
    ["integration", "claude-code", "--force", "--json"],
])
def test_every_advertised_invocation_parses(argv):
    """Help text that promises a flag the parser rejects is worse than none.

    Shipped exactly that: the listing advertised `--user` while the parser
    only understood `--workspace`, so the documented way to say "user scope"
    exited 2.  These are the forms the listing and the docstrings promise.
    """
    import argparse
    from shared.scaffold.__main__ import main
    try:
        main(argv + ["--dry-run"] if "--dry-run" not in argv else argv)
    except SystemExit as exc:            # argparse rejects with code 2
        assert exc.code != 2, f"{' '.join(argv)} was rejected by the parser"


def test_user_and_workspace_are_mutually_exclusive():
    from shared.scaffold.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main(["integration", "claude-code", "--user", "--workspace", "/tmp/x"])
    assert exc.value.code == 2
