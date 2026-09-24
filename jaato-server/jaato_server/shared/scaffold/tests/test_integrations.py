"""`jaato-scaffold integration` — provenance, and the states doctor reports.

The point of this verb is not copying files; `cp` does that. It is that every
copy carries the framework version it came from, so a stale one is DETECTABLE.
These tests are therefore mostly about `compare()` telling the four
after-install states apart — each wants a different answer from the operator,
and the historical failure was that all four looked identical on disk.
"""
import json

import pytest

from jaato_server.shared.scaffold import integrations as I


@pytest.fixture
def dest(tmp_path):
    return tmp_path / ".claude" / "skills" / "claude-code"


def test_the_build_ships_the_agent_harness_integrations():
    assert {"claude-code", "pi"} <= set(I.available())


def test_pi_reuses_the_agent_skill_payload():
    assert I.payload_dir("pi") == I.payload_dir("claude-code")


def test_missing_shared_payload_is_not_reported_as_unknown(dest, monkeypatch, tmp_path):
    monkeypatch.setattr(I, "payload_dir", lambda _name: tmp_path / "missing")

    changed, lines = I.install("pi", dest)

    assert not changed
    assert "payload 'claude-code' is missing" in " ".join(lines)
    assert "unknown integration" not in " ".join(lines)


def test_a_missing_shared_payload_is_found_through_payload_from(tmp_path, monkeypatch):
    """The same case as above, without stubbing `payload_dir`.

    The test above replaces `payload_dir` outright, so it pins the MESSAGE
    and would pass whether or not `payload_dir` honours `payload_from`.  Here
    the real Pi manifest ships alone — no `claude-code/` beside it — so the
    only route to the error is `payload_dir` resolving `payload_from` to a
    directory this build does not have.
    """
    import shutil
    root = tmp_path / "integrations"
    shutil.copytree(I._source_root() / "pi", root / "pi")
    monkeypatch.setattr(I, "_source_root", lambda: root)
    assert I.available() == ["pi"]
    assert I.payload_dir("pi") == root / "claude-code" / "payload"

    dest = tmp_path / "dest"
    changed, lines = I.install("pi", dest)
    assert not changed
    assert lines == ["integration 'pi' payload 'claude-code' is missing from this build"]
    assert not dest.exists()


def _pi_rows(checks):
    return [c for c in checks if c.name == "integration (pi)"]


def test_doctor_is_silent_about_pi_where_pi_is_not_installed(tmp_path, monkeypatch):
    """The real Pi manifest's `detect` is what keeps a Claude Code user from a
    permanent 'install the Pi skill' warning they cannot clear."""
    from jaato_sdk.doctor import check_integrations
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    empty = tmp_path / "bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert I.harness_present("pi") is False
    assert _pi_rows(check_integrations()) == []


def test_doctor_nudges_where_pi_is_installed_and_the_skill_is_not(tmp_path, monkeypatch):
    """The complement: with a `pi` executable on PATH the suggestion returns."""
    from jaato_sdk.doctor import WARN, check_integrations
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    pi = bindir / "pi"
    pi.write_text("#!/bin/sh\n")
    pi.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    assert I.harness_present("pi") is True
    rows = _pi_rows(check_integrations())
    assert [c.status for c in rows] == [WARN]
    assert "not applied" in rows[0].detail


def test_pi_install_stamps_pi(tmp_path):
    dest = tmp_path / ".pi" / "skills" / "jaato-sdk"
    changed, _ = I.install("pi", dest)
    assert changed
    assert (dest / "SKILL.md").is_file()
    stamp = json.loads((dest / I.STAMP).read_text())
    assert stamp["integration"] == "pi"


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
    assert any("SKILL.md" in line for line in lines)


def test_a_second_install_refuses_without_force(dest):
    I.install("claude-code", dest)
    changed, lines = I.install("claude-code", dest)
    assert not changed
    assert any("--force" in line for line in lines)
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
    d = json.loads(f.read_text())
    d["version"] = "0.0.1-old"
    f.write_text(json.dumps(d))
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
    fake_src = tmp_path / "moved"
    fake_src.mkdir()
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
    fake_src = tmp_path / "moved"
    fake_src.mkdir()
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
    d = json.loads(f.read_text())
    d.pop("digest")
    f.write_text(json.dumps(d))
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


def test_the_target_comes_from_the_manifest_not_from_code(tmp_path):
    """Each harness owns its user and workspace skill locations."""
    from pathlib import Path
    m = I.manifest("claude-code")
    assert m["target"] == ".claude/skills/jaato-sdk"
    assert I.target_dir("claude-code", user=True, workspace=None) == \
        Path.home() / ".claude" / "skills" / "jaato-sdk"
    assert I.target_dir("pi", user=True, workspace=None) == \
        Path.home() / ".pi" / "agent" / "skills" / "jaato-sdk"
    assert I.target_dir("pi", user=False, workspace=str(tmp_path)) == \
        tmp_path / ".pi" / "skills" / "jaato-sdk"


def test_listing_reports_every_shipped_integration():
    data, text = I.listing()
    names = [r["name"] for r in data["integrations"]]
    assert {"claude-code", "pi"} <= set(names)
    assert "Claude Code" in text
    assert "Pi" in text


# --- the CLI's promises must parse ------------------------------------------

@pytest.mark.parametrize("argv", [
    ["integration"],
    ["integration", "claude-code"],
    ["integration", "claude-code", "--user"],
    ["integration", "claude-code", "--workspace", "workspace"],
    ["integration", "claude-code", "--user", "--dry-run"],
    ["integration", "claude-code", "--force", "--json"],
    ["integration", "claude-code", "--refresh"],
    ["integration", "claude-code", "--refresh", "--json"],
    ["integration", "pi"],
    ["integration", "pi", "--workspace", "workspace"],
    ["integration", "pi", "--refresh"],
])
def test_every_advertised_invocation_parses(argv):
    """Help text that promises a flag the parser rejects is worse than none.

    Shipped exactly that: the listing advertised `--user` while the parser
    only understood `--workspace`, so the documented way to say "user scope"
    exited 2.  These are the forms the listing and the docstrings promise.
    """
    from jaato_server.shared.scaffold.__main__ import main
    try:
        main(argv + ["--dry-run"] if "--dry-run" not in argv else argv)
    except SystemExit as exc:            # argparse rejects with code 2
        assert exc.code != 2, f"{' '.join(argv)} was rejected by the parser"


def test_user_and_workspace_are_mutually_exclusive():
    from jaato_server.shared.scaffold.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main(["integration", "claude-code", "--user", "--workspace", "workspace"])
    assert exc.value.code == 2


# --- --refresh: overwrite only when nothing local is lost (#1261) -----------

def _copy_payload(tmp_path):
    """A writable copy of the shipped payload, for a monkeypatched src.

    The `outdated` and `diverged` states need the UPSTREAM payload to have
    moved, which the tests below simulate by pointing `payload_dir` at an
    edited copy — the same device `test_outdated_*` and `test_diverged_*` use.
    """
    fake_src = tmp_path / "moved"
    fake_src.mkdir()
    real = I.payload_dir("claude-code")
    for f in real.rglob("*"):
        if f.is_file():
            t = fake_src / f.relative_to(real)
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(f.read_bytes())
    return fake_src


def test_refresh_writes_absent_stale_outdated(dest, monkeypatch, tmp_path):
    """The write half of the flag's table: the three states that lose nothing
    local by being re-applied.  Each is built in isolation so no state leaks
    into the next, then refreshed, and the result must be `current`.
    """
    # absent: nothing installed yet.
    changed, _ = I.install("claude-code", dest, refresh=True)
    assert changed and I.compare("claude-code", dest)[0] == "current"

    # stale: a pristine copy from another build.
    f = dest / I.STAMP
    d = json.loads(f.read_text()); d["version"] = "0.0.1-old"; f.write_text(json.dumps(d))
    assert I.compare("claude-code", dest)[0] == "stale"
    changed, _ = I.install("claude-code", dest, refresh=True)
    assert changed and I.compare("claude-code", dest)[0] == "current"

    # outdated: the payload moved upstream at the same version.  The moved
    # payload stays in force through the refresh AND the follow-up compare, so
    # writing from it lands the copy at `current`.
    fake_src = _copy_payload(tmp_path)
    (fake_src / "SKILL.md").write_text("upstream gained a paragraph\n")
    monkeypatch.setattr(I, "payload_dir", lambda n: fake_src)
    assert I.compare("claude-code", dest)[0] == "outdated"
    changed, _ = I.install("claude-code", dest, refresh=True)
    assert changed and I.compare("claude-code", dest)[0] == "current"


def test_refresh_leaves_edited_diverged_unstamped(dest, monkeypatch, tmp_path):
    """The leave-alone half: the states that carry local content a rewrite
    would discard.  Each is built on its own dest, and the local content must
    survive the refresh untouched with changed=False."""
    # edited: a local change, payload unmoved.
    e = dest.parent / "edited"
    I.install("claude-code", e)
    (e / "SKILL.md").write_text("edited locally\n")
    assert I.compare("claude-code", e)[0] == "edited"
    changed, lines = I.install("claude-code", e, refresh=True)
    assert not changed
    assert (e / "SKILL.md").read_text() == "edited locally\n"
    assert any("left unchanged" in l and "edited" in l for l in lines)

    # diverged: both sides moved.
    v = dest.parent / "diverged"
    I.install("claude-code", v)
    (v / "SKILL.md").write_text("edited locally\n")
    fake_src = _copy_payload(tmp_path)
    (fake_src / "SKILL.md").write_text("upstream moved too\n")
    monkeypatch.setattr(I, "payload_dir", lambda n: fake_src)
    assert I.compare("claude-code", v)[0] == "diverged"
    changed, _ = I.install("claude-code", v, refresh=True)
    assert not changed
    assert (v / "SKILL.md").read_text() == "edited locally\n"
    monkeypatch.undo()

    # unstamped: hand-copied, provenance unknown.
    u = dest.parent / "unstamped"
    I.install("claude-code", u)
    (u / I.STAMP).unlink()
    assert I.compare("claude-code", u)[0] == "unstamped"
    changed, _ = I.install("claude-code", u, refresh=True)
    assert not changed
    assert not (u / I.STAMP).is_file()


def test_refresh_leaves_a_current_copy_untouched(dest):
    """A current copy is already up to date, so refresh writes nothing and
    reports success — the same exit-0 skip an edited copy gets."""
    I.install("claude-code", dest)
    assert I.compare("claude-code", dest)[0] == "current"
    changed, _ = I.install("claude-code", dest, refresh=True)
    assert not changed


def test_refresh_does_not_overwrite_edits_made_under_an_older_version(dest):
    """The digest-first check (#1261).

    `compare` returns `stale` the instant the version differs — so a copy
    edited under an OLDER framework version would read `stale`, a state
    refresh is entitled to overwrite, and the edit would be silently
    discarded on upgrade.  The digest recorded at apply time is compared
    FIRST, so a changed installed tree reads `edited` whatever the version,
    and refresh leaves it alone.
    """
    I.install("claude-code", dest)
    (dest / "SKILL.md").write_text("edited under the old version\n")
    f = dest / I.STAMP
    d = json.loads(f.read_text()); d["version"] = "0.0.1-old"; f.write_text(json.dumps(d))

    # Version differs AND the tree was edited: this is `edited`, not `stale`.
    assert I.compare("claude-code", dest)[0] == "edited"
    changed, lines = I.install("claude-code", dest, refresh=True)
    assert not changed
    assert (dest / "SKILL.md").read_text() == "edited under the old version\n"


def test_a_pristine_copy_at_another_version_is_still_stale(dest):
    """The complement of the digest-first check: an UNEDITED copy at a
    different version is `stale` (safe to overwrite), so refresh writes it."""
    I.install("claude-code", dest)
    f = dest / I.STAMP
    d = json.loads(f.read_text()); d["version"] = "0.0.1-old"; f.write_text(json.dumps(d))
    assert I.compare("claude-code", dest)[0] == "stale"
    changed, _ = I.install("claude-code", dest, refresh=True)
    assert changed
    assert I.compare("claude-code", dest)[0] == "current"


def test_refresh_and_force_are_mutually_exclusive():
    """They mean opposite things about local edits, so asking for both is a
    contradiction argparse refuses (exit 2)."""
    from jaato_server.shared.scaffold.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main(["integration", "claude-code", "--refresh", "--force", "--dry-run"])
    assert exc.value.code == 2


def test_refresh_json_reports_the_transition_and_exits_zero(tmp_path, capsys):
    """--json --refresh reports state_before / state_after / changed /
    skipped_reason, and exit code is 0 whether it wrote or skipped."""
    from jaato_server.shared.scaffold.__main__ import main

    ws = str(tmp_path)
    # The CLI resolves the install path from the manifest, not from us.
    target = I.target_dir("claude-code", user=False, workspace=ws)

    # absent -> writes: state_before absent, state_after current, changed, no skip.
    code = main(["integration", "claude-code", "--workspace", ws, "--refresh", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["state_before"] == "absent"
    assert out["state_after"] == "current"
    assert out["changed"] is True
    assert out["skipped_reason"] is None

    # edited -> skips: changed False, skipped_reason names the state, still exit 0.
    (target / "SKILL.md").write_text("edited locally\n")
    code = main(["integration", "claude-code", "--workspace", ws, "--refresh", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["state_before"] == "edited"
    assert out["state_after"] == "edited"
    assert out["changed"] is False
    assert out["skipped_reason"] and out["skipped_reason"].startswith("edited")
    assert (target / "SKILL.md").read_text() == "edited locally\n"
