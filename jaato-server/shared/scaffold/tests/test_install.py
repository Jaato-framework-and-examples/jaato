"""`jaato-scaffold install` — provenance, and the states doctor reports.

The point of this verb is not copying files; `cp` does that. It is that every
copy carries the framework version it came from, so a stale one is DETECTABLE.
These tests are therefore mostly about `compare()` telling the four
after-install states apart — each wants a different answer from the operator,
and the historical failure was that all four looked identical on disk.
"""
import json

import pytest

from shared.scaffold import install as I


@pytest.fixture
def dest(tmp_path):
    return tmp_path / ".claude" / "skills" / "jaato-sdk"


def test_the_build_ships_the_skill():
    assert "jaato-sdk" in I.available()


def test_install_writes_the_tree_and_a_stamp(dest):
    changed, _ = I.install("jaato-sdk", dest)
    assert changed
    assert (dest / "SKILL.md").is_file()
    assert (dest / "references").is_dir()
    stamp = json.loads((dest / I.STAMP).read_text())
    assert stamp["asset"] == "jaato-sdk"
    assert stamp["version"] == I.framework_version()


def test_dry_run_writes_nothing(dest):
    changed, lines = I.install("jaato-sdk", dest, dry_run=True)
    assert not changed
    assert not dest.exists()
    assert any("SKILL.md" in l for l in lines)


def test_a_second_install_refuses_without_force(dest):
    I.install("jaato-sdk", dest)
    changed, lines = I.install("jaato-sdk", dest)
    assert not changed
    assert any("--force" in l for l in lines)
    changed, _ = I.install("jaato-sdk", dest, force=True)
    assert changed


def test_unknown_asset_names_what_is_available(dest):
    changed, lines = I.install("no-such-skill", dest)
    assert not changed
    assert "jaato-sdk" in " ".join(lines)


# --- the four states, which are the whole reason the stamp exists -----------

def test_absent(dest):
    assert I.compare("jaato-sdk", dest)[0] == "absent"


def test_current(dest):
    I.install("jaato-sdk", dest)
    assert I.compare("jaato-sdk", dest)[0] == "current"


def test_stale_when_the_stamp_names_another_build(dest):
    I.install("jaato-sdk", dest)
    f = dest / I.STAMP
    d = json.loads(f.read_text()); d["version"] = "0.0.1-old"; f.write_text(json.dumps(d))
    state, detail = I.compare("jaato-sdk", dest)
    assert state == "stale" and "0.0.1-old" in detail


def test_modified_when_edited_at_the_same_version(dest):
    I.install("jaato-sdk", dest)
    (dest / "SKILL.md").write_text("edited locally\n")
    state, detail = I.compare("jaato-sdk", dest)
    assert state == "modified" and "differ" in detail


def test_unstamped_is_reported_not_treated_as_absent(dest):
    """The historical case: hand-copied, so provenance is unknown.

    Reporting it as `absent` would invite an --force that silently discards
    whatever the copier had; reporting it as `current` would hide the drift
    this verb exists to end."""
    I.install("jaato-sdk", dest)
    (dest / I.STAMP).unlink()
    assert I.compare("jaato-sdk", dest)[0] == "unstamped"


def test_user_scope_is_the_default_target():
    from pathlib import Path
    assert I.target_dir("jaato-sdk", user=True, workspace=None) == \
        Path.home() / ".claude" / "skills" / "jaato-sdk"
