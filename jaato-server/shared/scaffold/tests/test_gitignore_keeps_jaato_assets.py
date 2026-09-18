"""``jaato-scaffold new`` / ``validate`` and the workspace ``.gitignore``.

``<workspace>/.jaato/`` mixes authored assets (profiles, agents, schemas,
processors) with runtime state (sessions, logs, memories, stored
credentials), and nothing the scaffold wrote said which half is
committable: ``new profile-set`` ignored ``.env`` and nothing else, so a
workspace either committed its session records and ``<provider>_auth.json``
or ignored ``.jaato/`` wholesale and lost the profiles its sessions ran
under.  ``shared/scaffold/gitignore.py`` declares the split once; ``new``
merges the block wherever it writes under ``.jaato/`` (and on its own as
``new gitignore``), and ``validate`` judges an existing file by EFFECT.

The git-backed cases here are the evidence: the block's shape — ``!.jaato/``
before ``.jaato/*`` before the re-includes — is what makes it work on top
of an author's wholesale rule, and only ``git check-ignore`` can say so.
They are skipped where git is absent; the parser-backed assertions run
everywhere.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import subprocess
from pathlib import Path

import pytest

from shared.scaffold import archetypes as A
from shared.scaffold import build
from shared.scaffold import gitignore as G
from shared.scaffold import validate as V

PROVIDER = "nebius"

#: Files that must be COMMITTABLE once the block is in place — one under
#: each kind of authored entry, plus a nested set profile.
AUTHORED_FILES = (
    ".jaato/profiles/_base_a.yaml",
    ".jaato/profiles/s1/a.yaml",
    ".jaato/agents/a.md",
    ".jaato/instructions/00-base.md",
    ".jaato/completion_schemas/gate.json",
    ".jaato/scripts/processors/gate.py",
    ".jaato/services/gitlab/_service.yaml",
    ".jaato/gc.json",
    ".jaato/prompts/review.md",
)

#: Files that must stay IGNORED — including a directory the block does not
#: name, which is the "ignored unless named" direction the block commits to.
STATE_FILES = (
    ".jaato/sessions/20260101_000000/session.json",
    ".jaato/logs/session.log",
    ".jaato/memories/notes.md",
    ".jaato/openrouter_auth.json",
    ".jaato/scripts/__pycache__/gate.cpython-311.pyc",
    ".jaato/some_state_a_later_release_adds/x",
)


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    defaults = dict(archetype=None, workspace=None, provider=PROVIDER,
                    model="m", set=None, agents=None, force=False, json=False,
                    recoverable=False, dry_run=False, secrets=None,
                    secret_path=None, transport="ipc", url=None, token=None,
                    ca=None, name=None, no_gate=False, gate_name=None,
                    profile=None)
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def _run(args) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return build.run(args)


def _touch(ws: Path, rel: str) -> None:
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")


def _gitignore_codes(ws: Path) -> set:
    return {d.code for d in V.validate_workspace(str(ws))
            if d.code.startswith("gitignore_")}


_git = shutil.which("git")
needs_git = pytest.mark.skipif(_git is None, reason="git is not installed")


def _git_ignored(ws: Path, rel: str) -> bool:
    return subprocess.run([_git, "-C", str(ws), "check-ignore", "-q", rel]
                          ).returncode == 0


def _git_workspace(root: Path, gitignore_text: str) -> Path:
    ws = root / "ws"
    ws.mkdir()
    subprocess.run([_git, "init", "-q", str(ws)], check=True)
    (ws / ".gitignore").write_text(gitignore_text, encoding="utf-8")
    for rel in AUTHORED_FILES + STATE_FILES:
        _touch(ws, rel)
    return ws


# ------------------------------------------------------- the block, under git

@needs_git
def test_fresh_block_keeps_assets_and_ignores_state_under_git(tmp_path):
    ws = _git_workspace(tmp_path, G.render_block())
    for rel in AUTHORED_FILES:
        assert not _git_ignored(ws, rel), f"{rel} must be committable"
    for rel in STATE_FILES:
        assert _git_ignored(ws, rel), f"{rel} must stay ignored"
    # and the daemon's parser — what `validate` reads — agrees with git
    assert G.assess(ws).clean


@needs_git
@pytest.mark.parametrize("prior", [
    pytest.param(".jaato/\n", id="wholesale-dir"),
    pytest.param(".*\n!.gitignore\n", id="dotfiles"),
    pytest.param("/.jaato\n", id="anchored-dir"),
])
def test_block_repairs_a_wholesale_rule_without_editing_it(tmp_path, prior):
    """Git cannot re-include beneath an excluded directory, so on top of a
    wholesale rule a plain ``.jaato/*`` + re-includes would be inert.  The
    block's leading ``!.jaato/`` un-excludes the directory first — and the
    author's line is left exactly where it was."""
    merged = G.merge(prior)
    assert merged.startswith(prior), "the author's rule was rewritten"
    ws = _git_workspace(tmp_path, merged)
    for rel in AUTHORED_FILES:
        assert not _git_ignored(ws, rel), f"{rel} still hidden after {prior!r}"
    for rel in STATE_FILES:
        assert _git_ignored(ws, rel)
    assert G.assess(ws).clean


# ---------------------------------------------------------------- merge

def test_merge_is_idempotent():
    once = G.merge(None)
    assert G.merge(once) is None
    # whitespace and comments around the lines do not defeat the check
    padded = "\n".join("  " + ln + "  " for ln in once.splitlines()) + "\n"
    assert G.merge(padded) is None


def test_merge_adds_only_the_missing_lines_to_a_current_file():
    """A file written under an older, shorter list gains the new entries and
    nothing else — the anchors are not repeated."""
    older = "!.jaato/\n.jaato/*\n!.jaato/profiles/\n"
    merged = G.merge(older)
    assert merged.count(".jaato/*\n") == 1
    assert merged.count("!.jaato/profiles/\n") == 1
    assert "!.jaato/agents/\n" in merged
    assert G.merge(merged) is None


def test_merge_appends_the_whole_block_when_an_anchor_is_missing():
    """``!.jaato/profiles/`` with no ``.jaato/*`` before it: the re-include
    already present is appended AGAIN, because it has to come after the
    ``.jaato/*`` being added or that new rule would silence it."""
    stray = "!.jaato/profiles/\n"
    merged = G.merge(stray)
    lines = merged.splitlines()
    assert lines.index(".jaato/*") > 0
    assert lines.index(".jaato/*") < len(lines) - 1 - lines[::-1].index("!.jaato/profiles/")


def test_merge_preserves_a_file_that_does_not_end_in_a_newline():
    merged = G.merge("node_modules/")
    assert merged.startswith("node_modules/\n\n")


# ------------------------------------------------------ every `new` writes it

def _block_present(ws: Path) -> bool:
    lines = (ws / ".gitignore").read_text(encoding="utf-8").split()
    return ".jaato/*" in lines and "!.jaato/profiles/" in lines


def test_profile_set_writes_the_block_in_every_secrets_mode(tmp_path):
    for mode in ("env", "none", "pass"):
        ws = tmp_path / mode
        assert _run(_args(archetype=A.PROFILE_SET, workspace=str(ws),
                          set="s1", agents="a", secrets=mode)) == 0
        assert _block_present(ws), mode
        has_env_rule = ".env" in (ws / ".gitignore").read_text().split()
        assert has_env_rule == (mode != "pass"), mode


def test_processor_and_sweep_write_the_block(tmp_path):
    ws = tmp_path / "proc"
    assert _run(_args(archetype=A.PROCESSOR, workspace=str(ws), name="gate")) == 0
    assert _block_present(ws)
    ws = tmp_path / "sweep"
    assert _run(_args(archetype="sweep", workspace=str(ws))) == 0
    assert _block_present(ws)


def test_sweep_without_a_gate_writes_nothing_under_jaato_and_no_gitignore(tmp_path):
    ws = tmp_path / "sweep"
    assert _run(_args(archetype="sweep", workspace=str(ws), no_gate=True)) == 0
    assert not (ws / ".jaato").exists()
    assert not (ws / ".gitignore").exists()


def test_a_second_run_writes_the_gitignore_once(tmp_path):
    """The env rule and the .jaato block share one file and one write, so a
    re-run reports the file as untouched rather than appending a copy."""
    ws = tmp_path / "ws"
    _run(_args(archetype=A.PROFILE_SET, workspace=str(ws), set="s1", agents="a"))
    before = (ws / ".gitignore").read_text()
    _run(_args(archetype=A.PROFILE_SET, workspace=str(ws), set="s2", agents="b"))
    assert (ws / ".gitignore").read_text() == before


# ------------------------------------------------------------ new gitignore

def test_new_gitignore_alone(tmp_path, capsys):
    ws = tmp_path / "ws"
    assert build.run(_args(archetype=A.GITIGNORE, workspace=str(ws))) == 0
    assert _block_present(ws)
    assert G.assess(ws).clean
    capsys.readouterr()
    # idempotent, and says so
    assert build.run(_args(archetype=A.GITIGNORE, workspace=str(ws))) == 0
    assert "nothing written" in capsys.readouterr().out
    assert G.merge((ws / ".gitignore").read_text()) is None


def test_new_gitignore_dry_run_writes_nothing(tmp_path):
    ws = tmp_path / "never"
    assert _run(_args(archetype=A.GITIGNORE, workspace=str(ws), dry_run=True)) == 0
    assert not ws.exists()


def test_new_gitignore_is_an_accepted_documented_archetype():
    assert A.GITIGNORE in A.accepted()
    doc = A.resolve(A.GITIGNORE)
    assert doc is not None and doc.requires == ("--workspace",)
    assert [e.path for e in doc.writes] == [".gitignore"]


# ---------------------------------------------------------------- validate

def test_validate_is_silent_without_a_jaato_dir(tmp_path):
    (tmp_path / ".git").mkdir()
    assert not _gitignore_codes(tmp_path)


def test_validate_reports_a_missing_gitignore_only_at_a_repository_root(tmp_path):
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    # nested workspace: the parent's rules are unknown, so nothing is said
    assert not _gitignore_codes(tmp_path)
    (tmp_path / ".git").mkdir()
    assert _gitignore_codes(tmp_path) == {"gitignore_missing"}


def test_validate_reports_a_wholesale_rule_as_hiding_the_assets(tmp_path):
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    (tmp_path / ".gitignore").write_text(".jaato/\n")
    diags = [d for d in V.validate_workspace(str(tmp_path))
             if d.code.startswith("gitignore_")]
    assert {d.code for d in diags} == {"gitignore_hides_jaato_assets"}
    (d,) = diags
    assert d.severity == "warn" and d.tier == "workspace"
    assert "excludes .jaato/ itself" in d.message
    assert "jaato-scaffold new gitignore" in d.message


def test_validate_reports_state_left_committable(tmp_path):
    """The file `new profile-set` used to write — the .env rule alone — is
    exactly this shape: nothing under .jaato/ is ignored, stored credential
    included."""
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    (tmp_path / ".gitignore").write_text(build._ENV_RULE_BLOCK)
    diags = [d for d in V.validate_workspace(str(tmp_path))
             if d.code.startswith("gitignore_")]
    assert {d.code for d in diags} == {"gitignore_leaks_jaato_state"}
    assert "stored provider credential" in diags[0].message


def test_validate_judges_by_effect_not_by_spelling(tmp_path):
    """A hand-written file that reaches the same result — different order,
    its own comments, extra rules — is not reported."""
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    hand = ["# mine", "node_modules/", ".jaato/*", "!.jaato/"]
    hand += [f"!.jaato/{e.path}" for e in reversed(G.AUTHORED)]
    hand += ["", "# compiled", "__pycache__/"]
    (tmp_path / ".gitignore").write_text("\n".join(hand) + "\n")
    assert not _gitignore_codes(tmp_path)


def test_validate_is_clean_after_new_gitignore(tmp_path):
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(".jaato/\n")
    assert _gitignore_codes(tmp_path) == {"gitignore_hides_jaato_assets"}
    assert _run(_args(archetype=A.GITIGNORE, workspace=str(tmp_path))) == 0
    assert not _gitignore_codes(tmp_path)


def test_scaffolded_workspaces_validate_clean(tmp_path):
    """`new` re-validates its own output; the block it writes must not be
    what the validator then complains about."""
    ws = tmp_path / "ws"
    assert _run(_args(archetype=A.PROFILE_SET, workspace=str(ws), set="s1",
                      agents="a")) == 0
    (ws / ".git").mkdir()
    assert not _gitignore_codes(ws)
