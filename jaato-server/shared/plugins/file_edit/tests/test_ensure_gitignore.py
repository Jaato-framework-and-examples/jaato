"""_ensure_gitignore must not fight a deliberate .gitignore setup, must append
the dir form, and must target the SESSION workspace (not the process cwd)."""
import pytest
from shared.plugins.file_edit.plugin import FileEditPlugin
from shared.session_context import set_workspace_root, reset_workspace_root


def test_is_jaato_rule_recognizes_all_forms():
    f = FileEditPlugin._is_jaato_rule
    for r in [".jaato", ".jaato/", ".jaato/*", ".jaato/**", "/.jaato",
              ".jaato/logs/", "!.jaato/profiles/", "  .jaato/*  "]:
        assert f(r), f"should match: {r!r}"
    for r in ["", "# .jaato", ".jaatofoo", "node_modules", "src/jaato.txt"]:
        assert not f(r), f"should NOT match: {r!r}"


def _run_ensure(tmp_path):
    token = set_workspace_root(str(tmp_path))
    try:
        FileEditPlugin()._ensure_gitignore()
    finally:
        reset_workspace_root(token)


def test_does_not_fight_deliberate_setup(tmp_path):
    gi = tmp_path / ".gitignore"
    gi.write_text(".jaato/*\n!.jaato/profiles/\n!.jaato/agents/\n")
    before = gi.read_text()
    _run_ensure(tmp_path)
    assert gi.read_text() == before        # untouched — no stray bare `.jaato`


def test_appends_dir_form_when_absent(tmp_path):
    gi = tmp_path / ".gitignore"
    gi.write_text("node_modules\n")
    _run_ensure(tmp_path)
    assert gi.read_text() == "node_modules\n.jaato/\n"   # dir form, not bare


def test_skips_when_bare_jaato_already_present(tmp_path):
    gi = tmp_path / ".gitignore"
    gi.write_text(".jaato\n")
    _run_ensure(tmp_path)
    assert gi.read_text() == ".jaato\n"    # not duplicated


def test_targets_workspace_gitignore_not_cwd(tmp_path, monkeypatch):
    # cwd is elsewhere; only the workspace .gitignore should be touched.
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
    ws = tmp_path / "workspace"; ws.mkdir()
    (ws / ".gitignore").write_text("x\n")
    cwd = tmp_path / "elsewhere"; cwd.mkdir()
    (cwd / ".gitignore").write_text("y\n")
    monkeypatch.chdir(cwd)
    _run_ensure(ws)
    assert ".jaato/" in (ws / ".gitignore").read_text()      # workspace touched
    assert (cwd / ".gitignore").read_text() == "y\n"         # cwd untouched
