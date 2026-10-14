"""A ``.jaato`` inside a sub-project is an ordinary workspace path (#1679).

``is_jaato_path`` flagged any path whose workspace-relative form had a
``.jaato`` component, so ``<ws>/pipeline/.jaato/profiles/x.yaml`` (a
sub-project's own config, which a Builder session writes) was treated as
the workspace's ``.jaato``.  ``check_path_with_jaato_containment`` then
required the path to lie inside ``<ws>/.jaato`` before it looked at a
``sandbox add`` grant, so ``file_edit`` refused it whatever was granted,
while ``jaato-scaffold new`` run through ``cli`` wrote there freely.

Now only the workspace's OWN ``.jaato`` counts: a path that, walked from
the workspace root, stands at or under ``<ws>/.jaato`` at any step
(``.jaato/../x``, ``sub/../.jaato/x``), or one that resolves there through a
symlink.  What stays closed is pinned beside what opens.
"""

from __future__ import annotations

import os

import pytest

from jaato_server.shared.plugins import sandbox_utils
from jaato_server.shared.plugins.file_edit.plugin import create_plugin
from jaato_server.shared.plugins.sandbox_utils import (
    check_path_with_jaato_containment,
    is_jaato_path,
)
from jaato_server.shared.tests.reversion import Reversion

_SANDBOX = "jaato-server/jaato_server/shared/plugins/sandbox_utils.py"

REVERSIONS = [
    Reversion(
        target=_SANDBOX,
        find="        if stack[:n] == jaato_parts:\n",
        replace="        if JAATO_CONFIG_DIR in stack:\n",
        because="a sub-project's .jaato would again be read as the workspace's",
        test="TestFileEdit::test_writes_and_reads_a_sub_projects_config",
    ),
    Reversion(
        target=_SANDBOX,
        find="            if stack:\n                stack.pop()\n            continue\n",
        replace="            continue\n",
        because="traversal out of the workspace's .jaato would no longer be seen",
        test="TestStillClosed::test_traversal_is_seen_lexically_without_the_dir",
    ),
    Reversion(
        target=_SANDBOX,
        find="    if os.path.lexists(jaato_dir):\n",
        replace="    if False:\n",
        because="a symlink elsewhere in the tree could reach the config dir",
        test="TestStillClosed::test_a_link_into_the_workspace_jaato",
    ),
]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with its own .jaato and a sub-project with another.

    The temp allowance is pointed at an empty directory: ``tmp_path`` is
    under ``/tmp`` on Linux, and the allowance would otherwise decide the
    allow cases for a reason unrelated to the code under test.
    """
    empty = tmp_path / "no-temp"
    empty.mkdir()
    monkeypatch.setattr(sandbox_utils, "SYSTEM_TEMP_PATHS", [str(empty)])
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("workspaceRoot", raising=False)
    root = tmp_path / "ws"
    (root / ".jaato" / "profiles").mkdir(parents=True)
    (root / ".jaato" / "profiles" / "own.yaml").write_text("name: own\n")
    (root / "pipeline" / ".jaato" / "agents").mkdir(parents=True)
    (root / "pipeline" / ".jaato" / "profiles").mkdir(parents=True)
    (root / "pipeline" / ".jaato" / "agents" / "a.md").write_text("persona\n")
    (root / "secret.txt").write_text("s\n")
    return str(root)


class TestFileEdit:
    def test_writes_and_reads_a_sub_projects_config(self, ws, tmp_path):
        plugin = create_plugin()
        plugin.initialize({"workspace_root": ws, "config_root": str(tmp_path / "cfg")})

        target = os.path.join(ws, "pipeline", ".jaato", "profiles", "trial.yaml")
        result = plugin._execute_write_new_file({"path": target, "content": "name: trial\n"})
        assert not (isinstance(result, tuple) and result[0] is False), result
        assert open(target).read() == "name: trial\n"

        read = plugin._execute_read_file(
            {"path": os.path.join(ws, "pipeline", ".jaato", "agents", "a.md")}
        )
        assert isinstance(read, str) and "persona" in read

    def test_the_workspace_jaato_is_still_refused(self, ws, tmp_path):
        plugin = create_plugin()
        plugin.initialize({"workspace_root": ws, "config_root": str(tmp_path / "cfg")})
        ok, result = plugin._execute_read_file(
            {"path": os.path.join(ws, ".jaato", "profiles", "own.yaml")}
        )
        assert ok is False
        assert "denied by sandbox" in result["error"].lower()


class TestStillClosed:
    def test_traversal_through_the_workspace_jaato(self, ws):
        for p in (".jaato/../secret.txt",
                  os.path.join(ws, ".jaato", "..", "secret.txt"),
                  "pipeline/../.jaato/profiles/own.yaml",
                  "pipeline/.jaato/../../.jaato/profiles/own.yaml"):
            assert is_jaato_path(p, ws), p
            assert check_path_with_jaato_containment(p, ws) is False, p

    def test_traversal_is_seen_lexically_without_the_dir(self, tmp_path):
        """No .jaato on disk, so only the lexical walk can answer."""
        root = tmp_path / "bare"
        root.mkdir()
        assert is_jaato_path("pipeline/../.jaato/x", str(root))
        assert not is_jaato_path("pipeline/.jaato/x", str(root))

    def test_a_link_into_the_workspace_jaato(self, ws):
        os.symlink(os.path.join(ws, ".jaato"), os.path.join(ws, "pipeline", "cfg"))
        p = os.path.join(ws, "pipeline", "cfg", "profiles", "own.yaml")
        assert is_jaato_path(p, ws)
        assert check_path_with_jaato_containment(p, ws) is False

    def test_the_workspace_path_itself_may_contain_jaato(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sandbox_utils, "SYSTEM_TEMP_PATHS", [])
        root = tmp_path / ".jaato" / "workspaces" / "s1"
        root.mkdir(parents=True)
        (root / "f.txt").write_text("x")
        assert not is_jaato_path(str(root / "f.txt"), str(root))
        assert check_path_with_jaato_containment(str(root / "f.txt"), str(root))

    def test_similar_names_are_not_the_config_dir(self, ws):
        assert not is_jaato_path(os.path.join(ws, ".jaato-backup", "x"), ws)
        assert not is_jaato_path(os.path.join(ws, "pipeline", ".jaato", "x"), ws)
