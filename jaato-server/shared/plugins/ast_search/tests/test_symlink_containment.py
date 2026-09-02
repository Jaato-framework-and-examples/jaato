"""``ast_search`` is auto-approved, so its sandbox boundary has to hold.

Before jaato issue #669 this plugin performed no path validation at all: the
``path`` argument was resolved and walked as given, which made an
auto-approved tool a read of any absolute path on the host, and let ``rglob``
follow a symlinked directory out of the workspace. These tests pin both ends —
the search root and every individual result — and the special-file guard that
keeps a FIFO in the tree from blocking the worker.
"""

import os
import sys

import pytest

from shared.plugins.ast_search.plugin import ASTSearchPlugin

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX symlink semantics; Windows needs elevation to create links",
)

SECRET_SOURCE = "def exfiltrated_marker():\n    return 1\n"
INSIDE_SOURCE = "def ordinary_marker():\n    return 2\n"


@pytest.fixture
def escape_workspace(tmp_path):
    """A workspace with a symlinked directory pointing outside it.

    Layout::

        outside/secret.py           def exfiltrated_marker()
        workspace/inside.py         def ordinary_marker()
        workspace/vendor -> outside/

    Returns:
        Tuple of (workspace path, outside path).
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text(SECRET_SOURCE)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inside.py").write_text(INSIDE_SOURCE)
    (workspace / "vendor").symlink_to(outside, target_is_directory=True)

    return workspace, outside


@pytest.fixture
def plugin(escape_workspace):
    """An ``ASTSearchPlugin`` sandboxed to the escape workspace."""
    workspace, _ = escape_workspace
    p = ASTSearchPlugin()
    p.initialize({"allow_tmp": False})
    p.set_workspace_path(str(workspace))
    yield p
    p.shutdown()


def _names(result):
    """Collect the matched function names from an ast_search result."""
    return [m["text"] for m in result.get("matches", [])]


class TestSearchRootContainment:
    """An absolute path outside the workspace must be refused outright."""

    def test_absolute_path_outside_workspace_is_refused(self, plugin, escape_workspace):
        _, outside = escape_workspace
        result = plugin._execute_ast_search(
            {"pattern": "def $NAME($$$): $$$", "path": str(outside), "language": "python"}
        )
        assert "error" in result
        assert "not allowed" in result["error"]
        assert result["matches"] == []

    def test_workspace_path_still_searches(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        result = plugin._execute_ast_search(
            {"pattern": "def $NAME($$$): $$$", "path": str(workspace), "language": "python"}
        )
        assert any("ordinary_marker" in t for t in _names(result)), result

    def test_no_workspace_configured_disables_sandboxing(self, escape_workspace):
        """Unconfigured workspace keeps the previous permissive behaviour.

        Sessions that never bind a workspace (tests, embedded uses) must not
        start failing; the boundary only exists once there is one to enforce.
        """
        _, outside = escape_workspace
        p = ASTSearchPlugin()
        p.initialize({})
        try:
            result = p._execute_ast_search(
                {"pattern": "def $NAME($$$): $$$", "path": str(outside), "language": "python"}
            )
            assert any("exfiltrated_marker" in t for t in _names(result)), result
        finally:
            p.shutdown()


class TestResultContainment:
    """Results reached through a symlink out must be dropped."""

    def test_symlinked_directory_is_not_searched(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        result = plugin._execute_ast_search(
            {
                "pattern": "def $NAME($$$): $$$",
                "path": str(workspace),
                "language": "python",
            }
        )
        found = _names(result)
        assert any("ordinary_marker" in t for t in found), found
        assert not any("exfiltrated_marker" in t for t in found), found

    def test_symlinked_directory_via_explicit_file_pattern(self, plugin, escape_workspace):
        """``rglob``'s ``**`` skips links; an explicit segment does not."""
        workspace, _ = escape_workspace
        result = plugin._execute_ast_search(
            {
                "pattern": "def $NAME($$$): $$$",
                "path": str(workspace),
                "language": "python",
                "file_pattern": "vendor/*.py",
            }
        )
        assert _names(result) == [], result

    def test_symlinked_file_pointing_out_is_dropped(self, plugin, escape_workspace):
        workspace, outside = escape_workspace
        (workspace / "alias.py").symlink_to(outside / "secret.py")

        result = plugin._execute_ast_search(
            {
                "pattern": "def $NAME($$$): $$$",
                "path": str(workspace),
                "language": "python",
            }
        )
        found = _names(result)
        assert not any("exfiltrated_marker" in t for t in found), found

    def test_internal_symlink_still_searched(self, plugin, escape_workspace):
        """A symlink that stays inside the workspace remains legitimate."""
        workspace, _ = escape_workspace
        (workspace / "alias.py").symlink_to(workspace / "inside.py")

        result = plugin._execute_ast_search(
            {
                "pattern": "def $NAME($$$): $$$",
                "path": str(workspace),
                "language": "python",
            }
        )
        assert result["files_searched"] >= 2, result

    def test_fifo_in_the_tree_does_not_block(self, plugin, escape_workspace):
        """Reading a named pipe would hang the worker; it must be skipped."""
        workspace, _ = escape_workspace
        os.mkfifo(str(workspace / "pipe.py"))

        result = plugin._execute_ast_search(
            {
                "pattern": "def $NAME($$$): $$$",
                "path": str(workspace),
                "language": "python",
            }
        )
        assert any("ordinary_marker" in t for t in _names(result)), result


class TestStreamingContainment:
    """The streaming variant enforces the same boundary."""

    @pytest.mark.asyncio
    async def test_streaming_refuses_outside_root(self, plugin, escape_workspace):
        _, outside = escape_workspace
        chunks = [
            chunk
            async for chunk in plugin.execute_streaming(
                "ast_search",
                {
                    "pattern": "def $NAME($$$): $$$",
                    "path": str(outside),
                    "language": "python",
                },
            )
        ]
        assert any(c.chunk_type == "error" and "not allowed" in c.content for c in chunks)
        assert not any("exfiltrated_marker" in c.content for c in chunks)

    @pytest.mark.asyncio
    async def test_streaming_drops_symlinked_results(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        chunks = [
            chunk
            async for chunk in plugin.execute_streaming(
                "ast_search",
                {
                    "pattern": "def $NAME($$$): $$$",
                    "path": str(workspace),
                    "language": "python",
                    "file_pattern": "vendor/*.py",
                },
            )
        ]
        assert not any("exfiltrated_marker" in c.content for c in chunks)
