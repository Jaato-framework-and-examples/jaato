"""Containment must hold per search *result*, not only for the search root.

``Path.glob`` follows symlinked directories.  A link committed into a
repository (``data/logs -> /etc``) therefore turns a workspace-scoped search
into a read outside the workspace unless every hit is re-checked: the root
passes, and the walk quietly reaches through the link.  The same hole exposes
``.jaato``, which is denied by default but lives inside the workspace and so is
reachable from an allowed root.

These tests plant exactly that symlink and assert the escape is closed for
``glob_files`` and ``grep_content``, in both their blocking and streaming
forms.  See jaato issue #669 (shape B).
"""

import os
import sys

import pytest

from shared.plugins.filesystem_query.plugin import FilesystemQueryPlugin

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX symlink semantics; Windows needs elevation to create links",
)


@pytest.fixture
def escape_workspace(tmp_path):
    """A workspace holding a symlinked directory that points outside it.

    Layout::

        outside/secret.txt          "TOPSECRET sentinel"
        workspace/inside.txt        "ordinary sentinel"
        workspace/data/logs -> outside/
        workspace/link.txt -> outside/secret.txt

    Returns:
        Tuple of (workspace path, outside path).
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET sentinel\n")

    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "inside.txt").write_text("ordinary sentinel\n")
    (workspace / "data" / "logs").symlink_to(outside, target_is_directory=True)
    (workspace / "link.txt").symlink_to(outside / "secret.txt")

    return workspace, outside


@pytest.fixture
def plugin(escape_workspace):
    """A ``FilesystemQueryPlugin`` sandboxed to the escape workspace."""
    workspace, _ = escape_workspace
    p = FilesystemQueryPlugin()
    p.initialize({"workspace_root": str(workspace), "allow_tmp": False})
    yield p
    p.shutdown()


def _paths(result):
    """Collect the absolute paths from a glob_files result."""
    return [f["absolute_path"] for f in result["files"]]


class TestGlobFilesResultContainment:
    """glob_files must not report files reached through a symlink out."""

    def test_symlinked_directory_is_not_traversed(self, plugin, escape_workspace):
        """An explicit segment naming the link is the traversal that bites.

        ``pathlib``'s ``**`` declines to recurse into symlinked directories on
        its own, so the escape needs a pattern that names the link outright —
        which a model can write as easily as any other.
        """
        workspace, _ = escape_workspace
        result = plugin._execute_glob_files(
            {"pattern": "data/logs/*.txt", "root": str(workspace)}
        )
        assert _paths(result) == []
        assert result["total"] == 0

    def test_symlinked_file_pointing_out_is_not_reported(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        result = plugin._execute_glob_files(
            {"pattern": "*.txt", "root": str(workspace)}
        )
        found = _paths(result)
        assert any(p.endswith("inside.txt") for p in found), found
        assert not any(p.endswith("link.txt") for p in found), found

    def test_pointing_root_at_the_symlink_is_refused(self, plugin, escape_workspace):
        """The pre-existing root check still rejects the direct approach."""
        workspace, _ = escape_workspace
        result = plugin._execute_glob_files(
            {"pattern": "*", "root": str(workspace / "data" / "logs")}
        )
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_jaato_is_not_reachable_through_a_hidden_glob(self, plugin, escape_workspace):
        """`.jaato` is denied by default and must not leak via include_hidden."""
        workspace, _ = escape_workspace
        (workspace / ".jaato").mkdir()
        (workspace / ".jaato" / "creds.json").write_text('{"token": "hunter2"}')

        result = plugin._execute_glob_files(
            {"pattern": "**/*", "root": str(workspace), "include_hidden": True}
        )
        found = _paths(result)
        assert not any("creds.json" in p for p in found), found

    def test_ordinary_internal_symlink_still_works(self, plugin, escape_workspace):
        """A symlink that stays inside the workspace is legitimate."""
        workspace, _ = escape_workspace
        (workspace / "alias.txt").symlink_to(workspace / "inside.txt")

        result = plugin._execute_glob_files(
            {"pattern": "*.txt", "root": str(workspace)}
        )
        found = _paths(result)
        assert any(p.endswith("alias.txt") for p in found), found


class TestGrepContentResultContainment:
    """grep_content must not read files reached through a symlink out."""

    def test_symlinked_directory_contents_are_not_searched(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        result = plugin._execute_grep_content(
            {
                "pattern": "sentinel",
                "path": str(workspace),
                "file_glob": ["data/logs/*.txt", "*.txt"],
            }
        )
        texts = [m["text"] for m in result["matches"]]
        assert any("ordinary sentinel" in t for t in texts), texts
        assert not any("TOPSECRET" in t for t in texts), texts

    def test_symlinked_file_pointing_out_is_not_searched(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        result = plugin._execute_grep_content(
            {"pattern": "TOPSECRET", "path": str(workspace), "file_glob": ["*.txt"]}
        )
        assert result["matches"] == []
        assert result["total_matches"] == 0

    def test_fifo_does_not_block_the_search(self, plugin, escape_workspace):
        """A FIFO in the tree must be skipped, not opened and waited on.

        Without the special-file guard the binary sniff opens the pipe and
        blocks until a writer appears — i.e. forever.  The timeout here is
        the assertion: the call has to return.
        """
        workspace, _ = escape_workspace
        os.mkfifo(str(workspace / "pipe.log"))

        result = plugin._execute_grep_content(
            {"pattern": "sentinel", "path": str(workspace)}
        )
        assert any("ordinary sentinel" in m["text"] for m in result["matches"])

    def test_naming_the_fifo_directly_is_refused(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        fifo = workspace / "direct.pipe"
        os.mkfifo(str(fifo))

        result = plugin._execute_grep_content(
            {"pattern": "anything", "path": str(fifo)}
        )
        assert result["matches"] == []
        assert result["files_searched"] == 0


class TestStreamingContainment:
    """The streaming variants enforce the same boundary as the blocking ones."""

    async def _collect(self, plugin, tool, args):
        return [chunk async for chunk in plugin.execute_streaming(tool, args)]

    @pytest.mark.asyncio
    async def test_streaming_grep_checks_the_root(self, plugin, escape_workspace):
        """`grep_content:stream` used to skip the root check entirely."""
        _, outside = escape_workspace
        chunks = await self._collect(
            plugin, "grep_content", {"pattern": "TOPSECRET", "path": str(outside)}
        )
        assert any(c.chunk_type == "error" for c in chunks)
        assert not any("TOPSECRET" in c.content for c in chunks if c.chunk_type == "match")

    @pytest.mark.asyncio
    async def test_streaming_grep_checks_each_result(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        chunks = await self._collect(
            plugin, "grep_content", {"pattern": "sentinel", "path": str(workspace)}
        )
        matches = [c for c in chunks if c.chunk_type == "match"]
        assert any("inside.txt" in c.content for c in matches), matches
        assert not any("secret" in c.content for c in matches), matches

    @pytest.mark.asyncio
    async def test_streaming_glob_checks_each_result(self, plugin, escape_workspace):
        workspace, _ = escape_workspace
        chunks = await self._collect(
            plugin, "glob_files", {"pattern": "data/logs/*.txt", "root": str(workspace)}
        )
        files = [c for c in chunks if c.chunk_type == "file"]
        assert files == []


class TestTempAllowanceDoesNotLeak:
    """The same boundary, with the /tmp allowance left at its default.

    Every fixture above sets ``allow_tmp: False`` so that the symlink
    behaviour is isolated from the (deliberate, unrelated) temp allowance.
    That isolation also hid a real leak: with ``allow_tmp`` at its **default
    of True** and the workspace under a temp root, a leaf symlink pointing
    outside was admitted, because the allowance was decided on where the link
    lived rather than where it pointed.  The content returned was outside both
    the workspace and /tmp — and ``absolute_path`` in the result reported the
    symlink's own contained-looking path, so a caller re-checking the result
    would not have caught it either.

    Reported by the jaato-ac review session against jaato issue #669.
    """

    @pytest.fixture
    def tmp_rooted_workspace(self, tmp_path, monkeypatch):
        """A workspace *inside* a substituted temp root, leaking outward.

        The real temp root cannot be used: ``tmp_path`` is itself under
        ``/tmp`` on Linux, so the "outside" secret would resolve into the
        allowance and the test would pass for the wrong reason.

        Returns:
            The workspace path.
        """
        temp_root = tmp_path / "faketmp"
        temp_root.mkdir()
        monkeypatch.setattr(
            "shared.plugins.sandbox_utils.SYSTEM_TEMP_PATHS", [str(temp_root)]
        )

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("SECRET\n")

        workspace = temp_root / "ws"
        workspace.mkdir()
        (workspace / "ordinary.txt").write_text("ordinary\n")
        (workspace / "home_leaf.txt").symlink_to(outside / "secret.txt")
        return workspace

    @pytest.fixture
    def tmp_plugin(self, tmp_rooted_workspace):
        """A plugin with ``allow_tmp`` left at its default of True."""
        p = FilesystemQueryPlugin()
        p.initialize({})
        p.set_workspace_path(str(tmp_rooted_workspace))
        yield p
        p.shutdown()

    def test_grep_with_no_path_does_not_leak(self, tmp_plugin):
        """The reported repro, verbatim: no ``path`` argument at all."""
        result = tmp_plugin._execute_grep_content({"pattern": "SECRET"})
        assert result["total_matches"] == 0, result["matches"]

    def test_grep_still_finds_workspace_content(self, tmp_plugin):
        result = tmp_plugin._execute_grep_content({"pattern": "ordinary"})
        assert result["total_matches"] == 1, result

    def test_glob_does_not_report_the_leaking_link(self, tmp_plugin, tmp_rooted_workspace):
        result = tmp_plugin._execute_glob_files(
            {"pattern": "*.txt", "root": str(tmp_rooted_workspace)}
        )
        found = _paths(result)
        assert any(p.endswith("ordinary.txt") for p in found), found
        assert not any(p.endswith("home_leaf.txt") for p in found), found
