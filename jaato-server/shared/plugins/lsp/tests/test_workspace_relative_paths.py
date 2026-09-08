"""A caller-supplied path is resolved against the SESSION workspace, not cwd.

`enrich_tool_result` takes the path a writing tool reported, and `file_edit`
reports a WORKSPACE-RELATIVE one.  Unresolved, it is opened against the
daemon's cwd — so on a live cascade every `updateFile` produced
`get_diagnostics ERROR - [Errno 2] No such file or directory:
'src/main/java/.../Customer.java'`, while the sibling `artifact_tracker`
enrichment resolved the same string on the same result one line earlier.

`_call_lsp_method` is the chokepoint every file-based method passes through —
diagnostics, hover, goto_definition, find_references, document_symbols,
rename — so the resolution is asserted there, once, for all of them.
"""
import asyncio
import os

import pytest
from unittest.mock import patch

from ..plugin import LSPToolPlugin


class RecordingClient:
    """Captures the paths the plugin actually hands the LSP client."""

    def __init__(self):
        self.opened = []
        self.diagnostics_for = None
        self.symbols_for = None

    async def update_document(self, file_path):
        self.opened.append(file_path)

    async def await_diagnostics(self, file_path, **kwargs):
        return None

    def get_diagnostics(self, file_path):
        self.diagnostics_for = file_path
        return []

    async def get_document_symbols(self, file_path):
        self.symbols_for = file_path
        return []


def _plugin(workspace):
    plugin = LSPToolPlugin()
    with patch.object(plugin, "_ensure_thread"):
        plugin.initialize({"workspace_path": str(workspace)})
    return plugin


def _call(plugin, client, method, args):
    return asyncio.run(plugin._call_lsp_method(client, method, args))


def test_a_relative_path_reaches_the_server_resolved(tmp_path):
    """The `file_edit` case: the exact shape that failed on the live run."""
    plugin, client = _plugin(tmp_path), RecordingClient()
    rel = "src/main/java/com/acme/Customer.java"

    _call(plugin, client, "get_diagnostics", {"file_path": rel})

    assert client.opened == [os.path.join(str(tmp_path), rel)]
    assert client.diagnostics_for == os.path.join(str(tmp_path), rel)


def test_an_absolute_path_is_passed_through_untouched(tmp_path):
    """A model may hand in an absolute path; resolving it again would be wrong."""
    plugin, client = _plugin(tmp_path), RecordingClient()
    absolute = str(tmp_path / "src" / "A.java")

    _call(plugin, client, "get_diagnostics", {"file_path": absolute})

    assert client.diagnostics_for == absolute


def test_every_file_based_method_gets_the_same_treatment(tmp_path):
    """The fix is at the chokepoint, so it must hold for more than diagnostics."""
    plugin, client = _plugin(tmp_path), RecordingClient()

    _call(plugin, client, "document_symbols", {"file_path": "src/A.java"})

    assert client.symbols_for == os.path.join(str(tmp_path), "src/A.java")


def test_without_a_workspace_the_path_is_left_alone(tmp_path):
    """No workspace means nothing to resolve against.

    `_resolve_path` deliberately does NOT fall back to `os.path.abspath`
    here: in daemon mode that resolves against the server's own directory,
    which is how a session reads a file belonging to no one.
    """
    plugin = LSPToolPlugin()
    with patch.object(plugin, "_ensure_thread"):
        plugin.initialize({})
    client = RecordingClient()

    _call(plugin, client, "get_diagnostics", {"file_path": "src/A.java"})

    assert client.diagnostics_for == "src/A.java"
