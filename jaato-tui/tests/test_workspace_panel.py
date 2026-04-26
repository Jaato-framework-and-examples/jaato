"""Tests for ``WorkspacePanel`` selection helpers.

Focused on the ``get_selected_path`` / ``get_selected_file_path`` accessors
used by the ``workspace_paste_ref`` keybinding to insert ``@<path>`` into
the input line.
"""

import pytest

from workspace_panel import WorkspacePanel


@pytest.fixture
def panel():
    p = WorkspacePanel()
    p.apply_snapshot([
        {"path": "server/core.py", "status": "modified"},
        {"path": "server/util.py", "status": "modified"},
        {"path": "README.md", "status": "modified"},
    ])
    return p


class TestGetSelectedPath:
    """``get_selected_path`` returns the entry under the cursor, file or dir."""

    def test_empty_panel_returns_none(self):
        assert WorkspacePanel().get_selected_path() is None

    def test_directory_returns_dir_path_with_trailing_slash(self, panel):
        # Tree order: dirs first, so cursor 0 is the "server/" directory.
        panel._cursor_index = 0
        assert panel.get_selected_path() == "server/"

    def test_nested_file_returns_relative_path(self, panel):
        panel._cursor_index = 1
        assert panel.get_selected_path() == "server/core.py"

    def test_top_level_file_returns_bare_name(self, panel):
        panel._cursor_index = 3
        assert panel.get_selected_path() == "README.md"

    def test_out_of_range_returns_none(self, panel):
        panel._cursor_index = 99
        assert panel.get_selected_path() is None


class TestGetSelectedFilePath:
    """``get_selected_file_path`` ignores directories (legacy API)."""

    def test_directory_returns_none(self, panel):
        panel._cursor_index = 0  # "server/" directory
        assert panel.get_selected_file_path() is None

    def test_file_returns_path(self, panel):
        panel._cursor_index = 1
        assert panel.get_selected_file_path() == "server/core.py"
