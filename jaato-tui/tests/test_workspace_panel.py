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


class TestHideAndShowHidden:
    """``hide_at_cursor`` / ``toggle_show_hidden`` filter behavior."""

    def _visible_ids(self, panel):
        flat = panel._get_flat_entries()
        ids = []
        for e in flat:
            if e["is_dir"]:
                ids.append(e["dir_path"])
            else:
                ids.append(f"{e.get('parent_dir', '')}{e['name']}")
        return ids

    def test_hide_file_removes_it_from_listing(self, panel):
        panel._cursor_index = 1  # server/core.py
        toggled = panel.hide_at_cursor()
        assert toggled == "server/core.py"
        assert "server/core.py" not in self._visible_ids(panel)
        assert "server/util.py" in self._visible_ids(panel)

    def test_hide_directory_hides_descendants(self, panel):
        panel._cursor_index = 0  # "server/" directory
        toggled = panel.hide_at_cursor()
        assert toggled == "server/"
        ids = self._visible_ids(panel)
        assert "server/" not in ids
        assert "server/core.py" not in ids
        assert "server/util.py" not in ids
        # Sibling top-level file untouched
        assert "README.md" in ids

    def test_show_hidden_brings_them_back_with_marker(self, panel):
        panel._cursor_index = 1
        panel.hide_at_cursor()
        assert "server/core.py" not in self._visible_ids(panel)

        panel.toggle_show_hidden()
        flat = panel._get_flat_entries()
        hidden_entries = [e for e in flat if e.get("hidden")]
        assert any(
            f"{e.get('parent_dir', '')}{e['name']}" == "server/core.py"
            for e in hidden_entries
        )

    def test_hide_at_cursor_on_already_hidden_unhides(self, panel):
        # Hide, then enable show_hidden, then re-toggle to unhide.
        panel._cursor_index = 1
        panel.hide_at_cursor()
        assert "server/core.py" in panel.hidden_paths

        panel.toggle_show_hidden()
        # Find the now-visible hidden entry and place cursor on it.
        flat = panel._get_flat_entries()
        for i, e in enumerate(flat):
            if not e["is_dir"] and e["name"] == "core.py" and e.get("hidden"):
                panel._cursor_index = i
                break
        else:
            pytest.fail("hidden entry not found in show-hidden mode")

        panel.hide_at_cursor()
        assert "server/core.py" not in panel.hidden_paths

    def test_empty_panel_hide_returns_none(self):
        assert WorkspacePanel().hide_at_cursor() is None
