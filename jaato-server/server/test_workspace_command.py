"""Tests for the workspace command handler and formatters."""

import csv
import io
import json
import pytest

from server.workspace_command import (
    handle_workspace_command,
    format_tree,
    format_list,
    format_json,
    format_csv,
    _VALID_SUBCOMMANDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SNAPSHOT = [
    {"path": "src/main.py", "status": "modified"},
    {"path": "src/utils/helper.py", "status": "created"},
    {"path": "README.md", "status": "deleted"},
    {"path": "src/utils/constants.py", "status": "created"},
    {"path": "docs/guide.md", "status": "modified"},
]


class FakeMonitor:
    """Minimal mock of WorkspaceMonitor for testing."""

    def __init__(self, snapshot):
        self._snapshot = snapshot

    def get_snapshot(self):
        return list(self._snapshot)

    @property
    def active_file_count(self):
        return sum(1 for e in self._snapshot if e["status"] != "deleted")


# ---------------------------------------------------------------------------
# handle_workspace_command
# ---------------------------------------------------------------------------

class TestHandleWorkspaceCommand:
    """Tests for the top-level command dispatcher."""

    def test_no_monitor(self):
        result = handle_workspace_command(None, ["tree"])
        assert "error" in result
        assert "No workspace monitor" in result["error"]

    def test_unknown_subcommand(self):
        monitor = FakeMonitor(SAMPLE_SNAPSHOT)
        result = handle_workspace_command(monitor, ["unknown"])
        assert "error" in result
        assert "Unknown workspace subcommand" in result["error"]
        assert "tree" in result["error"]  # shows valid subcommands

    def test_default_subcommand_is_tree(self):
        monitor = FakeMonitor(SAMPLE_SNAPSHOT)
        result = handle_workspace_command(monitor, [])
        assert "result" in result
        # Tree format starts with the header line
        assert "Workspace changes" in result["result"]

    def test_empty_snapshot(self):
        monitor = FakeMonitor([])
        result = handle_workspace_command(monitor, ["tree"])
        assert "result" in result
        assert "No workspace file changes" in result["result"]

    @pytest.mark.parametrize("subcommand", list(_VALID_SUBCOMMANDS))
    def test_all_subcommands_produce_result(self, subcommand):
        monitor = FakeMonitor(SAMPLE_SNAPSHOT)
        result = handle_workspace_command(monitor, [subcommand])
        assert "result" in result
        assert len(result["result"]) > 0

    def test_subcommand_case_insensitive(self):
        monitor = FakeMonitor(SAMPLE_SNAPSHOT)
        result = handle_workspace_command(monitor, ["TREE"])
        assert "result" in result
        assert "Workspace changes" in result["result"]


# ---------------------------------------------------------------------------
# format_tree
# ---------------------------------------------------------------------------

class TestFormatTree:
    """Tests for tree format output."""

    def test_header_line(self):
        output = format_tree(SAMPLE_SNAPSHOT)
        first_line = output.split("\n")[0]
        assert "Workspace changes" in first_line
        assert "4 files" in first_line  # 4 active (not counting deleted in main count)

    def test_deleted_count_in_header(self):
        output = format_tree(SAMPLE_SNAPSHOT)
        first_line = output.split("\n")[0]
        assert "1 deleted" in first_line

    def test_directory_grouping(self):
        output = format_tree(SAMPLE_SNAPSHOT)
        assert "src/" in output
        assert "docs/" in output

    def test_status_symbols(self):
        output = format_tree(SAMPLE_SNAPSHOT)
        assert "[+]" in output  # created
        assert "[~]" in output  # modified
        assert "[-]" in output  # deleted

    def test_tree_connectors(self):
        output = format_tree(SAMPLE_SNAPSHOT)
        # Should have tree drawing characters
        assert "├── " in output or "└── " in output

    def test_single_file(self):
        snapshot = [{"path": "foo.txt", "status": "created"}]
        output = format_tree(snapshot)
        assert "1 file)" in output
        assert "foo.txt" in output
        assert "[+]" in output

    def test_nested_directories(self):
        snapshot = [
            {"path": "a/b/c/deep.py", "status": "created"},
        ]
        output = format_tree(snapshot)
        assert "a/" in output
        assert "b/" in output
        assert "c/" in output
        assert "deep.py" in output


# ---------------------------------------------------------------------------
# format_list
# ---------------------------------------------------------------------------

class TestFormatList:
    """Tests for flat list format output."""

    def test_all_files_listed(self):
        output = format_list(SAMPLE_SNAPSHOT)
        for entry in SAMPLE_SNAPSHOT:
            assert entry["path"] in output

    def test_status_symbols(self):
        output = format_list(SAMPLE_SNAPSHOT)
        assert "[+]" in output
        assert "[~]" in output
        assert "[-]" in output

    def test_sorted_output(self):
        output = format_list(SAMPLE_SNAPSHOT)
        lines = [l for l in output.split("\n") if l.startswith("[")]
        paths = [l.split("] ", 1)[1] for l in lines]
        assert paths == sorted(paths)

    def test_summary_line(self):
        output = format_list(SAMPLE_SNAPSHOT)
        assert "5 files" in output
        assert "4 active" in output
        assert "1 deleted" in output

    def test_single_file_summary(self):
        snapshot = [{"path": "x.py", "status": "created"}]
        output = format_list(snapshot)
        assert "1 file" in output
        # Should not say "1 files"
        assert "1 files" not in output


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------

class TestFormatJson:
    """Tests for JSON format output."""

    def test_valid_json(self):
        output = format_json(SAMPLE_SNAPSHOT)
        data = json.loads(output)
        assert "files" in data
        assert "summary" in data

    def test_file_entries(self):
        output = format_json(SAMPLE_SNAPSHOT)
        data = json.loads(output)
        assert len(data["files"]) == len(SAMPLE_SNAPSHOT)
        for entry in data["files"]:
            assert "path" in entry
            assert "status" in entry

    def test_summary_counts(self):
        output = format_json(SAMPLE_SNAPSHOT)
        data = json.loads(output)
        summary = data["summary"]
        assert summary["total"] == 5
        assert summary["created"] == 2
        assert summary["modified"] == 2
        assert summary["deleted"] == 1

    def test_sorted_by_path(self):
        output = format_json(SAMPLE_SNAPSHOT)
        data = json.loads(output)
        paths = [f["path"] for f in data["files"]]
        assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# format_csv
# ---------------------------------------------------------------------------

class TestFormatCsv:
    """Tests for CSV format output."""

    def test_valid_csv(self):
        output = format_csv(SAMPLE_SNAPSHOT)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        # Header + data rows
        assert len(rows) == len(SAMPLE_SNAPSHOT) + 1

    def test_header_row(self):
        output = format_csv(SAMPLE_SNAPSHOT)
        # CSV lines use \r\n per RFC 4180; split and strip
        first_line = output.split("\n")[0].strip()
        assert first_line == "path,status"

    def test_all_entries_present(self):
        output = format_csv(SAMPLE_SNAPSHOT)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        data_rows = rows[1:]  # skip header
        paths = {row[0] for row in data_rows}
        expected_paths = {e["path"] for e in SAMPLE_SNAPSHOT}
        assert paths == expected_paths

    def test_sorted_by_path(self):
        output = format_csv(SAMPLE_SNAPSHOT)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        data_paths = [row[0] for row in rows[1:]]
        assert data_paths == sorted(data_paths)

    def test_no_trailing_newline(self):
        output = format_csv(SAMPLE_SNAPSHOT)
        assert not output.endswith("\n")
