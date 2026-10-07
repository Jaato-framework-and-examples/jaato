"""``updateFile`` / ``writeNewFile`` attach a capped unified diff to their
own RESULT (jaato/#1304 phase 3, "diff (unified, capped) and path on
tool.call_end for file writers"), reusing the same
``diff_utils.generate_unified_diff`` / ``generate_new_file_diff`` the
permission-ask card has always used -- computed at the executor because
that is the only place that still has the file's "before" content, and
because a write reaching the executor did not necessarily go through an
ASK at all (whitelist / allow_all / turn-or-idle suspension /
auto_allow_housekeeping never call the ``_format_*`` preview methods).
"""

from __future__ import annotations

from jaato_server.shared.plugins.file_edit.plugin import FileEditPlugin


class TestUpdateFileResultDiff:
    def test_targeted_edit_result_carries_a_diff(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.py"
        test_file.write_text("line one\nline two\nline three\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "old": "line two\n",
            "new": "line TWO\n",
        })

        assert result["success"] is True
        assert "diff" in result
        assert "-line two" in result["diff"]
        assert "+line TWO" in result["diff"]
        assert result["diff_truncated"] is False
        assert "diff_total_lines" not in result

    def test_full_replacement_result_also_carries_a_diff(self, tmp_path):
        """Full-replacement mode never reads the old content for its OWN
        purposes -- the diff read has to happen anyway, and must not
        break the write it rides along with."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content",
        })

        assert result["success"] is True
        assert test_file.read_text() == "Updated content"
        assert "-Original content" in result["diff"]
        assert "+Updated content" in result["diff"]

    def test_long_diff_is_capped_and_says_so(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "long.txt"
        old_lines = "\n".join(f"line {i}" for i in range(200)) + "\n"
        test_file.write_text(old_lines)
        new_lines = "\n".join(f"LINE {i}" for i in range(200)) + "\n"

        result = plugin._execute_update_file({
            "path": str(test_file),
            "new_content": new_lines,
        })

        assert result["success"] is True
        assert result["diff_truncated"] is True
        assert result["diff_total_lines"] > 50  # DEFAULT_MAX_LINES
        assert result["diff"].count("\n") < result["diff_total_lines"]

    def test_a_read_failure_for_the_diff_does_not_fail_the_write(self, tmp_path, monkeypatch):
        """The diff read is best-effort: if it cannot be produced, the
        write that already succeeded must not be reported as a failure,
        and the result simply carries no diff."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        original_load = plugin._line_endings.load
        calls = {"n": 0}

        def _flaky_load(path, validate=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # The targeted-mode branch is not taken here (full
                # replacement), so this is the diff-only read.
                raise OSError("simulated read failure")
            return original_load(path, validate=validate)

        monkeypatch.setattr(plugin._line_endings, "load", _flaky_load)

        result = plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content",
        })

        assert result["success"] is True
        assert test_file.read_text() == "Updated content"
        assert "diff" not in result


class TestWriteNewFileResultDiff:
    def test_new_file_result_carries_an_all_additions_diff(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "new.py"

        result = plugin._execute_write_new_file({
            "path": str(new_file),
            "content": "import os\nprint(os.getcwd())\n",
        })

        assert result["success"] is True
        assert "diff" in result
        assert "+import os" in result["diff"]
        assert "/dev/null" in result["diff"]
        assert result["diff_truncated"] is False

    def test_long_new_file_diff_is_capped(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "big.py"
        content = "\n".join(f"x = {i}" for i in range(200)) + "\n"

        result = plugin._execute_write_new_file({
            "path": str(new_file),
            "content": content,
        })

        assert result["diff_truncated"] is True
        assert result["diff_total_lines"] > 50


class TestOtherWriteToolsDoNotClaimADiff:
    def test_remove_file_result_carries_no_diff_key(self, tmp_path):
        """removeFile does not declare TRAIT_FILE_WRITER today, and this
        pins the negative: nothing invents a diff for a tool that never
        computed one."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "gone.txt"
        test_file.write_text("bye")

        result = plugin._execute_remove_file({"path": str(test_file)})

        assert result["success"] is True
        assert "diff" not in result
