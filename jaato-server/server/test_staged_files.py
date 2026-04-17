"""Tests for ``_materialize_staged_files`` — the WS session.new inline-file handler.

The helper is the server-side counterpart to ``<jaato-task>.stageAssets()``:
the web component ships files inline on ``session.new`` (instead of via
the HTTP ``/api/task/artifacts`` upload path that depends on the
dashboard's SSO cookie) and the WS handler writes them into the freshly-
provisioned workspace before the session is routed to the command
router.

Because the files arrive from an authenticated but otherwise arbitrary
client, the helper is security-sensitive — path traversal, absolute
paths, and invalid base64 must never reach disk.  These tests pin down
each rejection path.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List

import pytest

from server.websocket import _materialize_staged_files


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode()


class TestMaterializeStagedFilesHappyPath:

    def test_writes_single_file_at_root(self, tmp_path: Path):
        entries = [{"name": "hello.md", "data": _b64(b"# Hi")}]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 1
        assert (tmp_path / "hello.md").read_bytes() == b"# Hi"

    def test_writes_multiple_files_different_directories(self, tmp_path: Path):
        entries = [
            {"name": ".jaato/agents/test.md", "data": _b64(b"agent body")},
            {"name": ".jaato/profiles/test.json", "data": _b64(b'{"k": 1}')},
            {"name": "docs/README.md", "data": _b64(b"docs")},
        ]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 3
        assert (tmp_path / ".jaato/agents/test.md").read_bytes() == b"agent body"
        assert (tmp_path / ".jaato/profiles/test.json").read_bytes() == b'{"k": 1}'
        assert (tmp_path / "docs/README.md").read_bytes() == b"docs"

    def test_creates_deeply_nested_parent_dirs(self, tmp_path: Path):
        entries = [{"name": "a/b/c/d/e/deep.txt", "data": _b64(b"deep")}]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 1
        assert (tmp_path / "a/b/c/d/e/deep.txt").read_bytes() == b"deep"

    def test_empty_list_writes_nothing(self, tmp_path: Path):
        assert _materialize_staged_files(tmp_path, []) == 0
        assert list(tmp_path.iterdir()) == []

    def test_none_writes_nothing(self, tmp_path: Path):
        """Defensive against a ``staged_files: None`` envelope field."""
        assert _materialize_staged_files(tmp_path, None) == 0  # type: ignore[arg-type]

    def test_binary_payload_roundtrips(self, tmp_path: Path):
        blob = bytes(range(256))
        entries = [{"name": "blob.bin", "data": _b64(blob)}]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 1
        assert (tmp_path / "blob.bin").read_bytes() == blob


class TestMaterializeStagedFilesSecurity:

    def test_rejects_absolute_path(self, tmp_path: Path, caplog):
        entries = [{"name": "/etc/passwd", "data": _b64(b"malicious")}]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 0
        # /etc/passwd not written — the only sane assertion on a Linux CI box
        assert "unsafe path" in caplog.text.lower() or any(
            "unsafe path" in r.message.lower() for r in caplog.records
        )

    def test_rejects_parent_traversal_at_start(self, tmp_path: Path):
        entries = [{"name": "../escape.txt", "data": _b64(b"escape")}]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 0
        # Ensure nothing was written above the workspace root.
        assert not (tmp_path.parent / "escape.txt").exists()

    def test_rejects_parent_traversal_in_middle(self, tmp_path: Path):
        entries = [{"name": "safe/../../../escape.txt", "data": _b64(b"escape")}]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 0
        assert not (tmp_path.parent / "escape.txt").exists()

    def test_rejects_invalid_base64(self, tmp_path: Path, caplog):
        entries = [{"name": "legit.md", "data": "this is !!!not valid base64!!!"}]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 0
        assert not (tmp_path / "legit.md").exists()

    def test_valid_entries_survive_alongside_rejected_ones(self, tmp_path: Path):
        """A poisoned entry must not poison the batch — good files still land."""
        entries = [
            {"name": "good.md", "data": _b64(b"good")},
            {"name": "../escape.txt", "data": _b64(b"bad")},
            {"name": "/etc/evil", "data": _b64(b"bad")},
            {"name": "also_good.md", "data": _b64(b"also good")},
            {"name": "invalid.md", "data": "!!!not-b64"},
        ]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 2
        assert (tmp_path / "good.md").read_bytes() == b"good"
        assert (tmp_path / "also_good.md").read_bytes() == b"also good"

    def test_rejects_non_dict_entries(self, tmp_path: Path):
        entries: List = [
            "not a dict",
            42,
            None,
            {"name": "good.md", "data": _b64(b"good")},
        ]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 1
        assert (tmp_path / "good.md").exists()

    def test_skips_entries_with_missing_name(self, tmp_path: Path):
        entries = [
            {"data": _b64(b"orphan")},  # no name
            {"name": "", "data": _b64(b"empty name")},  # empty name
            {"name": "kept.md", "data": _b64(b"kept")},
        ]
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 1
        assert (tmp_path / "kept.md").read_bytes() == b"kept"

    def test_missing_data_treated_as_empty_payload(self, tmp_path: Path):
        """Missing data field → b'' payload.  Creates an empty file; not a
        security risk — the client has to name it within workspace bounds."""
        entries = [{"name": "empty.txt"}]  # no data
        written = _materialize_staged_files(tmp_path, entries)
        assert written == 1
        assert (tmp_path / "empty.txt").read_bytes() == b""
