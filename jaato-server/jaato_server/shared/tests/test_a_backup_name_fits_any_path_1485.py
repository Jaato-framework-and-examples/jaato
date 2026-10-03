"""A file on a deep path can be edited, and undone (#1485).

``BackupManager`` named each backup after the RESOLVED ABSOLUTE path
flattened into one component, plus ``_<timestamp>.bak``.  A path longer
than about 224 bytes therefore produced a name over the 255-byte
component limit, the backup write raised ``ENAMETOOLONG`` and
``updateFile`` failed before editing anything.  Whether a session hit it
depended on where the workspace was checked out.

The name is now ``{readable tail}~{digest}_{timestamp}.bak``: the tail is
truncated so the whole name fits ``BACKUP_NAME_BUDGET`` bytes, and the
digest of the full path is what identifies the file.  Backups written in
the old format are still found.
"""

from __future__ import annotations

import os
from pathlib import Path

from jaato_server.shared.plugins.file_edit.backup import (
    BACKUP_NAME_BUDGET,
    BackupManager,
)
from jaato_server.shared.plugins.file_edit.plugin import FileEditPlugin
from jaato_server.shared.tests.reversion import Reversion

_BACKUP = "jaato-server/jaato_server/shared/plugins/file_edit/backup.py"

REVERSIONS = [
    Reversion(
        target=_BACKUP,
        find=(
            "        stem = self._backup_stem(file_path)\n"
            "        timestamp = datetime.now().strftime(self._BACKUP_TIMESTAMP_FMT)\n"
        ),
        replace=(
            "        stem = self._sanitize_path(file_path)\n"
            "        timestamp = datetime.now().strftime(self._BACKUP_TIMESTAMP_FMT)\n"
        ),
        test="test_update_file_on_a_deep_path_succeeds_and_backs_up",
        because="the name is the whole absolute path again: ENAMETOOLONG",
    ),
    Reversion(
        target=_BACKUP,
        find="            self._readable_tail(resolved), _TAIL_BUDGET,\n",
        replace="            self._readable_tail(resolved), 10_000,\n",
        test="test_every_backup_name_fits_the_budget_in_bytes",
        because="the readable part is no longer truncated",
    ),
    Reversion(
        target=_BACKUP,
        find="    cut = encoded[len(encoded) - budget:]\n",
        replace="    return text[-budget:]\n",
        test="test_every_backup_name_fits_the_budget_in_bytes",
        because="the truncation counts characters, not UTF-8 bytes",
    ),
    Reversion(
        target=_BACKUP,
        find="        stems = (self._backup_stem(file_path), self._sanitize_path(file_path))\n",
        replace="        stems = (self._backup_stem(file_path),)\n",
        test="test_a_legacy_backup_is_still_found_and_restored",
        because="an upgrade orphans every backup written in the old format",
    ),
    Reversion(
        target=_BACKUP,
        find="        return f\"{tail}{_HASH_SEP}{self._path_digest(resolved)}\"\n",
        replace="        return f\"{tail}{_HASH_SEP}{'0' * BACKUP_HASH_HEX}\"\n",
        test="test_two_long_paths_with_a_common_suffix_get_different_names",
        because="without the digest, two truncated tails collide",
    ),
]


def _deep_file(root: Path, *, segment: str = "segment", levels: int = 8) -> Path:
    """A file whose absolute path is well over 255 bytes."""
    deep = root
    for i in range(levels):
        deep = deep / f"{segment}_{i:02d}_{'x' * 30}"
    deep.mkdir(parents=True)
    target = deep / "Fixture.java"
    target.write_text("class Fixture { int v = 1; }\n", encoding="utf-8")
    assert len(str(target.resolve()).encode("utf-8")) > 255
    return target


def _plugin(tmp_path: Path) -> FileEditPlugin:
    plugin = FileEditPlugin()
    plugin.initialize({"backup_dir": str(tmp_path / "backups")})
    return plugin


def _names(tmp_path: Path):
    return [p.name for p in (tmp_path / "backups").glob("*.bak")]


def test_update_file_on_a_deep_path_succeeds_and_backs_up(tmp_path):
    target = _deep_file(tmp_path / "ws")
    plugin = _plugin(tmp_path)

    result = plugin._execute_update_file({
        "path": str(target),
        "old": "int v = 1;",
        "new": "int v = 2;",
    })

    assert "error" not in result, result
    assert "int v = 2;" in target.read_text(encoding="utf-8")
    assert len(_names(tmp_path)) == 1


def test_every_backup_name_fits_the_budget_in_bytes(tmp_path):
    ascii_file = _deep_file(tmp_path / "ascii")
    # Three-byte characters: a character-count truncation would overrun.
    multibyte = _deep_file(tmp_path / "mb", segment="日本語のディレクトリ", levels=6)
    manager = BackupManager(tmp_path / "backups")

    for target in (ascii_file, multibyte):
        for _ in range(3):
            assert manager.create_backup(target) is not None

    names = _names(tmp_path)
    assert len(names) == 6
    for name in names:
        assert len(name.encode("utf-8")) <= BACKUP_NAME_BUDGET, name
        name.encode("utf-8").decode("utf-8")  # no split character


def test_undo_restores_a_deep_file_from_a_new_format_backup(tmp_path):
    target = _deep_file(tmp_path / "ws")
    plugin = _plugin(tmp_path)
    plugin._execute_update_file({
        "path": str(target), "old": "int v = 1;", "new": "int v = 2;",
    })

    result = plugin._execute_undo_file_change({"path": str(target)})

    assert result.get("success") is True, result
    assert "int v = 1;" in target.read_text(encoding="utf-8")


def test_a_legacy_backup_is_still_found_and_restored(tmp_path):
    target = tmp_path / "ws" / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("new\n", encoding="utf-8")
    backups = tmp_path / "backups"
    backups.mkdir()
    legacy_prefix = str(target.resolve()).replace(os.sep, "_").lstrip("_")
    legacy = backups / f"{legacy_prefix}_2026-01-02T03-04-05-123456.bak"
    legacy.write_text("old\n", encoding="utf-8")
    manager = BackupManager(backups)

    assert manager.list_backups(target) == [legacy]
    listed = manager.list_all_backups()
    assert [b.backup_path for b in listed] == [legacy]
    assert manager.restore_from_backup(target)
    assert target.read_text(encoding="utf-8") == "old\n"


def test_two_long_paths_with_a_common_suffix_get_different_names(tmp_path):
    first = _deep_file(tmp_path / "alpha")
    second = _deep_file(tmp_path / "beta")
    manager = BackupManager(tmp_path / "backups")

    a = manager.create_backup(first)
    b = manager.create_backup(second)

    stem_a = a.name.rsplit("_", 1)[0]
    stem_b = b.name.rsplit("_", 1)[0]
    assert stem_a != stem_b
    assert manager.list_backups(first) == [a]
    assert manager.list_backups(second) == [b]


def test_a_workspace_relative_tail_and_the_recorded_path(tmp_path):
    workspace = tmp_path / "ws"
    target = workspace / "pkg" / "mod.py"
    target.parent.mkdir(parents=True)
    target.write_text("x\n", encoding="utf-8")
    manager = BackupManager(tmp_path / "backups", workspace_root=str(workspace))

    backup = manager.create_backup(target)

    assert backup.name.startswith("pkg_mod.py~")
    listed = manager.list_all_backups()
    assert listed[0].original_path == str(target.resolve())
