"""The documented ``storage_path`` must BE the path the store uses (#912).

``MemoryStore`` accepts a base **directory** and rewrites a legacy
``*.jsonl`` path to the sibling directory named after its stem.  Every
authoring surface nevertheless defaulted to — and described — the file
form, so the documented happy path was the one whose meaning the
constructor silently changed: a populated ``.jaato/memories.jsonl`` read
back as ZERO memories, indistinguishable from an empty store.

Two properties are pinned here, and they pull in opposite directions:

- the advertised default must be a path the constructor does **not**
  rewrite, so ``jaato-scaffold explain plugin memory`` cannot hand an
  author a spelling that means something else;
- the suffix rule must **stay**, because profiles in the wild carry an
  explicit ``storage_path: /x/y/mem.jsonl`` and removing it would repoint
  them at a fresh empty store — this issue's failure mode, inflicted
  deliberately.

What closes the silent part is that a populated legacy file is no longer
ignored without a word: it is migrated into ``curated.jsonl`` when that
would not clobber anything, and announced at WARNING when it would.
"""

import json
import logging
from pathlib import Path

import pytest

from shared.plugins.memory.plugin import MemoryPlugin
from shared.plugins.memory.storage import MemoryStore

_REC = {
    "id": "mem_legacy",
    "content": "c",
    "description": "una memoria real",
    "tags": ["t"],
    "timestamp": "2026-09-01T00:00:00",
    "usage_count": 0,
    "last_accessed": None,
    "maturity": "validated",
    "confidence": 1.0,
    "scope": "project",
    "evidence": None,
    "source_agent": None,
    "source_session": None,
}


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestAdvertisedDefaultIsTheEffectivePath:
    """The load-bearing pair: what `explain` prints vs what the code opens."""

    def test_schema_default_is_the_directory_the_constructor_uses(self):
        """A default the constructor rewrites is a default that lies.

        This is the assertion that stops #912 recurring: any future
        ``storage_path`` default which ``MemoryStore`` reinterprets fails
        the build instead of reaching an author through
        ``jaato-scaffold explain plugin memory``.
        """
        default = (
            MemoryPlugin()
            .get_config_schema()["properties"]["storage_path"]["default"]
        )
        assert MemoryStore(default).base_dir == Path(default)

    def test_runtime_default_matches_the_advertised_one(self, tmp_path,
                                                        monkeypatch):
        """``initialize()``'s fallback and the schema must agree.

        They are two separate literals in ``plugin.py``; nothing but this
        test keeps them equal, and the schema is the one an author reads.
        """
        monkeypatch.chdir(tmp_path)
        plugin = MemoryPlugin()
        plugin.initialize({"global_storage_path": str(tmp_path / "global")})
        advertised = (
            plugin.get_config_schema()["properties"]["storage_path"]["default"]
        )
        assert plugin._storage_path_template == advertised

    def test_schema_description_no_longer_calls_it_a_file(self):
        """``explain`` repeats this string verbatim, so it must be true."""
        described = (
            MemoryPlugin()
            .get_config_schema()["properties"]["storage_path"]["description"]
        )
        assert "file" not in described.lower()


class TestLegacyJsonlPath:
    """The suffix rule stays; what it does to real data changes."""

    def test_jsonl_still_routes_to_sibling_directory(self, tmp_path):
        """Explicit ``.jsonl`` profiles keep resolving where they always did.

        Removing this rule would repoint every such deployment at a fresh
        empty store — the very failure #912 is about.
        """
        store = MemoryStore(str(tmp_path / "memories.jsonl"))
        assert store.base_dir == tmp_path / "memories"

    def test_populated_legacy_file_is_recovered_and_announced(
        self, tmp_path, caplog
    ):
        """A file with memories in it must not read back as an empty store.

        Recovery can only GAIN data: a populated flat file can exist only
        on a pre-split install, where it has been unreadable ever since.
        """
        legacy = tmp_path / "memories.jsonl"
        legacy.write_text(json.dumps(_REC) + "\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            store = MemoryStore(str(legacy))
            loaded = store.load_curated()

        assert [m.id for m in loaded] == ["mem_legacy"]
        assert any(str(legacy) in r.getMessage() for r in _warnings(caplog))
        assert legacy.exists(), "the caller's file must be left in place"

    def test_existing_curated_store_is_never_clobbered(self, tmp_path, caplog):
        """Both populated: warn, change nothing.

        Merging two independently-populated stores by id is a judgement a
        constructor must not make silently.
        """
        legacy = tmp_path / "memories.jsonl"
        legacy.write_text(json.dumps(_REC) + "\n", encoding="utf-8")
        current = dict(_REC, id="mem_current")
        curated = tmp_path / "memories" / "curated.jsonl"
        curated.parent.mkdir(parents=True)
        curated.write_text(json.dumps(current) + "\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            store = MemoryStore(str(legacy))

        assert {m.id for m in store.load_curated()} == {"mem_current"}
        assert any("NOT read" in r.getMessage() for r in _warnings(caplog))

    def test_absent_and_empty_legacy_files_are_silent(self, tmp_path, caplog):
        """Silence must mean something, so the normal case stays quiet."""
        (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            MemoryStore(str(tmp_path / "empty.jsonl"))
            MemoryStore(str(tmp_path / "never-existed.jsonl"))
            MemoryStore(str(tmp_path / "a-directory"))

        assert _warnings(caplog) == []

    def test_directory_path_never_looks_for_a_legacy_file(self, tmp_path):
        """A sibling file is not this store's business.

        ``.jaato/memories`` and ``.jaato/memories.jsonl`` can coexist on a
        half-migrated install; only the spelling the caller passed selects
        the legacy read.
        """
        (tmp_path / "memories.jsonl").write_text(
            json.dumps(_REC) + "\n", encoding="utf-8"
        )
        store = MemoryStore(str(tmp_path / "memories"))
        assert store.load_curated() == []


class TestMigrationIsNotDestructive:

    def test_migration_happens_once_and_is_idempotent(self, tmp_path):
        """A second construction must not duplicate the recovered rows."""
        legacy = tmp_path / "memories.jsonl"
        legacy.write_text(json.dumps(_REC) + "\n", encoding="utf-8")

        MemoryStore(str(legacy))
        second = MemoryStore(str(legacy))
        assert [m.id for m in second.load_curated()] == ["mem_legacy"]

    def test_corrupt_lines_are_skipped_not_fatal(self, tmp_path):
        """Tolerance mirrors ``CuratedStore.load_all``: skip, never crash."""
        legacy = tmp_path / "memories.jsonl"
        legacy.write_text(
            "{not json\n" + json.dumps(_REC) + "\n", encoding="utf-8"
        )
        assert [m.id for m in MemoryStore(str(legacy)).load_curated()] == [
            "mem_legacy"
        ]

    def test_unreadable_legacy_path_does_not_raise(self, tmp_path,
                                                   monkeypatch):
        """A confined session is *correctly* denied this path.

        ``initialize()`` builds a store from a daemon-cwd-relative template
        and from HOME, both of which a confined runner may not open.  A
        constructor that raised there would disable the plugin outright.
        """
        def _boom(self):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "is_file", _boom)
        store = MemoryStore(str(tmp_path / "memories.jsonl"))
        assert store.base_dir == tmp_path / "memories"
