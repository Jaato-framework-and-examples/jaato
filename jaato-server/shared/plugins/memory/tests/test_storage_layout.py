"""Tests for the raw + curated split storage layout.

Direct tests against ``RawStore``, ``CuratedStore``, and the
``MemoryStore`` facade.  Complements ``test_plugin.py``'s
``TestSplitStorage`` (which exercises the same surface through the
plugin's storage facade).
"""

import json
import tempfile
from pathlib import Path

import pytest

from shared.plugins.memory.models import (
    MATURITY_RAW,
    MATURITY_VALIDATED,
    Memory,
)
from shared.plugins.memory.storage import (
    CuratedStore,
    MemoryStore,
    RawStore,
)


def _mem(**kwargs) -> Memory:
    defaults = {
        "id": "m",
        "content": "c",
        "description": "d",
        "tags": ["t"],
        "timestamp": "2024-01-01T00:00:00",
        "maturity": MATURITY_RAW,
    }
    defaults.update(kwargs)
    return Memory(**defaults)


# ─────────────────────────────────────────────────────────────────────
# RawStore
# ─────────────────────────────────────────────────────────────────────


class TestRawStore:

    def test_add_creates_file_per_memory(self, tmp_path):
        store = RawStore(tmp_path)
        store.add(_mem(id="mem_a"))
        store.add(_mem(id="mem_b"))
        files = sorted(p.name for p in (tmp_path / "raw").iterdir())
        assert files == ["mem_a.json", "mem_b.json"]

    def test_atomic_write_no_temp_files_left(self, tmp_path):
        """tempfile + rename should leave no .tmp residue on success."""
        store = RawStore(tmp_path)
        store.add(_mem(id="mem_x"))
        leftover = [p for p in (tmp_path / "raw").iterdir() if p.suffix == ".tmp"]
        assert leftover == []

    def test_get_returns_none_for_missing(self, tmp_path):
        store = RawStore(tmp_path)
        assert store.get("does_not_exist") is None

    def test_remove_returns_false_for_missing(self, tmp_path):
        store = RawStore(tmp_path)
        assert store.remove("does_not_exist") is False

    def test_list_all_tolerates_concurrent_deletion(self, tmp_path):
        """Simulate a curator unlinking a file mid-enumeration."""
        store = RawStore(tmp_path)
        store.add(_mem(id="mem_1"))
        store.add(_mem(id="mem_2"))
        # Hand-tamper: delete one file before listing.
        (tmp_path / "raw" / "mem_1.json").unlink()
        result = store.list_all()
        assert len(result) == 1
        assert result[0].id == "mem_2"

    def test_list_all_empty_when_dir_missing(self, tmp_path):
        store = RawStore(tmp_path)
        assert store.list_all() == []
        assert store.count() == 0


# ─────────────────────────────────────────────────────────────────────
# CuratedStore
# ─────────────────────────────────────────────────────────────────────


class TestCuratedStore:

    def test_upsert_appends_then_replaces(self, tmp_path):
        store = CuratedStore(tmp_path)
        store.upsert(_mem(id="mem_1", description="first"))
        store.upsert(_mem(id="mem_2", description="second"))
        memories = store.load_all()
        assert {m.id for m in memories} == {"mem_1", "mem_2"}
        # Replace mem_1
        store.upsert(_mem(id="mem_1", description="updated"))
        memories = store.load_all()
        assert len(memories) == 2  # not duplicated
        m1 = next(m for m in memories if m.id == "mem_1")
        assert m1.description == "updated"

    def test_remove_returns_false_for_missing(self, tmp_path):
        store = CuratedStore(tmp_path)
        assert store.remove("never_existed") is False

    def test_remove_rewrites_atomically(self, tmp_path):
        store = CuratedStore(tmp_path)
        for i in range(3):
            store.upsert(_mem(id=f"mem_{i}"))
        assert store.remove("mem_1") is True
        ids = {m.id for m in store.load_all()}
        assert ids == {"mem_0", "mem_2"}

    def test_load_all_tolerates_corrupt_lines(self, tmp_path):
        path = tmp_path / "curated.jsonl"
        path.write_text(
            json.dumps({"id": "good_1", "content": "x", "description": "y",
                        "tags": ["t"], "timestamp": "2024-01-01T00:00:00"}) + "\n"
            "{this is not json}\n" +
            json.dumps({"id": "good_2", "content": "x", "description": "y",
                        "tags": ["t"], "timestamp": "2024-01-02T00:00:00"}) + "\n"
        )
        store = CuratedStore(tmp_path)
        ids = {m.id for m in store.load_all()}
        assert ids == {"good_1", "good_2"}

    def test_load_all_empty_when_file_missing(self, tmp_path):
        store = CuratedStore(tmp_path)
        assert store.load_all() == []


# ─────────────────────────────────────────────────────────────────────
# MemoryStore facade
# ─────────────────────────────────────────────────────────────────────


class TestMemoryStoreFacade:

    def test_jsonl_path_routes_to_sibling_directory(self, tmp_path):
        """Legacy ``.jsonl`` config paths get a folder named after the stem."""
        store = MemoryStore(str(tmp_path / "memories.jsonl"))
        assert store.base_dir == tmp_path / "memories"

    def test_save_lands_in_raw(self, tmp_path):
        store = MemoryStore(str(tmp_path / "ws.jsonl"))
        store.save(_mem(id="mem_x"))
        assert store.list_raw()[0].id == "mem_x"
        assert store.load_curated() == []

    def test_full_curation_flow(self, tmp_path):
        store = MemoryStore(str(tmp_path / "ws.jsonl"))
        m = _mem(id="mem_y", maturity=MATURITY_RAW)
        store.save(m)
        # Curator validates → moves to curated.
        m.maturity = MATURITY_VALIDATED
        store.update(m)
        assert store.list_raw() == []
        assert len(store.load_curated()) == 1

    def test_dismiss_via_update_unlinks_raw(self, tmp_path):
        store = MemoryStore(str(tmp_path / "ws.jsonl"))
        m = _mem(id="mem_z")
        store.save(m)
        m.maturity = "dismissed"
        store.update(m)
        assert store.list_raw() == []
        assert store.load_curated() == []

    def test_dismiss_via_update_removes_from_curated(self, tmp_path):
        store = MemoryStore(str(tmp_path / "ws.jsonl"))
        m = _mem(id="mem_z")
        store.save(m)
        m.maturity = MATURITY_VALIDATED
        store.update(m)
        # Now dismiss.
        m.maturity = "dismissed"
        store.update(m)
        assert store.load_curated() == []

    def test_search_by_tags_curated_only(self, tmp_path):
        """search_by_tags must NOT surface raw memories."""
        store = MemoryStore(str(tmp_path / "ws.jsonl"))
        # Two memories: one stays raw, one gets validated.
        store.save(_mem(id="mem_raw", tags=["topic"]))
        m = _mem(id="mem_validated", tags=["topic"])
        store.save(m)
        m.maturity = MATURITY_VALIDATED
        store.update(m)
        results = store.search_by_tags(["topic"])
        assert len(results) == 1
        assert results[0].id == "mem_validated"

    def test_search_by_maturity_raw_uses_raw_store(self, tmp_path):
        store = MemoryStore(str(tmp_path / "ws.jsonl"))
        store.save(_mem(id="mem_raw_1"))
        store.save(_mem(id="mem_raw_2"))
        results = store.search_by_maturity({MATURITY_RAW})
        assert {m.id for m in results} == {"mem_raw_1", "mem_raw_2"}

    def test_atomic_write_survives_partial_disk_state(self, tmp_path):
        """If the curated file exists from a previous run, an atomic
        rewrite produces the new file with no transient half-state
        visible to readers."""
        store = MemoryStore(str(tmp_path / "ws.jsonl"))
        # Pre-populate.
        for i in range(5):
            m = _mem(id=f"m_{i}")
            store.save(m)
            m.maturity = MATURITY_VALIDATED
            store.update(m)
        # Now remove one.
        store.delete("m_2")
        ids = {m.id for m in store.load_curated()}
        assert ids == {"m_0", "m_1", "m_3", "m_4"}

    def test_count_combines_both_stores(self, tmp_path):
        store = MemoryStore(str(tmp_path / "ws.jsonl"))
        store.save(_mem(id="raw_a"))
        m = _mem(id="curated_b")
        store.save(m)
        m.maturity = MATURITY_VALIDATED
        store.update(m)
        assert store.count() == 2
