"""``retrieve_memories(maturity="raw")`` must reach the raw queue.

Commit ``3f019999`` split the raw queue (a folder) from the curated store (a
single file).  ``search_by_tags`` reads ``self._curated.load_all()``, so after
the split **no tag-search query can return a raw memory however well tagged**
— the store that path reads no longer contains any.

``search_by_maturity`` was added in that same commit to source ``raw`` from
the raw queue.  It was correct, and tested, and had **zero production
callers**: the tool handler was never repointed at it, and applied ``maturity``
as a post-filter over a curated-only result set instead.

Three shipped surfaces promised the behaviour that no longer existed — the
memory-advisor persona's Pass 2 opens with
``retrieve_memories(scope=..., maturity="raw", limit=50)``, the plugin's own
class docstring says the tool "can still fetch raw via its maturity filter",
and the tool schema advertises "Filter by maturity state."

It stayed invisible because ``list_memory_tags`` answers *"Found 0 memories"*
from the curated indexer while holding the true raw count in the same dict.
Two curator sessions concluded their store was empty with twelve files on
disk; one hedged *"the memory write hasn't landed yet."*  Absent read as empty.
"""

from __future__ import annotations

import pytest

from shared.plugins.memory.storage import (
    MATURITY_RAW,
    Memory,
    MemoryStore,
)


def _store(tmp_path):
    return MemoryStore(str(tmp_path / "ws.jsonl"))


def _mem(**kwargs) -> Memory:
    """Same shape the storage-layout tests use, so the fixture cannot drift
    from the one the store is actually exercised with."""
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


def test_a_raw_memory_is_invisible_to_tag_search(tmp_path):
    """The premise.  If this ever stops being true the routing below is moot.

    Not a defect in ``search_by_tags`` — it reads the curated store by design.
    The defect was sending a maturity query through it.
    """
    store = _store(tmp_path)
    store.save(_mem(id="m_alpha", tags=["alpha"], scope="project"))

    by_tag = store.search_by_tags(["alpha"], limit=50, active_only=False)
    by_maturity = store.search_by_maturity({MATURITY_RAW}, limit=50)

    assert by_maturity, "the raw queue is empty; the fixture is wrong"
    assert not by_tag, (
        "search_by_tags returned a raw memory -- the split this test is "
        "about may have been undone, and the routing fix reconsidered"
    )


def test_the_handler_routes_maturity_to_the_maturity_store(tmp_path):
    """End to end through the tool handler, which is where the gap was."""
    from shared.plugins.memory.plugin import MemoryPlugin

    plugin = MemoryPlugin.__new__(MemoryPlugin)
    plugin._storage = _store(tmp_path)
    plugin._global_storage = None
    plugin._trace = lambda _m: None

    plugin._storage.save(_mem(
        id="m_beta", content="pending curation",
        tags=["beta"], scope="project",
    ))

    result = plugin._execute_retrieve({
        "maturity": MATURITY_RAW, "limit": 50, "scope": "project",
    })

    found = result.get("memories") or []
    assert found, (
        f"retrieve_memories(maturity='raw') returned nothing with a raw "
        f"memory on disk: {result!r}.  That is the defect -- the shipped "
        f"curator persona opens with exactly this call."
    )
    assert any("pending curation" in (m.get("content") or "") for m in found)


def test_a_tagless_maturity_query_still_works(tmp_path):
    """The stricter half.

    Pre-fix a tagless query ALSO died on the tag-overlap scoring, so both
    "no tags" and "well-tagged" failed for different reasons.  Routing by
    maturity removes tags from the question entirely.
    """
    from shared.plugins.memory.plugin import MemoryPlugin

    plugin = MemoryPlugin.__new__(MemoryPlugin)
    plugin._storage = _store(tmp_path)
    plugin._global_storage = None
    plugin._trace = lambda _m: None

    plugin._storage.save(_mem(
        id="m_untagged", content="untagged raw note",
        tags=[], scope="project",
    ))

    result = plugin._execute_retrieve({"maturity": MATURITY_RAW, "limit": 50})
    assert (result.get("memories") or []), (
        "a raw memory with no tags is unreachable -- the curator's queue is "
        "exactly where untagged notes accumulate"
    )
