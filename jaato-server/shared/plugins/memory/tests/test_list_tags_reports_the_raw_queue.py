"""``list_memory_tags`` must not report an empty store while holding a
non-empty raw queue.

The counterpart to ``test_maturity_query_reaches_raw``.  That test fixed the
retrieval; this one fixes the *discovery* step that precedes it, and which is
what made the retrieval gap survive so long.

``memory_count`` comes from the indexer, which is built from the CURATED
store.  The raw count was computed on every call — and routed into
``_telemetry``, a key the model never sees.  So a curator asking what was in
the store was told, in the only sentence it could read, *"Found 0 memories"*,
while the same dict carried ``"jaato.memory.count_raw": 12``.  Two curator
sessions concluded there was nothing to curate with twelve raw memories on
disk.  Neither reasoned badly; both were answered badly.

The fix does not widen ``memory_count`` to the true total.  That number is the
count of the store ``tags`` indexes, and the two are read together — a
denominator that silently changed meaning would break that pairing to fix a
different problem.  The queue gets its own key instead.
"""

from __future__ import annotations

from shared.plugins.memory.models import MATURITY_RAW, MATURITY_VALIDATED
from shared.plugins.memory.plugin import MemoryPlugin
from shared.plugins.memory.storage import Memory, MemoryStore


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


def _curate(store: MemoryStore, memory: Memory) -> None:
    """Promote through the curator's own path.

    ``save`` always lands in the raw queue whatever the ``maturity`` field
    says, so a fixture that "saves a validated memory" is testing a state the
    store cannot be in.  Promotion is an ``update``.
    """
    store.save(memory)
    memory.maturity = MATURITY_VALIDATED
    store.update(memory)


def _plugin(tmp_path) -> MemoryPlugin:
    from shared.plugins.memory.indexer import MemoryIndexer

    plugin = MemoryPlugin.__new__(MemoryPlugin)
    plugin._storage = MemoryStore(str(tmp_path / "ws.jsonl"))
    plugin._global_storage = None
    plugin._indexer = MemoryIndexer()
    plugin._trace = lambda _m: None
    return plugin


def test_a_raw_only_store_does_not_report_itself_empty(tmp_path):
    """The exact false premise two curator sessions reasoned from."""
    plugin = _plugin(tmp_path)
    for i in range(12):
        plugin._storage.save(_mem(id=f"m_{i}", tags=[f"topic{i}"]))

    result = plugin._execute_list_tags({})

    assert result["pending_curation"] == 12
    assert "12 raw" in result["message"], (
        f"the queue is invisible in the only field the model reads: "
        f"{result['message']!r}"
    )
    # And it says how to reach them, because tag search cannot.
    assert "maturity='raw'" in result["message"]


def test_memory_count_still_means_the_curated_store(tmp_path):
    """The number paired with ``tags`` keeps its denominator.

    Widening it to the true total would have made this call self-consistent
    and every caller that reads ``memory_count`` alongside ``tags`` wrong.
    """
    plugin = _plugin(tmp_path)
    plugin._storage.save(_mem(id="m_raw", tags=["queued"]))
    _curate(plugin._storage, _mem(id="m_done", tags=["indexed"]))
    plugin._indexer.build_index(plugin._storage.load_curated())

    result = plugin._execute_list_tags({})

    assert result["memory_count"] == 1          # curated only
    assert result["pending_curation"] == 1      # the queue, named separately
    assert result["tags"] == ["indexed"]        # raw tags are not indexed
    assert "1 curated memories" in result["message"]


def test_an_empty_queue_says_nothing_about_curation(tmp_path):
    """No queue, no noise.  ``pending_curation`` is still reported as 0 so a
    consumer can distinguish "empty" from "this build doesn't report it"."""
    plugin = _plugin(tmp_path)
    _curate(plugin._storage, _mem(id="m_done", tags=["indexed"]))
    plugin._indexer.build_index(plugin._storage.load_curated())

    result = plugin._execute_list_tags({})

    assert result["pending_curation"] == 0
    assert "raw" not in result["message"]


def test_the_raw_count_is_still_in_telemetry(tmp_path):
    """The telemetry key was never wrong — it was just the only witness."""
    plugin = _plugin(tmp_path)
    plugin._storage.save(_mem(id="m_raw"))

    result = plugin._execute_list_tags({})

    assert result["_telemetry"]["jaato.memory.count_raw"] == 1
    assert result["pending_curation"] == 1
