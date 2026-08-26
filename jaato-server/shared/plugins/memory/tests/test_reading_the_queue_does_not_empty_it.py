"""Reading a raw memory must not move it out of the raw queue.

``_execute_retrieve`` bumps a usage counter on every memory it returns and
writes it back with ``MemoryStore.update``.  ``update`` tested the incoming
maturity against ``ACTIVE_MATURITIES`` — which CONTAINS ``raw``, because it
answers *"is this usable?"* — so any update of a raw memory promoted it into
the curated store, still marked raw.

The curator's own Pass-2 discovery call therefore emptied the queue it was
reading, one memory per read.  And it lands somewhere consequential: the
enrichment index is built from the curated store and admits maturity ∈
{raw, validated}, so an unvetted memory became enrichment material **by being
looked at** — the exact gate the curator exists to hold.

Measured on a live cascade: raw 9 → 5, curated 8 → 33, and the curated store
holding 18 raw against 1 validated.

Unreachable before the maturity-query routing fix, because no query could
return a raw memory at all.  Opening that path is what made this live.
"""

from __future__ import annotations

from shared.plugins.memory.models import (
    ACTIVE_MATURITIES,
    MATURITY_ESCALATED,
    MATURITY_RAW,
    MATURITY_VALIDATED,
    PROMOTES_OUT_OF_RAW,
    Memory,
)
from shared.plugins.memory.storage import MemoryStore


def _mem(**kw) -> Memory:
    d = {
        "id": "m", "content": "c", "description": "d", "tags": ["t"],
        "timestamp": "2024-01-01T00:00:00", "maturity": MATURITY_RAW,
    }
    d.update(kw)
    return Memory(**d)


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "ws.jsonl"))


def test_the_two_sets_answer_different_questions():
    """The root cause, stated as a property.

    ``ACTIVE_MATURITIES`` must keep RAW (a raw memory is usable), and the
    promotion test must NOT — reusing one for the other is the defect.
    """
    assert MATURITY_RAW in ACTIVE_MATURITIES, (
        "a raw memory is still usable; narrowing this would break enrichment"
    )
    assert MATURITY_RAW not in PROMOTES_OUT_OF_RAW, (
        "raw counts as 'has left the queue' -- any update evicts it"
    )
    assert PROMOTES_OUT_OF_RAW == {MATURITY_VALIDATED, MATURITY_ESCALATED}


def test_a_usage_bump_leaves_a_raw_memory_in_the_queue(tmp_path):
    """The live failure, reproduced through the real store."""
    store = _store(tmp_path)
    store.save(_mem(id="m1"))

    fetched = store.search_by_maturity({MATURITY_RAW})
    assert len(fetched) == 1

    mem = fetched[0]
    mem.usage_count += 1                      # what _execute_retrieve does
    mem.last_accessed = "2024-01-02T00:00:00"
    store.update(mem)

    still_raw = store.search_by_maturity({MATURITY_RAW})
    assert [m.id for m in still_raw] == ["m1"], (
        "reading a raw memory emptied it out of the queue -- the curator's "
        "discovery call drains the queue it is reading"
    )
    assert store.load_curated() == [] or all(
        m.id != "m1" for m in store.load_curated()
    ), "an unvetted memory reached the curated store by being read"


def test_the_usage_bump_is_actually_persisted(tmp_path):
    """Staying in the queue must not mean the write was dropped."""
    store = _store(tmp_path)
    store.save(_mem(id="m2"))

    mem = store.search_by_maturity({MATURITY_RAW})[0]
    mem.usage_count += 1
    store.update(mem)

    reloaded = store.search_by_maturity({MATURITY_RAW})[0]
    assert reloaded.usage_count == 1, (
        "the in-place raw update did not persist; staying put must not mean "
        "being discarded"
    )


def test_validating_still_promotes(tmp_path):
    """The half that must not regress — the curator's actual decision."""
    store = _store(tmp_path)
    store.save(_mem(id="m3"))

    mem = store.search_by_maturity({MATURITY_RAW})[0]
    mem.maturity = MATURITY_VALIDATED
    store.update(mem)

    assert store.search_by_maturity({MATURITY_RAW}) == []
    assert [m.id for m in store.load_curated()] == ["m3"]


def test_escalating_still_promotes(tmp_path):
    store = _store(tmp_path)
    store.save(_mem(id="m4"))

    mem = store.search_by_maturity({MATURITY_RAW})[0]
    mem.maturity = MATURITY_ESCALATED
    store.update(mem)

    assert store.search_by_maturity({MATURITY_RAW}) == []
    assert [m.id for m in store.load_curated()] == ["m4"]


def test_the_documented_leave_it_raw_action_works(tmp_path):
    """The shipped advisor persona documents this exact call.

        Adjust metadata: update_memory(id, confidence=..., tags=...)
          — fix without changing maturity (LEAVE RAW FOR NEXT PASS)

    Before the fix, the one action documented as "leave it pending" was the
    one that removed it from the queue.
    """
    store = _store(tmp_path)
    store.save(_mem(id="m5", tags=["old"]))

    mem = store.search_by_maturity({MATURITY_RAW})[0]
    mem.tags = ["new", "tags"]
    store.update(mem)                          # maturity untouched

    remaining = store.search_by_maturity({MATURITY_RAW})
    assert [m.id for m in remaining] == ["m5"]
    assert remaining[0].tags == ["new", "tags"]
