"""A usage write-back must not be able to undo a curator's decision.

The write-back in ``_execute_retrieve`` used to be ``update(stale_object)`` —
a full ``Memory`` carrying the maturity it had AT RETRIEVAL TIME, pushed
through the routing logic after an arbitrary delay.  The memory tools have no
parallel opt-out, so a curator emitting one retrieve plus several
``update_memory`` calls in a single response runs them CONCURRENTLY, and a
decision landing between the read and the write-back was silently undone:

    stale-raw over a now-CURATED object   -> upsert(raw): validation REVERTED
    stale-raw over a now-NOWHERE object   -> "not anywhere yet, goes to raw":
                                             dismissal RESURRECTED

Measured live: 10 of 32 curator decisions contradicted by on-disk state, a
four-decisions-one-file case, and a re-decide livelock (32 decisions over 19
distinct ids) because every resurrected memory reappears in the next batch.

Both shapes reproduced deterministically below — no threads, just the
ordering.  The threads were only ever the scheduler for it.
"""

from __future__ import annotations

from shared.plugins.memory.models import (
    MATURITY_DISMISSED,
    MATURITY_RAW,
    MATURITY_VALIDATED,
    Memory,
)
from shared.plugins.memory.storage import MemoryStore


def _mem(id: str) -> Memory:
    return Memory(id=id, content="c", description="d", tags=["t"],
                  timestamp="2024-01-01T00:00:00", maturity=MATURITY_RAW)


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "ws.jsonl"))


def test_a_validation_survives_a_stale_writeback(tmp_path):
    """Shape (b): validated must not land as raw."""
    store = _store(tmp_path)
    store.save(_mem("X"))

    # retrieve reads X (raw) ...
    retrieved_id = store.search_by_maturity({MATURITY_RAW})[0].id

    # ... the decision lands ...
    decided = store.search_by_maturity({MATURITY_RAW})[0]
    decided.maturity = MATURITY_VALIDATED
    store.update(decided)

    # ... and the write-back fires, by ID, not by stale object.
    store.record_usage(retrieved_id)

    curated = {m.id: m for m in store.load_curated()}
    assert curated["X"].maturity == MATURITY_VALIDATED, (
        f"the usage write-back reverted a validation: curated maturity is "
        f"{curated['X'].maturity!r}.  10 of 32 live decisions were lost to "
        f"exactly this."
    )
    assert curated["X"].usage_count == 1, "the usage bump itself was lost"
    assert store.search_by_maturity({MATURITY_RAW}) == []


def test_a_dismissal_survives_a_stale_writeback(tmp_path):
    """Shape (a): dismissed must stay dismissed — absent is a no-op."""
    store = _store(tmp_path)
    store.save(_mem("Y"))

    retrieved_id = store.search_by_maturity({MATURITY_RAW})[0].id

    decided = store.search_by_maturity({MATURITY_RAW})[0]
    decided.maturity = MATURITY_DISMISSED
    store.update(decided)
    assert store.search_by_maturity({MATURITY_RAW}) == []

    store.record_usage(retrieved_id)

    assert store.search_by_maturity({MATURITY_RAW}) == [], (
        "a dismissed memory was RESURRECTED by the usage write-back -- this "
        "is the re-decide livelock: it reappears in the next batch and gets "
        "decided again, forever"
    )
    assert all(m.id != "Y" for m in store.load_curated())


def test_record_usage_bumps_in_place_without_relocating(tmp_path):
    """The narrow contract: usage moves, nothing else does."""
    store = _store(tmp_path)
    store.save(_mem("Z"))

    store.record_usage("Z")
    store.record_usage("Z")

    still_raw = store.search_by_maturity({MATURITY_RAW})
    assert [m.id for m in still_raw] == ["Z"], "usage recording relocated it"
    assert still_raw[0].usage_count == 2
    assert store.load_curated() == []


def test_the_store_is_lockable_and_reentrant(tmp_path):
    """The belt for the braces.

    ``update`` routes between two sub-stores and ``CuratedStore`` rewrites a
    shared file; with no lock, two genuine DECISIONS 1ms apart could still
    lose one (three validations, one survivor, observed live).  The lock
    lives inside the store so every caller inherits it — a guard at one call
    site is how #626's save race happened.
    """
    import threading

    store = _store(tmp_path)
    assert isinstance(store._lock, type(threading.RLock())), (
        "MemoryStore has no internal lock; concurrent decisions on the "
        "shared curated file are last-writer-wins again"
    )
    # Re-entrant: record_usage under an outer hold must not deadlock.
    store.save(_mem("W"))
    with store._lock:
        store.record_usage("W")


def test_concurrent_decisions_all_survive(tmp_path):
    """The original Finding B: three validations, three survivors.

    Live, three ``update_memory(..., validated)`` calls 3-4ms apart kept ONE:
    each read the same pre-image of curated.jsonl and rewrote the whole file.
    With the store lock they serialize.
    """
    import threading

    store = _store(tmp_path)
    for i in range(3):
        store.save(_mem(f"m{i}"))

    def _validate(mid: str) -> None:
        mem = store.get_by_id(mid)
        mem.maturity = MATURITY_VALIDATED
        store.update(mem)

    threads = [
        threading.Thread(target=_validate, args=(f"m{i}",)) for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    curated = {m.id: m.maturity for m in store.load_curated()}
    assert curated == {
        "m0": MATURITY_VALIDATED,
        "m1": MATURITY_VALIDATED,
        "m2": MATURITY_VALIDATED,
    }, (
        f"concurrent validations lost writes: {curated}.  Live this was "
        f"three decisions, one survivor."
    )
