"""Regression tests for #982 — ``retrieve_memories`` says how much it left.

Reported: an agent asked to say everything it knew called
``retrieve_memories`` with tags and no ``limit``, and answered with three
memories out of twenty-six.  ``count`` was ``len(memories)`` AFTER
truncation, so a complete answer and a 12%-complete answer were
byte-identical in shape, both stamped ``success``.

Two defects, one cause.  The obvious one is that the total was discarded.
The other is that each store was truncated to ``limit`` SEPARATELY before
the merge, so the surviving three were not even the best three overall —
which is also why an honest total could not be reconstructed downstream.
"""

import pytest

from shared.plugins.memory.models import (
    MATURITY_RAW,
    MATURITY_VALIDATED,
    SCOPE_PROJECT,
    SCOPE_UNIVERSAL,
    Memory,
)
from shared.plugins.memory.plugin import MemoryPlugin


@pytest.fixture
def plugin(tmp_path):
    p = MemoryPlugin()
    p.initialize({
        "storage_path": str(tmp_path / "ws"),
        "global_storage_path": str(tmp_path / "global"),
    })
    return p


def _curate(store, mem_id, tags, *, scope=SCOPE_PROJECT, stamp="2026-09-01",
            maturity=MATURITY_VALIDATED):
    store.curated.upsert(Memory(
        id=mem_id, content=f"content of {mem_id}",
        description=f"description of {mem_id}", tags=list(tags),
        timestamp=stamp, maturity=maturity, scope=scope,
    ))


def _seed(plugin, count, *, prefix="mem", tags=("shared",), store=None,
          scope=SCOPE_PROJECT):
    target = store or plugin._storage
    for i in range(count):
        _curate(target, f"{prefix}_{i:03d}", tags, scope=scope,
                stamp=f"2026-09-{i + 1:02d}")


# ── the reported failure ────────────────────────────────────────────


def test_a_truncated_result_says_how_many_matched(plugin):
    """The whole correctness fix: 3 of 26 no longer looks like 3 of 3."""
    _seed(plugin, 26)

    result = plugin._execute_retrieve({"tags": ["shared"]})

    assert result["status"] == "success"
    assert result["count"] == 3
    assert result["matched"] == 26
    assert result["truncated"] is True


def test_a_complete_result_is_distinguishable_from_a_truncated_one(plugin):
    """'3 is the maximum' and '3 is all there was' must not be the same
    result."""
    _seed(plugin, 3)

    result = plugin._execute_retrieve({"tags": ["shared"]})

    assert result["count"] == 3
    assert result["matched"] == 3
    assert result["truncated"] is False


def test_the_truncation_is_stated_in_prose_too(plugin):
    """A field is what a caller reads; prose is what a model reads.  It is
    also the anchor field this result is enriched through (#922)."""
    _seed(plugin, 26)

    message = plugin._execute_retrieve({"tags": ["shared"]})["message"]

    assert "3 of 26" in message
    assert "limit" in message


def test_a_complete_result_says_so_rather_than_staying_silent(plugin):
    _seed(plugin, 2)

    message = plugin._execute_retrieve({"tags": ["shared"]})["message"]

    assert "all 2" in message


def test_a_larger_limit_returns_the_rest(plugin):
    """The remedy the message names actually works."""
    _seed(plugin, 26)

    result = plugin._execute_retrieve({"tags": ["shared"], "limit": 50})

    assert result["count"] == 26
    assert result["matched"] == 26
    assert result["truncated"] is False


# ── the second defect: per-store truncation chose the wrong page ────


def test_matched_counts_across_both_stores(plugin):
    """Each store used to be cut to `limit` before the merge, so no total
    computed after it could be right."""
    _seed(plugin, 13, prefix="ws")
    _seed(plugin, 13, prefix="gl", store=plugin._global_storage,
          scope=SCOPE_UNIVERSAL)

    result = plugin._execute_retrieve({"tags": ["shared"]})

    assert result["matched"] == 26
    assert result["count"] == 3


def test_the_returned_page_is_ranked_across_both_stores(plugin):
    """The best match must be returned even when it sits behind several
    per-store-better candidates in the OTHER store.

    Under per-store truncation the workspace store returned its own top 3
    by overlap and the merged set was then ordered by timestamp alone, so a
    3/3 tag match in the global store could lose to newer 1/3 matches.
    """
    # Workspace: three weak but very recent matches.
    for i in range(3):
        _curate(plugin._storage, f"ws_{i}", ["alpha"],
                stamp=f"2026-12-{i + 1:02d}")
    # Global: the one memory that matches every tag, and is older.
    _curate(plugin._global_storage, "gl_best", ["alpha", "beta", "gamma"],
            scope=SCOPE_UNIVERSAL, stamp="2026-01-01")

    result = plugin._execute_retrieve(
        {"tags": ["alpha", "beta", "gamma"], "limit": 1})

    assert result["matched"] == 4
    assert result["memories"][0]["id"] == "gl_best"


def test_the_same_id_in_both_stores_is_counted_once(plugin):
    """`matched` is a count of memories, not of rows read."""
    _curate(plugin._storage, "mem_dup", ["shared"])
    _curate(plugin._global_storage, "mem_dup", ["shared"],
            scope=SCOPE_UNIVERSAL)

    result = plugin._execute_retrieve({"tags": ["shared"], "limit": 10})

    assert result["matched"] == 1
    assert result["count"] == 1


def test_a_scope_filter_narrows_matched_too(plugin):
    """`matched` means "what an unlimited call would have returned", and an
    unlimited call applies the scope filter."""
    _seed(plugin, 5, prefix="ws")
    _seed(plugin, 4, prefix="gl", store=plugin._global_storage,
          scope=SCOPE_UNIVERSAL)

    result = plugin._execute_retrieve(
        {"tags": ["shared"], "scope": SCOPE_UNIVERSAL, "limit": 100})

    assert result["matched"] == 4


# ── the other result shapes ─────────────────────────────────────────


def test_the_ids_path_reports_a_complete_result(plugin):
    """Nothing is truncated on the ids path, and the fields say so rather
    than being absent — a caller reads the same four keys either way."""
    _seed(plugin, 5)

    result = plugin._execute_retrieve(
        {"ids": ["mem_000", "mem_001", "mem_002", "mem_003"]})

    assert result["count"] == 4
    assert result["matched"] == 4
    assert result["truncated"] is False


def test_no_results_carries_the_same_fields(plugin):
    """So a caller can read `matched` on every non-error outcome instead of
    branching on status first."""
    for args in ({"tags": ["nothing"]}, {"ids": ["mem_absent"]}):
        result = plugin._execute_retrieve(args)
        assert result["status"] == "no_results"
        assert result["count"] == 0
        assert result["matched"] == 0
        assert result["truncated"] is False


def test_the_maturity_path_reports_its_total_too(plugin):
    """`maturity` queries reach the raw queue via search_by_maturity, and
    were truncated per store in exactly the same way."""
    for i in range(7):
        plugin._storage.save(Memory(
            id=f"raw_{i}", content="c", description="d", tags=["queued"],
            timestamp=f"2026-09-{i + 1:02d}", maturity=MATURITY_RAW,
        ))

    result = plugin._execute_retrieve({"maturity": MATURITY_RAW, "limit": 2})

    assert result["count"] == 2
    assert result["matched"] == 7
    assert result["truncated"] is True


def test_telemetry_carries_the_total(plugin):
    _seed(plugin, 26)

    telemetry = plugin._execute_retrieve({"tags": ["shared"]})["_telemetry"]

    assert telemetry["jaato.memory.count_retrieved"] == 3
    assert telemetry["jaato.memory.count_matched"] == 26
    assert telemetry["jaato.memory.truncated"] is True


# ── the storage primitive ───────────────────────────────────────────


def test_search_by_tags_unlimited_returns_everything(plugin):
    """`limit=None` is what makes an honest total possible, and it is not
    a second scan: the store was already fully loaded and scored."""
    _seed(plugin, 26)

    assert len(plugin._storage.search_by_tags(["shared"], limit=None)) == 26
    assert len(plugin._storage.search_by_tags(["shared"])) == 3


def test_search_by_maturity_unlimited_returns_everything(plugin):
    for i in range(7):
        plugin._storage.save(Memory(
            id=f"raw_{i}", content="c", description="d", tags=["q"],
            timestamp=f"2026-09-{i + 1:02d}", maturity=MATURITY_RAW,
        ))

    assert len(plugin._storage.search_by_maturity(
        {MATURITY_RAW}, limit=None)) == 7
    assert len(plugin._storage.search_by_maturity(
        {MATURITY_RAW}, limit=2)) == 2


# ── the model-facing contract ───────────────────────────────────────


def test_the_tool_description_tells_the_model_what_truncated_means(plugin):
    """The issue is explicit that this is NOT a docs gap about the default
    — the default was always documented.  What the model could not know was
    whether the default BIT on this call."""
    schema = next(s for s in plugin.get_tool_schemas()
                  if s.name == "retrieve_memories")

    assert "matched" in schema.description
    assert "truncated" in schema.description
    limit_doc = schema.parameters["properties"]["limit"]["description"]
    assert "default: 3" in limit_doc
    assert "truncated" in limit_doc
