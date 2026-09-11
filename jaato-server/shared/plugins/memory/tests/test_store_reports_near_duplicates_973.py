"""Regression tests for #973 — ``store_memory`` says when it already knows.

The reported incident: 83 stores in 179 seconds, 3 distinct facts, 30/29/24
copies, every one answered ``{"status": "success"}``.  All three facts were
already in ``curated.jsonl``.  Nothing in the plugin noticed, and nothing
could have told the agent it was repeating itself.

The fix is **reporting, not rejection** (the issue's option 1).  So the tests
below assert in both directions: that a duplicate is named, AND that it is
still stored — a change that started silently dropping memories would be a
worse defect than the one it fixed.
"""

import pytest

from shared.plugins.memory.models import MATURITY_VALIDATED, Memory
from shared.plugins.memory.plugin import MemoryPlugin
from shared.plugins.memory.similarity import (
    DEFAULT_DUPLICATE_THRESHOLD,
    MIN_COMPARABLE_TOKENS,
    content_tokens,
    find_near_duplicate,
    similarity,
)


# The three facts, in the style #973 quotes.  Reconstructions: the issue
# truncates all three, so these stand in for the real strings.
FACT = "El usuario ha trabajado durante dos anos en un harness corporativo"
REPHRASED = (
    "El usuario ha compartido que durante dos anos trabajo en un harness "
    "corporativo"
)
UNRELATED = "El proyecto usa PostgreSQL con indices parciales para busquedas"


@pytest.fixture
def plugin(tmp_path):
    """A memory plugin with both tiers rooted under tmp_path."""
    p = MemoryPlugin()
    p.initialize({
        "storage_path": str(tmp_path / "ws"),
        "global_storage_path": str(tmp_path / "global"),
    })
    return p


def _curate(plugin, content, description, tags, mem_id="mem_curated_1"):
    """Put a validated memory in the workspace curated store."""
    plugin._storage.curated.upsert(Memory(
        id=mem_id, content=content, description=description, tags=tags,
        timestamp="2026-09-01T10:00:00", maturity=MATURITY_VALIDATED,
    ))


def _store(plugin, content, description, tags=("harness",)):
    return plugin._execute_store({
        "content": content, "description": description, "tags": list(tags),
    })


# ── the measure ─────────────────────────────────────────────────────


def test_rephrasing_scores_far_above_unrelated_same_language_text():
    """The threshold sits between two measured bands, not at a guess.

    Function words are counted, so same-language pairs share a floor.  The
    default is defensible only if that floor is well below it and the
    rephrasing band well above.
    """
    dup = similarity(content_tokens(FACT), content_tokens(REPHRASED))
    noise = similarity(content_tokens(FACT), content_tokens(UNRELATED))

    assert dup >= 0.85, f"rephrasing scored only {dup}"
    assert noise <= 0.40, f"unrelated same-language text scored {noise}"
    assert noise < DEFAULT_DUPLICATE_THRESHOLD < dup


def test_a_text_too_short_to_judge_gets_no_verdict():
    """The overlap coefficient reports 1.0 for any short text inside a long
    one.  Below the token floor that is noise, so the answer is "no opinion"
    rather than a duplicate claim."""
    tiny = content_tokens("usuario prefiere")
    assert len(tiny) < MIN_COMPARABLE_TOKENS
    assert similarity(tiny, content_tokens(FACT + " " + UNRELATED)) == 0.0


def test_the_best_match_is_reported_not_the_first():
    """With several near-duplicates on file, "you already know this, as
    mem_x" is only actionable if mem_x is the closest one.

    Both entries clear the threshold and the weaker one is listed first, so
    a first-match implementation would name it.  (Exact ties fall back to
    pool order, which is fine: a tie means both contain the candidate.)
    """
    weaker = "El usuario ha trabajado durante dos anos como consultor externo"
    pool = [
        Memory(id="mem_weaker", content=weaker, description="consultor",
               tags=[], timestamp="2026-09-01"),
        Memory(id="mem_closer", content=REPHRASED, description="harness",
               tags=[], timestamp="2026-09-01"),
    ]
    candidate = content_tokens(FACT)
    weaker_score = similarity(candidate, content_tokens("consultor", weaker))
    closer_score = similarity(candidate, content_tokens("harness", REPHRASED))
    assert weaker_score < closer_score, "fixture no longer separates the two"

    match = find_near_duplicate(
        candidate, [("curated", pool)], threshold=min(weaker_score, 0.5))

    assert match is not None
    assert match.memory_id == "mem_closer"


# ── the tool result ─────────────────────────────────────────────────


def test_a_rephrasing_of_a_curated_memory_is_named(plugin):
    """The incident's exact shape: the fact is already curated, the agent
    rephrases it."""
    _curate(plugin, FACT, "harness corporativo", ["harness"])

    result = _store(plugin, REPHRASED, "harness corporativo, otra vez")

    assert result["duplicate_of"] == "mem_curated_1"
    assert result["duplicate_source"] == "curated"
    assert result["duplicate_similarity"] >= DEFAULT_DUPLICATE_THRESHOLD


def test_the_duplicate_is_still_stored(plugin):
    """Reporting, never rejection (issue option 1, not option 2).

    A fix that silently discarded a real memory would be the worst
    available failure for this plugin.
    """
    _curate(plugin, FACT, "harness corporativo", ["harness"])

    result = _store(plugin, REPHRASED, "harness corporativo, otra vez")

    assert result["status"] == "success"
    assert plugin._storage.raw.get(result["memory_id"]) is not None


def test_the_message_carries_the_finding_as_prose(plugin):
    """`message` is store_memory's anchor field (#922), so the prose is what
    a model reads most reliably — and what enrichment writes back to."""
    _curate(plugin, FACT, "harness corporativo", ["harness"])

    message = _store(plugin, REPHRASED, "otra vez")["message"]

    assert "mem_curated_1" in message
    assert "near-duplicate" in message
    assert "Do not store it again" in message


def test_a_new_fact_keeps_the_original_result_shape(plugin):
    """The three keys are ABSENT, not None, when nothing resembled the new
    memory — so an existing caller sees exactly what it always saw."""
    _curate(plugin, FACT, "harness corporativo", ["harness"])

    result = _store(plugin, UNRELATED, "postgres indices", tags=["postgres"])

    assert result["status"] == "success"
    assert "duplicate_of" not in result
    assert "duplicate_similarity" not in result
    assert "duplicate_source" not in result
    assert result["message"] == "Stored memory: postgres indices"


def test_the_loop_becomes_self_limiting_with_nothing_curated(plugin):
    """The first session of every deployment has an empty curated store.

    The raw queue is deliberately not scanned, so this is caught by the
    session's own-writes pool instead — which is what makes the runaway
    loop (one turn, 83 writes) self-limiting rather than only the
    already-curated case.
    """
    first = _store(plugin, FACT, "harness corporativo")
    assert "duplicate_of" not in first

    second = _store(plugin, REPHRASED, "harness corporativo, otra vez")
    assert second["duplicate_of"] == first["memory_id"]
    assert second["duplicate_source"] == "session"


def test_a_duplicate_of_a_previous_sessions_raw_memory_is_not_caught(plugin):
    """The documented limit of the chosen scope, asserted so it stays a
    decision rather than becoming an accident.

    Raw is the curator's queue and is unindexed by design; scanning it per
    write would make every store O(queue) in file opens, growing exactly
    when curation has fallen behind.  Consolidating near-duplicates that
    are awaiting curation is the curator's job.
    """
    plugin._execute_store({
        "content": FACT, "description": "harness", "tags": ["harness"]})
    plugin._recent_stores.clear()          # as if a new session had started

    result = _store(plugin, REPHRASED, "harness otra vez")

    assert "duplicate_of" not in result


# ── the opt-in knob ─────────────────────────────────────────────────


def test_rejection_is_off_by_default(plugin):
    assert plugin._reject_duplicates is False


def test_reject_duplicates_refuses_and_stores_nothing(tmp_path):
    """The issue's option 2: available, explicit, and never the default."""
    p = MemoryPlugin()
    p.initialize({
        "storage_path": str(tmp_path / "ws"),
        "global_storage_path": str(tmp_path / "global"),
        "reject_duplicates": True,
    })
    _curate(p, FACT, "harness corporativo", ["harness"])

    result = _store(p, REPHRASED, "harness otra vez")

    assert result["status"] == "rejected"
    assert result["duplicate_of"] == "mem_curated_1"
    assert "mem_curated_1" in result["error"]
    assert p._storage.raw.count() == 0


def test_threshold_is_configurable_and_a_bad_value_falls_back(tmp_path):
    """A malformed knob must not fail a session's memory writes."""
    p = MemoryPlugin()
    p.initialize({
        "storage_path": str(tmp_path / "ws"),
        "global_storage_path": str(tmp_path / "g"),
        "duplicate_threshold": "not-a-number",
    })
    assert p._duplicate_threshold == DEFAULT_DUPLICATE_THRESHOLD

    p2 = MemoryPlugin()
    p2.initialize({
        "storage_path": str(tmp_path / "ws2"),
        "global_storage_path": str(tmp_path / "g2"),
        "duplicate_threshold": 0.5,
    })
    assert p2._duplicate_threshold == 0.5


def test_both_knobs_are_declared_in_the_config_schema(plugin):
    """`jaato-scaffold validate` enforces declared types and ranges (#925),
    so an undeclared knob is an unvalidatable one."""
    props = plugin.get_config_schema()["properties"]
    assert props["duplicate_threshold"]["type"] == "number"
    assert props["reject_duplicates"]["type"] == "boolean"
    assert props["reject_duplicates"]["default"] is False


def test_the_model_facing_description_explains_the_field(plugin):
    """The model reads the tool description; a field it is never told about
    is a field it will not act on."""
    schema = next(s for s in plugin.get_tool_schemas()
                  if s.name == "store_memory")
    assert "duplicate_of" in schema.description


def test_the_session_pool_is_capped(plugin):
    """The plugin deliberately survives reset_for_next_session, so nothing
    else bounds this list."""
    from shared.plugins.memory.plugin import RECENT_STORE_CACHE_SIZE

    for i in range(RECENT_STORE_CACHE_SIZE + 25):
        _store(plugin, f"contenido numero {i} sobre un asunto distinto cada vez",
               f"asunto {i}", tags=[f"tag{i}"])

    assert len(plugin._recent_stores) == RECENT_STORE_CACHE_SIZE
