"""Memory provenance and the curation gate -- EU AI Act Art. 15(4) (#1123).

Article 15(4) is about systems that "continue to learn after being placed
on the market": feedback loops must be addressed so that possibly biased
outputs do not feed back as inputs without mitigation.  jaato's learning
loop is the memory plugin -- the model writes memories during a session
and they are re-injected into later sessions' prompts.

Two things were missing.  A memory did not record WHO WROTE IT: no
binding, no model, nothing.  The ``generated_by`` stamp (#1109) existed
for media on the wire and had an obvious second application -- storage --
that was not made.  And curation was a PATTERN, not a knob: the plugin's
own docstring describes the raw -> curated lifecycle and nothing in a
profile could require it, so an uncurated memory was re-injected exactly
as a curated one.

Five properties, each attached to a way it could silently stop holding:

A. the stamp lands, and is the SAME shape model media carries;
B. **the plugin stamps, never the model** -- provenance a subject
   asserts about itself is not provenance;
C. it holds under sibling subagents: nothing is stashed on ``self``,
   which is shared;
D. the gate withholds an uncurated memory and SAYS SO -- a silently
   shorter list is a model reasoning from a subset it believes is
   everything;
E. default unchanged, and `validate` warns when the knob closes the loop
   instead of mitigating it.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path

import pytest

from shared.plugins.memory.models import Memory
from shared.plugins.memory.plugin import MemoryPlugin
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_PLUGIN = "jaato-server/shared/plugins/memory/plugin.py"
_VALIDATE = "jaato-server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="            generated_by=self._model_provenance(),\n        )",
        replace="        )",
        because=(
            "a memory that does not record which model wrote it cannot be "
            "audited when that model turns out to have been wrong -- which "
            "is the whole of Art. 15(4)'s feedback-loop concern"
        ),
        test="test_the_stamp_lands_on_a_stored_memory",
    ),
    Reversion(
        target=_PLUGIN,
        find='        kept = [m for m in memories if m.is_curated]\n        return kept, len(memories) - len(kept)',
        replace='        return memories, 0',
        because=(
            "require_curation must actually withhold: a knob that reads as "
            "on and re-injects everything is worse than no knob, because a "
            "dossier then prints a mitigation that is not happening"
        ),
        test="test_the_gate_withholds_an_uncurated_memory",
    ),
    Reversion(
        target=_VALIDATE,
        find="    _check_memory_curation(result.profiles, out)\n",
        replace="",
        because=(
            "the knob with no curator anywhere means 'never re-inject "
            "anything' -- a learning loop CLOSED rather than mitigated, "
            "which is not what Art. 15(4) asks for"
        ),
        test="test_validate_warns_when_the_knob_closes_the_loop",
    ),
]


_BINDING = {"kind": "ai", "provider": "anthropic",
            "model": "claude-sonnet-5", "session_id": "s1", "agent_id": "main"}


class _Session:
    """The one method the plugin asks the current session for."""

    def __init__(self, stamp=_BINDING):
        self._stamp = stamp

    def _model_provenance(self):
        return self._stamp


@contextmanager
def _in_session(session):
    """Run with ``session`` as the current one, restored on exit.

    ``isolated_current_session`` is the framework's own restorer and the
    ONE writer of the "no session" state (#974); ``set_current_session``
    returns nothing, so a test cannot reset the var by hand.
    """
    from shared.session_context import (
        isolated_current_session, set_current_session,
    )
    with isolated_current_session():
        set_current_session(session)
        yield


def _plugin(tmp_path, **config):
    plugin = MemoryPlugin()
    plugin.initialize({"storage_path": str(tmp_path / "mem"), **config})
    plugin.set_workspace_path(str(tmp_path))
    return plugin


def _memory(**kw):
    base = dict(id="mem_1", content="c", description="d", tags=["build"],
                timestamp="2026-09-18T00:00:00")
    base.update(kw)
    return Memory(**base)


# ------------------------------------------------------------- A. the stamp

def test_the_stamp_lands_on_a_stored_memory(tmp_path):
    plugin = _plugin(tmp_path)
    with _in_session(_Session()):
        result = plugin._execute_store({
            "content": "the build needs node 20",
            "description": "node version",
            "tags": ["build"],
        })

    assert result["status"] == "success"
    stored = plugin._storage.get_by_id(result["memory_id"])
    assert stored.generated_by == _BINDING


def test_it_is_the_same_shape_model_media_carries():
    """One definition of the stamp, so a memory and a spoken answer name
    their binding identically."""
    from jaato_sdk.events import ai_generated_by

    assert set(ai_generated_by("anthropic", "claude-sonnet-5", "s1", "main")) \
        == set(_BINDING)


def test_no_session_in_context_is_provenance_unknown(tmp_path):
    # Never invented: a memory whose author cannot be established must not
    # claim one.  `None` reads as unknown, and emphatically not as
    # human-authored.
    plugin = _plugin(tmp_path)
    assert plugin._model_provenance() is None


def test_a_session_that_cannot_answer_does_not_fail_the_store(tmp_path):
    class Mute:
        pass

    class Exploding:
        def _model_provenance(self):
            raise RuntimeError("boom")

    plugin = _plugin(tmp_path)
    for session in (Mute(), Exploding()):
        with _in_session(session):
            assert plugin._model_provenance() is None


def test_an_older_record_reads_as_unknown_not_as_human():
    assert _memory().generated_by is None
    assert _memory().curated_by is None
    assert _memory().is_curated is False


# ------------------------------------------- B. the plugin stamps, not the model

def test_the_tool_schema_offers_the_model_no_provenance_parameter(tmp_path):
    plugin = _plugin(tmp_path)
    schema = next(s for s in plugin.get_tool_schemas()
                  if s.name == "store_memory")
    properties = (schema.parameters or {}).get("properties", {})
    assert "generated_by" not in properties
    assert "curated_by" not in properties


def test_a_model_supplied_stamp_is_ignored(tmp_path):
    plugin = _plugin(tmp_path)
    with _in_session(_Session()):
        result = plugin._execute_store({
            "content": "c", "description": "d", "tags": ["build"],
            # What a model would have to pass to forge provenance.
            "generated_by": {"kind": "human", "model": "a person"},
        })
    stored = plugin._storage.get_by_id(result["memory_id"])
    assert stored.generated_by == _BINDING, (
        "provenance a subject asserts about itself is not provenance")


def test_who_wrote_it_and_who_approved_it_are_two_fields():
    # Collapsing them loses the one an auditor asks for.
    curated = _memory(generated_by=_BINDING,
                      curated_by={"agent": "curator", "at": "2026-09-19"})
    assert curated.generated_by == _BINDING
    assert curated.is_curated is True


# --------------------------------------------------- C. shared-instance safety

def test_nothing_is_stashed_on_self():
    """The instance is SHARED across sibling subagents.

    Source-level, because the property is an absence: a
    ``self._session = session`` added later is invisible to any
    behavioural test that does not happen to spawn two siblings, and the
    plugin's own ``set_session`` docstring records that this exact
    mistake shipped once (PR-196).
    """
    tree = ast.parse(Path(_PLUGIN).read_text())
    fn = next(f for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef)
              and f.name == "_model_provenance")
    stores = [n for n in ast.walk(fn)
              if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)]
    assert not stores, (
        f"_model_provenance writes to self: {[n.attr for n in stores]}. "
        "The instance is shared with sibling subagents, so a stashed value "
        "is whichever sibling wrote last.")


def test_it_reads_the_session_per_execution():
    plugin = MemoryPlugin()
    with _in_session(_Session({"kind": "ai", "model": "a"})):
        assert plugin._model_provenance()["model"] == "a"
    with _in_session(_Session({"kind": "ai", "model": "b"})):
        assert plugin._model_provenance()["model"] == "b"


# ------------------------------------------------------------ D. the gate

def test_the_gate_withholds_an_uncurated_memory(tmp_path):
    plugin = _plugin(tmp_path, require_curation=True)
    raw = _memory(id="raw_1")
    approved = _memory(id="ok_1", curated_by={"agent": "curator"})

    kept, withheld = plugin._apply_curation_gate([raw, approved])
    assert [m.id for m in kept] == ["ok_1"]
    assert withheld == 1


def test_the_result_says_how_many_were_withheld(tmp_path):
    plugin = _plugin(tmp_path, require_curation=True)
    plugin._storage.save(_memory(id="raw_1", tags=["build"]))

    result = plugin._execute_retrieve({"ids": ["raw_1"]})
    assert result["status"] == "no_results"
    assert result["withheld_uncurated"] == 1
    assert "WITHHELD" in result["message"], (
        "a silently shorter list is a model reasoning from a subset it "
        "believes is everything")
    assert "stored, not lost" in result["message"]


def test_a_curated_memory_comes_back(tmp_path):
    plugin = _plugin(tmp_path, require_curation=True)
    plugin._storage.save(_memory(id="ok_1", curated_by={"agent": "curator"}))
    result = plugin._execute_retrieve({"ids": ["ok_1"]})
    assert result["status"] == "success"
    assert [m["id"] for m in result["memories"]] == ["ok_1"]


def test_the_gate_binds_both_retrieval_paths():
    """ids AND tag search.

    A gate on one of two paths is a gate the model routes around by
    asking for ids -- and ids is the path the enrichment hints steer it
    to (`retrieve_memories(ids=[...])`).
    """
    tree = ast.parse(Path(_PLUGIN).read_text())
    fn = next(f for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef)
              and f.name == "_execute_retrieve")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_apply_curation_gate"]
    assert len(calls) == 1, (
        "exactly one gate, placed after BOTH paths have built `memories` -- "
        f"found {len(calls)}")


def test_storage_is_not_gated(tmp_path):
    """Writing continues, which leaves the curator something to curate."""
    plugin = _plugin(tmp_path, require_curation=True)
    with _in_session(_Session()):
        result = plugin._execute_store({
            "content": "c", "description": "d", "tags": ["build"]})
    assert result["status"] == "success"
    assert plugin._storage.get_by_id(result["memory_id"]) is not None


# ------------------------------------------------------- E. default + validate

def test_the_default_is_unchanged(tmp_path):
    plugin = _plugin(tmp_path)
    assert plugin._require_curation is False
    raw = _memory(id="raw_1")
    assert plugin._apply_curation_gate([raw]) == ([raw], 0)


def test_a_double_without_the_attribute_gets_the_pre_1123_behaviour():
    # The #881 rule: a `__new__` double must not raise, and the default is
    # the safe direction -- gate off, exactly as before the knob existed.
    plugin = MemoryPlugin.__new__(MemoryPlugin)
    raw = _memory(id="raw_1")
    assert plugin._apply_curation_gate([raw]) == ([raw], 0)


def test_the_knob_is_declared_with_its_reason(tmp_path):
    plugin = _plugin(tmp_path)
    knob = plugin.get_config_schema()["properties"]["require_curation"]
    assert knob["default"] is False
    assert "15(4)" in knob["description"]


def _profile(name, **kw):
    from types import SimpleNamespace
    return SimpleNamespace(name=name, default_agent=kw.pop("default_agent", None),
                           plugin_configs=kw.pop("plugin_configs", {}), **kw)


def test_validate_warns_when_the_knob_closes_the_loop():
    from shared.scaffold.validate import _check_memory_curation

    out = []
    _check_memory_curation(
        {"learner": _profile("learner", plugin_configs={
            "memory": {"require_curation": True}})}, out)
    assert [d.code for d in out] == ["require_curation_without_curator"]
    assert out[0].severity == "warn"
    assert "MITIGATED, not closed" in out[0].message


def test_a_curator_sibling_satisfies_it():
    from shared.scaffold.validate import _check_memory_curation

    out = []
    _check_memory_curation({
        "learner": _profile("learner", plugin_configs={
            "memory": {"require_curation": True}}),
        "cur": _profile("cur", default_agent="memory-curator"),
    }, out)
    assert out == [], (
        "the curator is a SEPARATE profile by design, so asking whether THIS "
        "one binds it would warn on every correct setup")


def test_the_knob_off_says_nothing():
    from shared.scaffold.validate import _check_memory_curation

    out = []
    _check_memory_curation({"learner": _profile("learner")}, out)
    assert out == []
