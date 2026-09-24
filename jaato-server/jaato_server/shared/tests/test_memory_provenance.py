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

from jaato_server.shared.plugins.memory.models import Memory
from jaato_server.shared.plugins.memory.plugin import MemoryPlugin
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/memory/plugin.py"
_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"

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
        target=_PLUGIN,
        find=("                memory.maturity = new_maturity\n"
              "                self._stamp_curation(memory, new_maturity)"),
        replace="                memory.maturity = new_maturity",
        because=(
            "the promotion path not stamping curated_by -- every field of "
            "the gate correct and nothing ever writing the one it reads, "
            "so require_curation withholds the whole corpus forever"
        ),
        test="test_the_promotion_path_stamps_the_approval",
    ),
    Reversion(
        target=_PLUGIN,
        find=('                memory.maturity = parsed["maturity"]\n'
              '                self._stamp_curation(memory, parsed["maturity"])'),
        replace='                memory.maturity = parsed["maturity"]',
        because=(
            "the EDITOR is the second writer of maturity, and a human "
            "curator promoting a memory there produced a validated record "
            "with no curator -- which require_curation then withholds, the "
            "same defect one command over"
        ),
        test="test_the_editor_promotion_stamps_the_approval",
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
        # The WIRING, not the check: a test calling the helper directly
        # passes with the call site deleted, which is what the reversion
        # meta-guard caught this declaring.
        test="test_the_validator_actually_runs_the_curation_check",
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
    from jaato_server.shared.session_context import (
        isolated_current_session, set_current_session,
    )
    with isolated_current_session():
        set_current_session(session)
        yield


@contextmanager
def _env(**values):
    """Set env vars for the block, restoring exactly what was there.

    ``%memory edit`` spawns ``$EDITOR`` on a temp file, so the command
    is drivable end to end with a script that rewrites the YAML -- which
    is what makes the editor test exercise the real path rather than the
    assignment inside it.
    """
    import os as _os
    previous = {k: _os.environ.get(k) for k in values}
    _os.environ.update(values)
    try:
        yield
    finally:
        for key, was in previous.items():
            if was is None:
                _os.environ.pop(key, None)
            else:
                _os.environ[key] = was


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


def test_the_promotion_path_stamps_the_approval(tmp_path):
    """raw -> validated -> retrievable, driven through the plugin.

    The whole loop, because every part of it was individually correct
    and nothing joined them: ``curated_by`` was declared, ``is_curated``
    read it, the gate read THAT -- and no code in the tree ever wrote
    the field.  So ``require_curation: true`` withheld the entire corpus
    forever, while the model was told a curator promoting a memory
    would make it retrievable and ``validate``'s own remedy named that
    promotion.

    Asserted end to end rather than on ``_stamp_curation`` directly: a
    unit test of the stamp would have passed on the broken tree too,
    since the stamp is not what was missing -- its CALL was.
    """
    plugin = _plugin(tmp_path, require_curation=True)
    with _in_session(_Session()):
        stored = plugin._execute_store({
            "content": "c", "description": "d", "tags": ["build"]})
    memory_id = stored["memory_id"]

    withheld = plugin._execute_retrieve({"ids": [memory_id]})
    assert withheld["status"] == "no_results", (
        "a freshly stored memory is raw, so the gate must withhold it")

    with _in_session(_Session()):
        promoted = plugin._execute_update({"id": memory_id,
                                           "maturity": "validated"})
    assert promoted["status"] == "success"

    after = plugin._execute_retrieve({"ids": [memory_id]})
    assert after["status"] == "success", (
        "the promotion path is the documented way to open the gate; if it "
        "does not stamp, require_curation closes the learning loop entirely")
    assert [m["id"] for m in after["memories"]] == [memory_id]


def test_the_editor_promotion_stamps_the_approval(tmp_path):
    """The OTHER writer of ``maturity``: ``%memory edit <id>``.

    ``update_memory`` is the model's promotion path; this is the
    HUMAN's, and it is the one a curator actually reaches for.  It
    assigned ``memory.maturity`` straight from the edited YAML with no
    stamp, so a person promoting a memory in their editor produced a
    ``validated`` record carrying no ``curated_by`` -- which
    ``require_curation`` then withholds, telling them nothing.  The
    same defect the promotion path had, one command over, and the
    reason the AST guard below exists rather than a second hand-written
    case per writer.

    Driven through the real command with a scripted ``$EDITOR``: the
    stamp is not what was missing on either path, its CALL was, so a
    unit test of ``_stamp_curation`` would pass on the broken tree.
    """
    plugin = _plugin(tmp_path, require_curation=True)
    with _in_session(_Session()):
        stored = plugin._execute_store({
            "content": "c", "description": "d", "tags": ["build"]})
    memory_id = stored["memory_id"]

    assert plugin._execute_retrieve({"ids": [memory_id]})["status"] == (
        "no_results"), "a freshly stored memory is raw"

    editor = tmp_path / "promote.sh"
    editor.write_text(
        "#!/bin/sh\n"
        "sed -i 's/^maturity: raw$/maturity: validated/' \"$1\"\n"
    )
    editor.chmod(0o755)

    with _in_session(_Session()):
        with _env(EDITOR=str(editor)):
            answer = plugin._memory_edit(memory_id)
    assert "Updated memory" in answer, answer

    after = plugin._execute_retrieve({"ids": [memory_id]})
    assert after["status"] == "success", (
        "a human curator promoting in the editor is an approval; unstamped, "
        "require_curation withholds the memory they just approved")
    assert [m["id"] for m in after["memories"]] == [memory_id]


def test_every_maturity_writer_stamps():
    """Structural, because the failure is a writer nobody thought about.

    Two sites write ``memory.maturity`` in the plugin and both had to
    be found by reading -- the second only after the first was fixed and
    reviewed.  A third would be silent in exactly the same way: the
    record looks promoted, ``is_curated`` reads ``False``, and the gate
    withholds it with no error anywhere.

    So the contract is checked at the source rather than case by case:
    a statement block that assigns ``.maturity`` must also call
    ``_stamp_curation``.  The sibling precedent is
    ``test_budget_mid_turn_955.py`` -- a new path that records without
    observing fails the build whether or not its author remembered.
    """
    tree = ast.parse((Path(__file__).resolve().parents[4] / _PLUGIN).read_text())

    def _writes_maturity(stmt):
        return (isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == "maturity"
                        for t in stmt.targets))

    def _stamps(stmt):
        for node in ast.walk(stmt):
            func = getattr(node, "func", None)
            if (isinstance(node, ast.Call)
                    and isinstance(func, ast.Attribute)
                    and func.attr == "_stamp_curation"):
                return True
        return False

    unstamped = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            writes = [s for s in block if _writes_maturity(s)]
            if writes and not any(_stamps(s) for s in block):
                unstamped.extend(s.lineno for s in writes)

    assert not unstamped, (
        f"{_PLUGIN} assigns memory.maturity at line(s) "
        f"{sorted(unstamped)} without calling _stamp_curation in the same "
        f"block. Every writer of maturity is a curation decision: a "
        f"promotion that does not stamp curated_by is withheld by "
        f"require_curation, and a demotion that does not clear it keeps "
        f"reading as approved (Art. 15(4), #1123)."
    )


def test_a_withdrawn_approval_is_withdrawn(tmp_path):
    """Demoting out of a curated maturity CLEARS the stamp.

    ``is_curated`` is what the gate reads, and it is deliberately not
    derived from ``maturity`` -- so a memory demoted back to ``raw``
    while still carrying ``curated_by`` reads as approved and keeps being
    surfaced, the curator's withdrawal silently undone by the field that
    recorded their approval.

    Demotion to ``raw`` rather than to ``dismissed`` because a dismissed
    memory is unlinked from storage entirely (``MemoryStore.update``), so
    that path has nothing left to mislead anyone with; ``raw`` is the one
    that stays readable.
    """
    plugin = _plugin(tmp_path, require_curation=True)
    with _in_session(_Session()):
        memory_id = plugin._execute_store({
            "content": "c", "description": "d", "tags": ["build"]})["memory_id"]
        plugin._execute_update({"id": memory_id, "maturity": "validated"})
    assert plugin._execute_retrieve({"ids": [memory_id]})["status"] == "success"

    with _in_session(_Session()):
        plugin._execute_update({"id": memory_id, "maturity": "raw"})

    stored = plugin._storage.get_by_id(memory_id)
    assert stored is not None
    assert stored.curated_by is None
    assert stored.is_curated is False
    assert plugin._execute_retrieve({"ids": [memory_id]})["status"] == "no_results"


def test_the_approval_records_the_curator_not_the_author(tmp_path):
    """Two fields, two sessions, two answers.

    The point of ``curated_by`` being a second field is that it can name
    a different party from ``generated_by``; stamping the author's
    binding at promotion time would make it a slower copy of the first.
    """
    plugin = _plugin(tmp_path, require_curation=True)
    author = _Session({"kind": "ai", "model": "author-model"})
    curator = _Session({"kind": "ai", "model": "curator-model"})

    with _in_session(author):
        memory_id = plugin._execute_store({
            "content": "c", "description": "d", "tags": ["build"]})["memory_id"]
    with _in_session(curator):
        plugin._execute_update({"id": memory_id, "maturity": "validated"})

    stored = (plugin._storage.get_by_id(memory_id)
              or plugin._global_storage.get_by_id(memory_id))
    assert stored.generated_by["model"] == "author-model"
    assert stored.curated_by["model"] == "curator-model"
    assert stored.curated_by.get("at"), "an approval with no time is not a record"


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
    from jaato_server.shared.scaffold.validate import _check_memory_curation

    out = []
    _check_memory_curation(
        {"learner": _profile("learner", plugin_configs={
            "memory": {"require_curation": True}})}, out)
    assert [d.code for d in out] == ["require_curation_without_curator"]
    assert out[0].severity == "warn"
    assert "MITIGATED, not closed" in out[0].message


def test_the_validator_actually_runs_the_curation_check(tmp_path):
    """Through ``validate_workspace``, not the helper.

    Every other test here calls ``_check_memory_curation`` directly,
    which asks whether the CHECK is right and cannot see whether it is
    WIRED -- so deleting its one call site in ``validate_workspace``
    left the whole suite green.  The meta-guard said so, and this is the
    half that makes the reversion bite.
    """
    from jaato_server.shared.scaffold.validate import validate_workspace

    profiles = tmp_path / ".jaato" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "learner.yaml").write_text(
        "name: learner\n"
        "description: a learner\n"
        "provider: nebius\n"
        "model: m\n"
        "plugins: [memory]\n"
        "plugin_configs:\n"
        "  memory:\n"
        "    require_curation: true\n",
        encoding="utf-8")

    codes = [d.code for d in validate_workspace(str(tmp_path))]
    assert "require_curation_without_curator" in codes, (
        "the knob with no curator anywhere closes the learning loop rather "
        "than mitigating it, and the validator must be the thing that says so")


def test_a_curator_sibling_satisfies_it():
    from jaato_server.shared.scaffold.validate import _check_memory_curation

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
    from jaato_server.shared.scaffold.validate import _check_memory_curation

    out = []
    _check_memory_curation({"learner": _profile("learner")}, out)
    assert out == []
