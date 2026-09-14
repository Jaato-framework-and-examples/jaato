"""The string-only spawn contract (#883).

``spawn_payload_schema`` reads as the mirror of
``completion_payload_schema``, but ``agent_params`` never cross the IPC
wire as JSON — ``IPCClient.create_session`` flattens them into
``key=value`` argv tokens and the daemon partitions them back on the
first ``=``.  So the daemon validates strings, and a property declared
``integer`` / ``boolean`` / ``object`` / ``array`` is refused on every
spawn whatever the caller passes.  #883 ratified the wire's behaviour as
the contract rather than reopening the transport.

These tests pin the three things ratification bought:

1. one predicate for "string-shaped", shared by the static validator and
   both runtime boundaries, so they cannot drift;
2. the same verdict at both boundaries — the in-process
   ``spawn_subagent`` call used to accept a real ``int`` the wire could
   never deliver;
3. a refusal that names the PROFILE, because the bare ``jsonschema``
   message blames a value the caller could not have sent differently.
"""

import json

import pytest

from shared.spawn_schema_loader import (
    WIRE_SAFE_SPAWN_TYPES,
    spawn_params_for_validation,
    spawn_type_contract_note,
    unreachable_spawn_types,
    validate_spawn_params,
)


# ── The string view the wire produces ────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("already", "already"),
    (1, "1"),
    (2.5, "2.5"),
    (True, "True"),
    ([1, 2], "[1, 2]"),
    ({"a": 1}, "{'a': 1}"),
])
def test_every_non_string_renders_as_the_argv_token_would(value, expected):
    """``f"{key}={value}"`` is the whole transport; this reproduces it."""
    assert spawn_params_for_validation({"k": value}) == {"k": expected}


def test_none_survives_as_none_rather_than_the_string_none():
    """``null`` is the one type a caller can still name.

    The wire would render ``None`` as ``"None"``, but a caller passing
    ``None`` means "absent", not those four characters — and
    ``["string", "null"]`` is a schema an author may legitimately write.
    """
    assert spawn_params_for_validation({"k": None}) == {"k": None}


def test_the_callers_dict_is_never_mutated():
    original = {"n": 1}
    assert spawn_params_for_validation(original) == {"n": "1"}
    assert original == {"n": 1}, "params reaching the session must be untouched"


def test_empty_and_missing_params_are_the_same_empty_dict():
    assert spawn_params_for_validation(None) == {}
    assert spawn_params_for_validation({}) == {}


# ── The shared predicate ─────────────────────────────────────────────

def test_string_shaped_properties_have_no_offenders():
    schema = {"type": "object", "properties": {
        "loop": {"type": "string"},
        "subphase": {"type": "string", "pattern": "^[0-9.]+$"},
    }}
    assert unreachable_spawn_types(schema) == {}


def test_a_typed_property_is_reported_with_its_type():
    schema = {"properties": {"iteration": {"type": "integer"},
                             "flag": {"type": "boolean"}}}
    assert unreachable_spawn_types(schema) == {
        "iteration": ["integer"], "flag": ["boolean"]}


def test_a_union_reports_only_its_dead_branch():
    """``["string", "integer"]`` accepts what arrives; the int branch is
    dead code, and saying so is cheaper than letting an author believe
    the typed form works."""
    assert unreachable_spawn_types(
        {"properties": {"n": {"type": ["string", "integer"]}}}
    ) == {"n": ["integer"]}


def test_null_is_wire_safe_because_string_null_is_satisfiable():
    assert "null" in WIRE_SAFE_SPAWN_TYPES
    assert unreachable_spawn_types(
        {"properties": {"n": {"type": ["string", "null"]}}}) == {}


def test_an_untyped_or_malformed_property_is_not_an_offender():
    schema = {"properties": {"free": {"description": "no type"},
                             "broken": "not-a-dict"}}
    assert unreachable_spawn_types(schema) == {}


def test_a_non_schema_is_quiet_rather_than_raising():
    assert unreachable_spawn_types(None) == {}
    assert unreachable_spawn_types({}) == {}


# ── The note that points at the profile ──────────────────────────────

def test_a_string_shaped_schema_gets_no_note():
    assert spawn_type_contract_note({}) == ""


def test_the_note_names_the_properties_and_the_remedy():
    note = spawn_type_contract_note({"iteration": ["integer"]})
    assert "iteration (integer)" in note
    assert "pattern" in note
    assert "spawn_schema_type_unreachable" in note
    assert note.endswith(" "), "concatenated after a details fragment"


# ── The boundary check both spawn paths share ────────────────────────

_INT_SCHEMA = {"type": "object",
               "properties": {"iteration": {"type": "integer", "minimum": 1}},
               "required": ["iteration"]}

_STR_SCHEMA = {"type": "object",
               "properties": {"iteration": {"type": "string",
                                            "pattern": "^[0-9]+$"}},
               "required": ["iteration"]}


def test_no_schema_declared_means_no_gate():
    assert validate_spawn_params(None, {"anything": "goes"}) is None


def test_the_string_form_passes_the_string_schema():
    assert validate_spawn_params(_STR_SCHEMA, {"iteration": "1"}) is None


def test_a_python_int_passes_the_string_schema_too():
    """The in-process boundary: the model emitted ``1``, the wire would
    have delivered ``"1"``, and the schema is the same either way."""
    assert validate_spawn_params(_STR_SCHEMA, {"iteration": 1}) is None


def test_a_python_int_still_fails_an_integer_schema():
    """The ratified rule, enforced where it previously was not.

    This is the case #883 measured: the in-process path accepted it and
    the IPC path refused it, so one profile meant two things.
    """
    details = validate_spawn_params(_INT_SCHEMA, {"iteration": 1})
    assert details is not None
    assert "is not of type 'integer'" in details


def test_the_refusal_blames_the_profile_not_the_call():
    details = validate_spawn_params(_INT_SCHEMA, {"iteration": 1})
    assert "REFUSES EVERY SPAWN" in details
    assert "iteration (integer)" in details
    assert "spawn_schema_type_unreachable" in details


def test_a_real_failure_carries_no_contract_note():
    """A string-shaped schema that legitimately rejects a value must not
    be reported as an authoring bug."""
    details = validate_spawn_params(_STR_SCHEMA, {"iteration": "not-a-number"})
    assert details is not None
    assert "REFUSES EVERY SPAWN" not in details


def test_missing_required_fields_are_reported_together():
    """A supervisor fixes them all in one retry rather than hammering the
    spawn loop one field at a time."""
    schema = {"type": "object",
              "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                             "c": {"type": "string"}},
              "required": ["a", "b", "c"]}
    details = validate_spawn_params(schema, {"a": "1"})
    assert "missing required fields: ['b', 'c']" in details


def test_a_path_form_schema_resolves_against_the_config_root(tmp_path):
    (tmp_path / "spawn_schemas").mkdir()
    (tmp_path / "spawn_schemas" / "fixer.json").write_text(
        json.dumps(_STR_SCHEMA), encoding="utf-8")
    assert validate_spawn_params(
        "spawn_schemas/fixer.json", {"iteration": 1},
        config_root=str(tmp_path)) is None
    assert validate_spawn_params(
        "spawn_schemas/fixer.json", {"iteration": "x"},
        config_root=str(tmp_path)) is not None


def test_an_unresolvable_schema_never_blocks_a_spawn(tmp_path):
    """A broken profile asset must not become an outage: the loader logs
    it and the spawn proceeds."""
    assert validate_spawn_params(
        "spawn_schemas/nope.json", {"x": "1"},
        config_root=str(tmp_path)) is None


def test_a_schema_jsonschema_itself_rejects_never_blocks_a_spawn():
    """``jsonschema`` raising ``SchemaError`` is a profile bug, not a
    caller bug, and degrading is what the two call sites always did."""
    assert validate_spawn_params(
        {"type": "object", "properties": {"a": {"type": 17}}},
        {"a": "1"}) is None
