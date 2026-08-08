"""prepare_completion free-shape object (minProperties, no `properties`) via dot-notation.

Regression for the kb context-cascade blocker: a floor field declared as a
free-shape object — ``{type:object, minProperties:1}`` with NO
``properties``/``required`` — populated via dot-notation
(``prepare_completion(field_path='stack_config.language')``) was REJECTED by
``_resolve_path_schema`` (`language` not in the empty ``properties`` → None →
"does not match schema structure"), so the value never stored, the object stayed
absent, and the floor (``_walk_required``) flagged it forever → NudgeExhaust loop.
One-shot passed because jsonschema only checks the ASSEMBLED object against
``minProperties``.  Two fixes: ``_resolve_path_schema`` honors
``additionalProperties``/free-shape (the store), ``_walk_required`` honors
``minProperties`` (the floor).
"""
from shared.lifecycle_tools import LifecycleTools


def _lt():
    # The path/floor helpers take schema as args — no session state needed.
    return LifecycleTools.__new__(LifecycleTools)


FREE_SHAPE_SCHEMA = {
    "type": "object",
    "required": ["stack_config"],
    "properties": {
        # free-shape: only constraint is non-emptiness; arbitrary sub-keys.
        "stack_config": {"type": "object", "minProperties": 1},
    },
}


def test_resolve_path_schema_accepts_free_shape_subkey():
    # 'stack_config.language' must resolve (to a permissive {}), NOT None — so
    # the value is accepted + stored instead of rejected as "not in schema".
    assert _lt()._resolve_path_schema(
        ["stack_config", "language"], FREE_SHAPE_SCHEMA) == {}


def test_resolve_path_schema_rejects_when_additional_properties_false():
    closed = {
        "type": "object",
        "properties": {"x": {"type": "object", "additionalProperties": False}},
    }
    assert _lt()._resolve_path_schema(["x", "y"], closed) is None


def test_floor_met_when_free_shape_object_filled():
    # stack_config filled (1 key) -> minProperties:1 met -> NOT pending.
    pending = _lt()._compute_pending_required_fields(
        {"stack_config": {"language": "java"}}, FREE_SHAPE_SCHEMA)
    assert pending == []


def test_floor_unmet_when_free_shape_object_empty():
    # present but empty -> minProperties:1 unmet -> pending (one-shot would
    # also fail; the accumulator now matches).
    pending = _lt()._compute_pending_required_fields(
        {"stack_config": {}}, FREE_SHAPE_SCHEMA)
    assert pending != []


def test_floor_unmet_when_free_shape_object_absent():
    # not set at all -> required[] flags it (unchanged behaviour).
    pending = _lt()._compute_pending_required_fields({}, FREE_SHAPE_SCHEMA)
    assert pending != []
