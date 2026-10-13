"""A required array whose answer is "none" can be completed field by field (#1626).

A completion schema declared ``errors: {type: array, items: {type: string,
minLength: 1}}`` as required, and its correct value is almost always the
EMPTY array.  A model on the field-by-field shape (``prepare_completion``
then ``signal_completion()``) could not set it: its wire turned ``[]`` into
the string ``"[]"``, which was rejected with a bare type-mismatch, and the
only call ever accepted was ``prepare_completion("errors[0]", <placeholder>)``
-- which made the array non-empty with the placeholder the schema's own
description forbids.  Measured: 7 of 7 field-by-field sessions flailed on
the field, one completed with ``["no-error-placeholder"]``.

Three changes, each guarded here:

- ``"[]"`` on an array path is read as the empty array (the #1358
  schema-driven coercion rule, via ``tool_arg_coercion.coerce_value``);
  ``""`` is still refused, because it is not a serialisation of any array;
- a rejection on an array path names the accepted shape (``value=[]``);
- every pending array field that may be empty carries ``set_empty_with``.

Every test drives the real ``LifecycleTools`` executors.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from jaato_server.shared.lifecycle_tools import LifecycleTools
from jaato_server.shared.tests.reversion import Reversion

_LIFECYCLE = "jaato-server/jaato_server/shared/lifecycle_tools.py"

REVERSIONS = [
    Reversion(
        target=_LIFECYCLE,
        find="        coerced, value = coerce_value(value, relaxed_leaf)\n",
        replace="        coerced = False\n",
        test="test_the_string_brackets_set_the_empty_array",
        because="without the coercion the model's \"[]\" is a string, refused",
    ),
    Reversion(
        target=_LIFECYCLE,
        find="            return value, (f\"{reason}. {hint}\" if hint else reason)\n",
        replace="            return value, reason\n",
        test="test_a_rejection_on_an_array_path_names_the_empty_array",
        because="a bare type mismatch never told the model what is accepted",
    ),
    Reversion(
        target=_LIFECYCLE,
        find="        _annotate_empty_array_fields(pending)\n        return pending\n",
        replace="        return pending\n",
        test="test_a_pending_array_field_says_how_to_set_it_empty",
        because="the pending list must say which fields emptiness satisfies",
    ),
]


SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["result", "errors"],
    "properties": {
        "result": {"type": "string"},
        "errors": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "description": (
                "Why this stage could not produce its result. An EMPTY "
                "ARRAY means there was no error; never a placeholder."
            ),
        },
        "blockers": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
    },
}


class _Hooks:
    def on_agent_completed(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Session:
    """What LifecycleTools reads off a session, and nothing else."""

    def __init__(self, schema: Optional[Dict[str, Any]]) -> None:
        self._completion_payload_schema = schema
        self._agent_id = "main"
        self.workspace_path = None
        self._ui_hooks = _Hooks()

    def get_context_usage(self) -> Dict[str, Any]:
        return {}


def _tools() -> LifecycleTools:
    return LifecycleTools(_Session(SCHEMA))


def _set(lt: LifecycleTools, path: str, value: Any) -> Dict[str, Any]:
    return lt._execute_prepare_completion({"field_path": path, "value": value})


def _pending_paths(result: Dict[str, Any]) -> list:
    return [
        p["path"] for p in result["pending_required_fields_with_descriptions"]
    ]


def test_a_typed_empty_array_completes_the_field() -> None:
    lt = _tools()
    _set(lt, "result", "done")
    out = _set(lt, "errors", [])
    assert out["rejected"] == {}
    assert out["is_complete"] is True
    assert lt._accumulated_payload["errors"] == []


def test_the_string_brackets_set_the_empty_array() -> None:
    lt = _tools()
    _set(lt, "result", "done")
    out = _set(lt, "errors", "[]")
    assert out["rejected"] == {}, out["rejected"]
    assert out["accepted"] == {"errors": []}
    assert lt._accumulated_payload["errors"] == []
    assert out["is_complete"] is True


def test_a_json_string_array_of_strings_is_read_as_the_array() -> None:
    lt = _tools()
    out = _set(lt, "errors", '["upstream timed out"]')
    assert out["accepted"] == {"errors": ["upstream timed out"]}


def test_the_empty_string_is_not_read_as_none() -> None:
    lt = _tools()
    out = _set(lt, "errors", "")
    assert "errors" in out["rejected"]
    assert "errors" not in lt._accumulated_payload
    assert "errors" in _pending_paths(out)


def test_a_string_on_a_string_path_is_never_parsed() -> None:
    lt = _tools()
    out = _set(lt, "result", "[]")
    assert out["accepted"] == {"result": "[]"}


def test_a_rejection_on_an_array_path_names_the_empty_array() -> None:
    lt = _tools()
    reason = _set(lt, "errors", "")["rejected"]["errors"]
    assert "value=[]" in reason
    assert "ARRAY" in reason


def test_a_pending_array_field_says_how_to_set_it_empty() -> None:
    lt = _tools()
    out = lt._execute_query_completion({})
    errors = next(
        p for p in out["pending_required_fields_with_descriptions"]
        if p["path"] == "errors"
    )
    assert errors["set_empty_with"] == (
        "prepare_completion(field_path='errors', value=[])"
    )
    result = next(
        p for p in out["pending_required_fields_with_descriptions"]
        if p["path"] == "result"
    )
    assert "set_empty_with" not in result


def test_an_array_that_must_be_non_empty_gets_no_empty_hint() -> None:
    from jaato_server.shared.lifecycle_tools import _annotate_empty_array_fields
    pending = [{"path": "blockers", "type": "array", "minItems": 1}]
    _annotate_empty_array_fields(pending)
    assert "set_empty_with" not in pending[0]


def test_the_tool_description_names_the_empty_array() -> None:
    schema = next(
        s for s in _tools().get_tool_schemas() if s.name == "prepare_completion"
    )
    assert "value=[]" in schema.description
    assert "[]" in schema.parameters["properties"]["value"]["description"]
