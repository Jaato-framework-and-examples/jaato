"""``PLAN_FILE_FIELDS`` describes what ``parse_plan_document`` accepts (#1222).

``jaato-scaffold explain plugin todo`` renders the authored plan-file schema
from ``initial_plan.PLAN_FILE_FIELDS``.  The whole point of an ``explain``
surface is that it is computed from the tree and cannot rot, so this pins the
declared ``required`` set to the one the parser actually enforces: omit each
field the constant marks required and ``parse_plan_document`` must reject the
document; supply only the required fields and it must accept it, defaulting
``started`` to true (the ``started opt→true`` claim the surface renders).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..initial_plan import (
    PLAN_FILE_FIELDS,
    InitialPlanError,
    parse_plan_document,
)

_PATH = Path("plans/example.yaml")


def _minimal() -> dict:
    """A document carrying exactly the fields PLAN_FILE_FIELDS marks required."""
    return {"title": "T", "steps": [{"description": "do the thing"}]}


def test_the_minimal_required_document_is_accepted():
    plan = parse_plan_document(_minimal(), _PATH)
    assert plan.title == "T"
    assert [s.description for s in plan.steps] == ["do the thing"]


def test_started_defaults_to_true_when_absent():
    # The surface renders "started  optional  Defaults to true"; prove it.
    assert "started" not in _minimal()
    assert parse_plan_document(_minimal(), _PATH).started is True


def test_context_is_optional():
    assert "context" not in _minimal()
    parse_plan_document(_minimal(), _PATH)  # no raise


def test_every_required_top_level_field_is_enforced():
    """Dropping title or steps must raise — so the constant cannot claim a
    field is required that the parser lets through."""
    for field, required, _meaning in PLAN_FILE_FIELDS:
        if "." in field or not required:
            continue  # nested / optional handled below
        doc = _minimal()
        doc.pop(field)
        with pytest.raises(InitialPlanError):
            parse_plan_document(doc, _PATH)


def test_a_step_without_description_is_rejected():
    # steps[].description is the one required NESTED field.
    assert any(f == "steps[].description" and req
               for (f, req, _m) in PLAN_FILE_FIELDS)
    doc = {"title": "T", "steps": [{"step_id": "s1"}]}
    with pytest.raises(InitialPlanError):
        parse_plan_document(doc, _PATH)


def test_declared_required_set_is_exactly_what_the_parser_enforces():
    """The bidirectional guard: the fields the constant marks required are
    exactly {title, steps, steps[].description}.  A required field ADDED to
    the constant that the parser does not enforce would slip past the
    per-field probes above (dropping it would not raise); this catches that
    direction by naming the enforced set explicitly."""
    declared_required = {f for (f, req, _m) in PLAN_FILE_FIELDS if req}
    assert declared_required == {"title", "steps", "steps[].description"}
