"""``explain plugin todo`` describes the authored plan FILE, not only the knob (#1222).

``plugin_configs.todo.initial_plan_name`` names ``<config_root>/plans/<id>.yaml``,
an authored asset ``jaato-scaffold validate`` fully checks.  ``explain`` used to
describe only the pointer — the knob — leaving the file's layout reachable only
through the plugin README.  It now renders the file's required/optional fields
beneath the knob, sourced from ``initial_plan.PLAN_FILE_FIELDS`` so the surface
cannot drift from what ``parse_plan_document`` accepts.

The fields are their OWN block, not ``children`` of the knob: ``initial_plan_name``
is a string that names a file, and rendering the file's fields as children would
read as if the knob itself took ``title`` / ``steps``.
"""

from __future__ import annotations

import pytest

from shared.scaffold import explain


@pytest.fixture(scope="module")
def page():
    return explain.plugin("todo")


def test_the_plan_file_fields_reach_the_text(page):
    _data, text = page
    # The knob itself is still rendered as a config setting …
    assert "initial_plan_name" in text
    # … and now the FILE's shape is too.
    for token in ("title", "steps", "steps[].description", "started", "context"):
        assert token in text, token


def test_the_required_and_optional_tags_are_shown(page):
    _data, text = page
    assert "required" in text
    assert "optional" in text
    # started is documented as defaulting to true (the opt→true claim).
    assert "Defaults to true" in text


def test_the_block_names_the_authored_file_not_a_config_object(page):
    _data, text = page
    # It reads as a file the knob names, resolved against config_root, and
    # points at the README for the full field table.
    assert "<config_root>/plans/<id>.yaml" in text
    assert "README" in text


def test_the_json_view_carries_the_plan_file_schema(page):
    data, _text = page
    fields = {f["name"]: f for f in data["initial_plan_file"]}
    assert fields["title"]["required"] is True
    assert fields["steps[].description"]["required"] is True
    assert fields["started"]["required"] is False
    assert fields["context"]["required"] is False


def test_a_plugin_without_the_knob_carries_an_empty_schema():
    # The json key is present on every plugin (zero-branch), and empty for
    # any that does not declare initial_plan_name — a machine consumer reads
    # [] as "not applicable".
    data, _text = explain.plugin("cli")
    assert data["initial_plan_file"] == []


def test_the_source_of_truth_is_the_initial_plan_constant(page):
    # The rendered set is exactly PLAN_FILE_FIELDS, so it cannot drift from
    # the module the parser lives in.
    from shared.plugins.todo.initial_plan import PLAN_FILE_FIELDS

    data, _text = page
    rendered = [(f["name"], f["required"]) for f in data["initial_plan_file"]]
    declared = [(n, req) for (n, req, _m) in PLAN_FILE_FIELDS]
    assert rendered == declared
