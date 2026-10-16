"""A Jinja2 ``{% for %}`` source must be a list, and is told so (#1706).

``renderTemplateToFile``'s ``variables`` is ``additionalProperties: true``,
so nothing tells the provider or #1358's coercion that a variable is a list.
MiniMax-M3 sent ``rules`` as JSON text, YAML text, ``{"item": {...}}`` (a
list with its XML wrapper kept) and plain strings.  The shape check
classified ``rules`` as ``iterable`` and let every one through; Jinja2 then
iterated a string's characters and failed with ``'str object' has no
attribute 'statement'``, naming neither the variable nor the type.

Now, for a name a loop iterates directly:

- a string that parses as a JSON array is decoded (the #1358 rule, with the
  template as the schema);
- any other string, ``None``, a bool or a number is refused, naming the
  variable and the type;
- a mapping is refused when the loop body reads fields of each item
  (iterating a mapping yields its keys), and accepted otherwise.

A name reached through an expression (``rules.values()``) is not typed.
"""

from __future__ import annotations

import pytest

from jaato_server.shared.plugins.template.plugin import TemplatePlugin
from jaato_server.shared.tests.reversion import Reversion

pytest.importorskip("jinja2")

_PLUGIN = "jaato-server/jaato_server/shared/plugins/template/plugin.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find=(
            "                errors.extend(self._iterable_value_errors(\n"
            "                    name, variables[name], iter_sources.get(name),\n"
            "                ))\n"
        ),
        replace="                pass\n",
        because="a string for a for-loop source reaches Jinja2 and fails unnamed",
        test="TestANonListIsRefusedByName::test_the_issues_shapes_are_refused",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if isinstance(value, dict) and not reads_fields:\n",
        replace="        if isinstance(value, dict):\n",
        because="the XML-wrapped {'item': ...} dict passes and iterates its keys",
        test="TestANonListIsRefusedByName::test_the_item_wrapper_is_refused",
    ),
    Reversion(
        target=_PLUGIN,
        find="        variables = self._decode_iterable_json_text(template, variables)\n",
        replace="",
        because="JSON-array text is refused instead of decoded",
        test="TestJsonTextIsDecoded::test_the_render_path_decodes_it",
    ),
]

_TEMPLATE = "{% for r in rules %}{{ r.statement }};{% endfor %}"


def _plugin() -> TemplatePlugin:
    plugin = TemplatePlugin.__new__(TemplatePlugin)
    plugin._agent_name = None
    return plugin


def _check(template: str, variables: dict):
    return _plugin()._validate_render_inputs_against_structure(
        template, variables,
    )


class TestANonListIsRefusedByName:
    def test_the_issues_shapes_are_refused(self):
        for value in ['[{"statement": "a"}]', "- statement: a", "R1", ""]:
            err = _check(_TEMPLATE, {"rules": value})
            assert err is not None, value
            text = " ".join(err["validation_errors"])
            assert "variable 'rules' (kind=iterable) must be a list; got str" in text

    def test_scalars_are_refused(self):
        for value in [None, True, 3, 1.5]:
            err = _check(_TEMPLATE, {"rules": value})
            assert err is not None, value
            assert "must be a list" in " ".join(err["validation_errors"])

    def test_the_item_wrapper_is_refused(self):
        err = _check(_TEMPLATE, {"rules": {"item": {"statement": "a"}}})
        assert err is not None
        assert "got dict" in " ".join(err["validation_errors"])

    def test_a_list_passes(self):
        assert _check(_TEMPLATE, {"rules": [{"statement": "a"}]}) is None


class TestWhatTheTemplateDoesNotTypeIsUntouched:
    def test_a_mapping_whose_keys_are_iterated_passes(self):
        assert _check("{% for k in m %}{{ k }}{% endfor %}", {"m": {"a": 1}}) is None

    def test_a_name_reached_through_an_expression_is_not_typed(self):
        assert _check(
            "{% for k, v in m.items() %}{{ k }}{% endfor %}", {"m": {"a": 1}},
        ) is None

    def test_an_absent_iterable_is_still_optional(self):
        assert _check(_TEMPLATE, {}) is None


class TestJsonTextIsDecoded:
    def test_json_array_text_becomes_a_list(self):
        out = _plugin()._decode_iterable_json_text(
            _TEMPLATE, {"rules": '[{"statement": "a"}]', "x": "[1]"},
        )
        assert out == {"rules": [{"statement": "a"}], "x": "[1]"}

    def test_yaml_and_non_array_json_are_left_alone(self):
        payload = {"rules": "- statement: a"}
        assert _plugin()._decode_iterable_json_text(_TEMPLATE, payload) is payload
        payload = {"rules": '{"item": {}}'}
        assert _plugin()._decode_iterable_json_text(_TEMPLATE, payload) is payload

    def test_the_render_path_decodes_it(self, tmp_path):
        plugin = TemplatePlugin()
        plugin.initialize({"base_path": str(tmp_path)})
        out = tmp_path / "out.txt"
        result = plugin._execute_render_template_to_file({
            "template": _TEMPLATE,
            "variables": {"rules": '[{"statement": "a"}, {"statement": "b"}]'},
            "output_path": str(out),
        })
        assert result.get("success") is True, result
        assert out.read_text() == "a;b;"
