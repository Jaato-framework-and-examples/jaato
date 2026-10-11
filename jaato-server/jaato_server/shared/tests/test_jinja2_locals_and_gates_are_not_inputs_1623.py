"""A Jinja2 template's locals and gated references are not required inputs (#1623).

Before rendering, ``renderTemplate`` checks the variables payload against
what ``TemplatePlugin._parse_jinja2_structure`` reads from the template.
That reading demanded three kinds of names the caller cannot or need not
supply:

1. Jinja2's built-in ``loop`` inside a ``{% for %}``;
2. names bound by ``{% set %}`` / ``{% with %}`` / macro arguments;
3. a reference inside ``{% if x %}...{% endif %}``: ``{{ x }}`` in the
   body walked to the root and "scalar wins" made ``x`` required, while
   the missing-scalar error message recommends exactly that idiom.

The parser now classifies only names ``jinja2.meta.find_undeclared_variables``
reports, and a reference in the body of an ``if`` / ``elif`` whose test
guarantees the name (the bare name, ``x is defined``, an ``and`` chain)
is ``conditional_test``.  An ``else`` branch, an ``or`` test, or an
ungated reference elsewhere still makes the name required.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jinja2")

from jaato_server.shared.plugins.template.plugin import TemplatePlugin
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/template/plugin.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find='            if name not in undeclared or name_node.ctx != "load":\n',
        replace='            if name_node.ctx != "load":\n',
        because="loop and {% set %} names are demanded from the caller again",
        test="TestLocalsAreNotInputs::test_loop_and_set",
    ),
    Reversion(
        target=_PLUGIN,
        find='                and any(current is n for n in parent.body)\n',
        replace='                and False\n',
        because="a reference inside {% if x %} makes x required, against the error's own advice",
        test="TestIfGatedReferenceIsOptional::test_bare_name_gate",
    ),
    Reversion(
        target=_PLUGIN,
        find='            return test.node.name == name and test.name == "defined"\n',
        replace='            return False\n',
        because="{% if x is defined %} no longer gates its body",
        test="TestIfGatedReferenceIsOptional::test_is_defined_and_elif",
    ),
]


def _plugin() -> TemplatePlugin:
    return TemplatePlugin.__new__(TemplatePlugin)


def _kinds(template: str) -> dict:
    return {
        v["name"]: v["kind"]
        for v in _plugin()._parse_jinja2_structure(template)
    }


class TestLocalsAreNotInputs:
    def test_loop_and_set(self):
        assert _kinds(
            "{% for r in rules %}{% set n = loop.index %}"
            "R{{ n }} {{ r.text }}{% endfor %}"
        ) == {"rules": "iterable"}

    def test_loop_in_a_filter_argument(self):
        assert _kinds(
            '{% for r in rules %}{{ "R%d" | format(loop.index) }}{% endfor %}'
        ) == {"rules": "iterable"}

    def test_with_block_set_and_macro_arguments(self):
        assert _kinds(
            "{% with a = b %}{{ a }}{% endwith %}"
            "{% set body %}x{% endset %}{{ body }}"
            "{% macro m(q) %}{{ q }}{% endmacro %}{{ m(z) }}"
        ) == {"b": "scalar", "z": "scalar"}


class TestIfGatedReferenceIsOptional:
    def test_bare_name_gate(self):
        assert _kinds(
            "{% if signature %}Signed: {{ signature }}{% endif %}"
        ) == {"signature": "conditional_test"}

    def test_attribute_and_and_chain(self):
        assert _kinds(
            "{% if s and t %}{{ s.name }} {{ t }}{% endif %}"
        ) == {"s": "conditional_test", "t": "conditional_test"}

    def test_is_defined_and_elif(self):
        assert _kinds(
            "{% if a %}x{% elif u is defined %}{{ u }}{% endif %}"
        ) == {"a": "conditional_test", "u": "conditional_test"}

    def test_ungated_reference_still_required(self):
        assert _kinds("{{ x }}{% if x %}{{ x }}{% endif %}") == {"x": "scalar"}

    def test_else_branch_is_not_gated(self):
        assert _kinds("{% if x %}a{% else %}{{ x }}{% endif %}") == {"x": "scalar"}

    def test_or_does_not_gate(self):
        assert _kinds("{% if x or y %}{{ x }}{% endif %}")["x"] == "scalar"


class TestTheValidatorAgrees:
    """The render-input check accepts what the issue's caller sent."""

    TEMPLATE = (
        "{% for r in rules %}R{{ loop.index }} {{ r.text }}\n{% endfor %}"
        "{% if signature %}Signed: {{ signature }}{% endif %}"
    )

    def test_rules_without_signature_pass(self):
        err = _plugin()._validate_render_inputs_against_structure(
            self.TEMPLATE, {"rules": [{"text": "be kind"}]},
        )
        assert err is None

    def test_a_required_scalar_is_still_refused(self):
        err = _plugin()._validate_render_inputs_against_structure(
            "Hello {{ name }}", {},
        )
        assert err is not None and "name" in str(err)
