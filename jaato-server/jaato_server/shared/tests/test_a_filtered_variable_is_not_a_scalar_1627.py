"""A Jinja2 variable used inside an expression is required, not typed (#1627).

``renderTemplate`` checks the variables payload against
``TemplatePlugin._parse_jinja2_structure``.  Every name that was not an
``if`` test, a ``for`` source or a ``default`` argument fell through to
``kind=scalar``, and the check then refused any non-primitive value.  So
``{{ decisions | join(', ') }}`` refused the list ``join`` needs, and
``{{ user.name }}`` refused the dict it reads, although both rendered.

Only a name that IS the whole ``{{ }}`` expression is a scalar now.  A name
inside a larger expression (a filter argument, attribute or item access, a
call argument, a ``{% set %}`` value) is ``operand``: it must be supplied,
and its type is not asserted, because the template does not state one.
"""

from __future__ import annotations

from jaato_server.shared.plugins.template.plugin import TemplatePlugin
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/template/plugin.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find=('                    return "scalar"\n'
              '                return "operand"\n'),
        replace=('                    return "scalar"\n'
                 '                return "scalar"\n'),
        because="a filtered name is typed scalar and the list join needs is refused",
        test="TestAnOperandIsNotTyped::test_the_issues_join_payload_passes",
    ),
    Reversion(
        target=_PLUGIN,
        find='            elif kind == "operand" and existing != "scalar":\n',
        replace='            elif False:\n',
        because="an optional gate elsewhere makes a required operand optional",
        test="TestAnOperandIsRequired::test_operand_beats_an_optional_kind",
    ),
    Reversion(
        target=_PLUGIN,
        find='        if kind == "operand":\n            return (\n',
        replace='        if False:\n            return (\n',
        because="an absent operand is accepted and renders as nothing",
        test="TestAnOperandIsRequired::test_absent_operand_is_refused",
    ),
]


def _plugin() -> TemplatePlugin:
    return TemplatePlugin.__new__(TemplatePlugin)


def _kinds(template: str) -> dict:
    return {
        v["name"]: v["kind"]
        for v in _plugin()._parse_jinja2_structure(template)
    }


def _check(template: str, variables: dict):
    return _plugin()._validate_render_inputs_against_structure(
        template, variables,
    )


class TestAnOperandIsNotTyped:
    def test_the_issues_join_payload_passes(self):
        assert _check(
            "{{ client_decisions | join(', ') }}",
            {"client_decisions": ["a", "b"]},
        ) is None

    def test_attribute_access_takes_a_dict(self):
        assert _check("{{ user.name }}", {"user": {"name": "x"}}) is None

    def test_kinds(self):
        assert _kinds(
            "{{ a | join(', ') }}{{ b.c }}{{ d[0] }}{{ e | length }}"
        ) == {"a": "operand", "b": "operand", "d": "operand", "e": "operand"}


class TestScalarAndOptionalKindsAreUnchanged:
    def test_bare_reference_is_still_a_scalar(self):
        err = _check("{{ x }}", {"x": ["a"]})
        assert err is not None and "kind=scalar" in str(err)

    def test_bare_reference_wins_over_an_operand(self):
        assert _kinds("{{ x | upper }}{{ x }}") == {"x": "scalar"}

    def test_default_and_for_source(self):
        assert _kinds(
            "{{ x | default('') }}{% for i in ys | sort %}{% endfor %}"
        ) == {"x": "scalar_with_default", "ys": "iterable"}


class TestAnOperandIsRequired:
    def test_absent_operand_is_refused(self):
        err = _check("{{ xs | join(', ') }}", {})
        assert err is not None and "kind=operand" in str(err)

    def test_operand_beats_an_optional_kind(self):
        assert _kinds("{% if x %}y{% endif %}{{ x | length }}") == {
            "x": "operand",
        }

    def test_gated_operand_stays_optional(self):
        assert _kinds("{% if x %}{{ x | join(',') }}{% endif %}") == {
            "x": "conditional_test",
        }
