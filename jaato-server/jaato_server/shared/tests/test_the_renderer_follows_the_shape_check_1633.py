"""The Jinja2 renderer accepts what the shape check accepts (#1633).

After #1623, #1627 and #1628 the shape check treats an absent name as
optional where the template only tests it (``{% if x %}``), loops over it
(``{% for r in rows %}``) or compares it for equality.  ``_render_jinja2``
rendered with ``StrictUndefined``, whose ``__bool__``, ``__iter__`` and
``__eq__`` raise, so the same payload was refused at render with
``undefined_variable``.

The renderer now uses an undefined that is falsy, iterates as empty and
compares unequal, and is still strict when printed.  The check, in turn,
requires a name an ``{% if %}`` test computes with (``x > 3``,
``x | upper``), because no undefined can answer that.
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
        find="        env = SandboxedEnvironment(undefined=_gateable_undefined())\n",
        replace=(
            "        from jinja2 import StrictUndefined\n"
            "        env = SandboxedEnvironment(undefined=StrictUndefined)\n"
        ),
        because="a gated absent name passes the check and is refused at render",
        test="TestTheCheckAndTheRendererAgree::test_accepted_payloads_render",
    ),
    Reversion(
        target=_PLUGIN,
        find="        __str__ = Undefined._fail_with_undefined_error\n",
        replace="        pass\n",
        because="a printed absent name renders as nothing",
        test="TestPrintingStaysStrict::test_printed_absent_names_raise",
    ),
    Reversion(
        target=_PLUGIN,
        find=(
            "                if self._jinja2_absent_name_survives(\n"
            "                    name_node, current, parent_map,\n"
            "                ):\n"
            "                    return \"conditional_test\"\n"
            "                return \"operand\"\n"
        ),
        replace="                return \"conditional_test\"\n",
        because="an if test that computes with an absent name passes the check and fails at render",
        test="TestTheCheckAndTheRendererAgree::test_refused_payloads_are_refused_by_the_check",
    ),
]


def _plugin() -> TemplatePlugin:
    return TemplatePlugin.__new__(TemplatePlugin)


def _check(template: str, variables: dict):
    return _plugin()._validate_render_inputs_against_structure(
        template, variables,
    )


def _render(template: str, variables: dict):
    return _plugin()._render_jinja2(template, variables)


# (template, payload, expected render) -- each accepted by the check.
_ACCEPTED = [
    ("{%- if signature %}{{ signature }}{% endif %}", {}, ""),
    ("{% for r in rules %}{% if r.open %}O{% endif %}{% endfor %}",
     {"rules": [{}]}, ""),
    ("{% for s in signatures %}{{ s }}{% endfor %}", {}, ""),
    ("{% if signature is defined %}{{ signature }}{% endif %}", {}, ""),
    ("{% if variant == 'a' %}A{% else %}B{% endif %}", {}, "B"),
    ("{% if not x %}none{% endif %}", {}, "none"),
    ("{% if user.is_admin and flag %}x{% endif %}", {}, ""),
    ("{% if x | length > 0 %}x{% endif %}", {}, ""),
    ("{{ x | default('d') }}", {}, "d"),
]

# Absent names the check must refuse, because the render cannot evaluate them.
_REFUSED = [
    "{% if n > 3 %}big{% endif %}",
    "{% if x | upper == 'A' %}y{% endif %}",
    "{% if n + 1 %}y{% endif %}",
    "{{ signature }}",
]


class TestTheCheckAndTheRendererAgree:
    def test_accepted_payloads_render(self):
        for template, payload, expected in _ACCEPTED:
            assert _check(template, payload) is None, template
            out, err = _render(template, payload)
            assert err is None, (template, err)
            assert out == expected, template

    def test_refused_payloads_are_refused_by_the_check(self):
        for template in _REFUSED:
            assert _check(template, {}) is not None, template
            _, err = _render(template, {})
            assert err is not None and err["status"] == "undefined_variable", template

    def test_the_compared_name_is_an_operand(self):
        kinds = {
            v["name"]: v["kind"]
            for v in _plugin()._parse_jinja2_structure(
                "{% if n > 3 %}{% endif %}{% if v == 'a' %}{% endif %}"
            )
        }
        assert kinds == {"n": "operand", "v": "conditional_test"}


class TestPrintingStaysStrict:
    def test_printed_absent_names_raise(self):
        for template, payload in [
            ("{{ signature }}", {}),
            ("{{ r.open }}", {"r": {}}),
            ("{% for r in rules %}{{ r.open }}{% endfor %}", {"rules": [{}]}),
            ("{{ user.name }}", {}),
            ("{{ x | upper }}", {}),
        ]:
            _, err = _render(template, payload)
            assert err is not None and err["status"] == "undefined_variable", template

    def test_the_error_names_the_variable(self):
        _, err = _render("{{ signature }}", {})
        assert "signature" in err["error"]
