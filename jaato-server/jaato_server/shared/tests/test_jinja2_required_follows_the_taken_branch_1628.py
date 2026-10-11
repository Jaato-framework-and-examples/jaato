"""A Jinja2 variable is required only where the payload's branches render it (#1628).

``_parse_jinja2_structure`` reads the template alone, so in::

    {% if variant == 'neutral' %}{{ intention }}{% else %}{{ how_it_arrives }}{% endif %}

both ``intention`` and ``how_it_arrives`` were ``scalar`` and the render
check demanded both whichever variant the payload chose: the only way to
pass was to supply values that never appear in the output.

``TemplatePlugin._jinja2_drop_untaken_branches`` renders once with the
payload, in a sandbox, and keeps an absent ``scalar`` required only when
that render OUTPUTS it.  A name a taken branch renders is still refused
(the #iter-49 strictness), and a render that raises keeps the static,
every-branch reading.
"""

from __future__ import annotations

from jaato_server.shared.plugins.template.plugin import TemplatePlugin
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/template/plugin.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find=(
            "        structured = self._jinja2_drop_untaken_branches(\n"
            "            syntax, template_content, structured, variables,\n"
            "        )\n"
        ),
        replace="",
        because="every branch's variables are demanded again, whichever branch renders",
        test="TestTheTakenBranchDecides::test_neutral_variant",
    ),
    Reversion(
        target=_PLUGIN,
        find=(
            "            env.from_string(template_content).render(**variables)\n"
            "        except Exception:\n"
            "            return None\n"
        ),
        replace=(
            "            env.from_string(template_content).render(**variables)\n"
            "        except Exception:\n"
            "            return rendered\n"
        ),
        because="a render that stopped early would excuse every name after the failure",
        test="TestAFailedRenderKeepsTheStaticReading::test_names_after_the_failure_stay_required",
    ),
    Reversion(
        target=_PLUGIN,
        find=(
            "            def __str__(self) -> str:\n"
            "                if self._undefined_obj is missing and self._undefined_name:\n"
            "                    rendered.add(self._undefined_name)\n"
            "                return \"\"\n"
        ),
        replace="",
        because="nothing records an output, so a name the taken branch renders is excused",
        test="TestTheTakenBranchDecides::test_a_field_of_the_taken_branch_is_still_required",
    ),
]

_TEMPLATE = (
    "{% if variant == 'neutral' %}{{ intention }} {{ scope }}"
    "{% else %}{{ how_it_arrives }}{% endif %}"
)


def _check(template: str, variables: dict):
    plugin = TemplatePlugin.__new__(TemplatePlugin)
    return plugin._validate_render_inputs_against_structure(template, variables)


def _refused(err) -> set:
    return {
        e.split("'")[1]
        for e in (err or {}).get("validation_errors", [])
    }


class TestTheTakenBranchDecides:
    def test_neutral_variant(self):
        assert _check(
            _TEMPLATE, {"variant": "neutral", "intention": "i", "scope": "s"},
        ) is None

    def test_other_variant(self):
        assert _check(
            _TEMPLATE, {"variant": "target", "how_it_arrives": "h"},
        ) is None

    def test_a_field_of_the_taken_branch_is_still_required(self):
        err = _check(_TEMPLATE, {"variant": "neutral", "intention": "i"})
        assert _refused(err) == {"scope"}

    def test_elif_and_nested_branches(self):
        template = (
            "{% if k == 'a' %}{{ a1 }}"
            "{% elif k == 'b' %}{% if deep %}{{ b_deep }}{% endif %}{{ b1 }}"
            "{% else %}{{ c1 }}{% endif %}{{ always }}"
        )
        assert _refused(_check(template, {"k": "b", "always": 1})) == {"b1"}
        assert _check(template, {"k": "b", "b1": 1, "always": 1}) is None

    def test_an_unconditional_reference_is_still_required(self):
        assert _refused(_check("Hello {{ name }}", {})) == {"name"}


class TestAFailedRenderKeepsTheStaticReading:
    def test_names_after_the_failure_stay_required(self):
        # ``n + 1`` raises on an absent ``n`` before ``x`` is output;
        # the partial render must not decide that ``x`` is optional.
        err = _check("{{ n + 1 }}{{ x }}", {})
        assert {"n", "x"} <= _refused(err)
