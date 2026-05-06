"""Tests for the public ``classify_template_evaluation_kind`` helper.

Server 0.6.58+ exposes a module-level classifier so external walkers
(kb-enablement-2.0, future tenants) can populate
``TemplateIndexEntry.template_evaluation_kind`` without mirroring
the internal parser's HELPER_KEYWORDS set locally — guaranteeing
walker-vs-framework agreement on the same content byte-for-byte.

Empirical motivation: 7:3's build_descriptor cascade (2026-05-06)
rendered ``new CustomerSystemApiRequest( , , , , )`` because their
walker emitted index.json entries without
``template_evaluation_kind``.  The framework's loader defaulted to
``"substitution"`` despite the materialized template carrying
``{{#if isString}}...{{/if}}`` / ``{{#if isEnum}}...{{/if}}`` /
... helpers.  The codegen agent's persona reads the field to decide
whether to compute the helper-branch booleans; "substitution" → all
branches inert → broken constructor call.

These tests pin the public helper's contract so walkers can rely
on it as a single source of truth.
"""

from ..plugin import (
    HELPER_KEYWORDS,
    classify_template_evaluation_kind,
)


class TestClassifierBasics:
    """Bedrock cases — empty, plain text, simple substitution."""

    def test_empty_string(self):
        """Empty content classifies as substitution (no helpers possible)."""
        assert classify_template_evaluation_kind("") == "substitution"

    def test_whitespace_only(self):
        """Whitespace-only content classifies as substitution."""
        assert classify_template_evaluation_kind("   \n\n  ") == "substitution"

    def test_plain_text_no_template_syntax(self):
        """Plain text with no Mustache markers classifies as substitution."""
        assert classify_template_evaluation_kind(
            "Just a regular file with no template syntax."
        ) == "substitution"

    def test_simple_substitution(self):
        """``{{var}}`` only — no helpers — is substitution."""
        assert classify_template_evaluation_kind(
            "Hello {{name}}, welcome to {{site}}."
        ) == "substitution"

    def test_mustache_section_no_helper(self):
        """``{{#section}}...{{/section}}`` is mustache but not a helper.

        Sections that share names with helpers (``{{#if}}`` would be
        a Mustache section literally named "if") are excluded by the
        helper detector — it requires a non-empty argument after the
        keyword.  A bare ``{{#if}}`` with no argument is a section
        named "if", not a helper invocation.
        """
        assert classify_template_evaluation_kind(
            "{{#users}}{{name}}{{/users}}"
        ) == "substitution"


class TestHelperDetection:
    """Each Handlebars helper keyword is recognised."""

    def test_if_helper(self):
        """``{{#if X}}...{{/if}}`` classifies as helpers."""
        assert classify_template_evaluation_kind(
            "{{#if isActive}}Active{{/if}}"
        ) == "helpers"

    def test_unless_helper(self):
        """``{{#unless X}}...{{/unless}}`` classifies as helpers."""
        assert classify_template_evaluation_kind(
            "{{#unless @last}}, {{/unless}}"
        ) == "helpers"

    def test_each_helper(self):
        """``{{#each XS}}...{{/each}}`` classifies as helpers.

        ``{{#each}}`` is iteration but the parser registers it as a
        helper (it's a Handlebars-only construct, not Mustache —
        Mustache uses ``{{#XS}}`` for iteration).  Walkers must
        classify ``each``-using templates as helpers so the codegen
        agent knows to provide the iteration shape.
        """
        assert classify_template_evaluation_kind(
            "{{#each items}}{{name}}{{/each}}"
        ) == "helpers"

    def test_with_helper(self):
        """``{{#with OBJ}}...{{/with}}`` classifies as helpers."""
        assert classify_template_evaluation_kind(
            "{{#with author}}{{firstName}} {{lastName}}{{/with}}"
        ) == "helpers"

    def test_lookup_helper(self):
        """``{{#lookup OBJ KEY}}...{{/lookup}}`` classifies as helpers."""
        assert classify_template_evaluation_kind(
            "{{#lookup map 'foo'}}{{value}}{{/lookup}}"
        ) == "helpers"


class TestRealisticWalkerInputs:
    """Templates resembling real kb-enablement-2.0 walker inputs."""

    def test_seven_three_systemapi_mapper_shape(self):
        """7:3's empirical failure shape — multi-helper per-field dispatch.

        Verbatim shape of the SystemApiMapper.java.tpl that broke v24
        with bare commas.  Multiple helpers in sequence per field for
        type-conditional rendering.  Must classify as helpers so the
        codegen agent computes isString/isEnum/isDate/isOther booleans
        and renders the correct branch per field.
        """
        content = """package {{basePackage}}.adapter.out.systemapi;

public class {{Entity}}SystemApiMapper {

    public static {{Entity}}SystemApiRequest toRequest({{Entity}} domain) {
        return new {{Entity}}SystemApiRequest(
            {{#each fields}}
            {{#if isString}}domain.get{{name}}().toUpperCase(){{/if}}{{#if isEnum}}domain.get{{name}}().name(){{/if}}{{#if isDate}}domain.get{{name}}().toString(){{/if}}{{#if isOther}}domain.get{{name}}(){{/if}}{{#unless @last}}, {{/unless}}
            {{/each}}
        );
    }
}
"""
        assert classify_template_evaluation_kind(content) == "helpers"

    def test_jinja2_template_classifies_as_substitution(self):
        """Jinja2 control flow (``{% if %}``) is NOT Mustache helpers.

        Walkers shouldn't classify a Jinja2 template as helpers just
        because it contains conditional logic — Jinja2's syntax is
        different and the agent's helper-resolution path doesn't
        apply.  Stay defensive: only Mustache+Handlebars-shaped
        helpers get the ``"helpers"`` classification.
        """
        content = """{% if user.is_authenticated %}
Hello {{ user.name }}
{% endif %}
"""
        assert classify_template_evaluation_kind(content) == "substitution"

    def test_inverted_helper(self):
        """``{{^if X}}...{{/if}}`` (inverted helper) also classifies as helpers.

        Edge case — ``^`` prefix is the Mustache inverted-section
        marker but Handlebars also accepts it for helper inversion.
        Parser treats both ``#`` and ``^`` prefixes as helper
        invocation candidates.
        """
        assert classify_template_evaluation_kind(
            "{{^if isHidden}}visible{{/if}}"
        ) == "helpers"

    def test_helper_with_dotted_argument(self):
        """``{{#if validation.maxLength}}`` — dotted arg path.

        Common kb-enablement shape: nested validation rules accessed
        by dotted path.  Must still classify as helpers.
        """
        assert classify_template_evaluation_kind(
            "{{#if validation.maxLength}}@Size(max={{validation.maxLength}}){{/if}}"
        ) == "helpers"


class TestHelperKeywordsSourceOfTruth:
    """Verify the public ``HELPER_KEYWORDS`` set IS the contract.

    Walkers may import this set directly for their own logic; pin
    its contents so accidental drift in the framework breaks tests
    rather than silently changing walker behavior.
    """

    def test_helper_keywords_pinned(self):
        """0.6.58 ships with these five keywords; new ones bump tests."""
        assert HELPER_KEYWORDS == frozenset({
            'if', 'unless', 'each', 'with', 'lookup',
        })

    def test_helper_keywords_is_frozen(self):
        """Frozen so consumers can use it in set operations safely."""
        assert isinstance(HELPER_KEYWORDS, frozenset)


class TestClassifierMatchesInternalParser:
    """The public helper's output must match the internal parser's
    classification on the same content.  This is the load-bearing
    guarantee — walkers using the helper get the SAME answer the
    framework's discovery path would produce.
    """

    def _internal_parser_kind(self, plugin, content: str) -> str:
        plugin._parse_mustache_structure(content)
        return getattr(plugin._tls, "last_evaluation_kind", "substitution")

    def test_agreement_on_helpers_template(self):
        from ..plugin import TemplatePlugin
        plugin = TemplatePlugin()
        content = "{{#if x}}A{{/if}}{{#unless @last}}, {{/unless}}"
        assert (
            classify_template_evaluation_kind(content)
            == self._internal_parser_kind(plugin, content)
            == "helpers"
        )

    def test_agreement_on_plain_substitution(self):
        from ..plugin import TemplatePlugin
        plugin = TemplatePlugin()
        content = "{{name}} works at {{company}}."
        assert (
            classify_template_evaluation_kind(content)
            == self._internal_parser_kind(plugin, content)
            == "substitution"
        )

    def test_agreement_on_section_named_after_helper(self):
        """Bare ``{{#if}}`` without argument is a section named "if",
        not a helper.  Both paths must agree it's substitution.
        """
        from ..plugin import TemplatePlugin
        plugin = TemplatePlugin()
        content = "{{#if}}A{{/if}}"  # Mustache section, no helper arg
        assert (
            classify_template_evaluation_kind(content)
            == self._internal_parser_kind(plugin, content)
            == "substitution"
        )
