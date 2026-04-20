"""Tests for the template plugin's tag-based indexer and contextual enrichment.

Mirrors the matching contracts already covered for ``memory`` and
``references``: tag coherence applied to both the user prompt and tool
results, with the same shared :mod:`shared.tag_coherence` engine driving
all three plugins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.plugins.template.indexer import TemplateIndexer
from shared.plugins.template.plugin import (
    TemplateIndexEntry,
    TemplatePlugin,
)


# ──────────────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────────────


def _entry(name: str, tags=None, description: str = "", variables=None):
    return TemplateIndexEntry(
        name=name,
        source_path=f"/tmp/{name}",
        syntax="mustache",
        variables=variables or [],
        origin="standalone",
        tags=tags or [],
        description=description,
    )


def _initialised_plugin(tmp_path: Path) -> TemplatePlugin:
    """Build a plugin pointed at *tmp_path* as its workspace."""
    plugin = TemplatePlugin()
    plugin.initialize({"base_path": str(tmp_path), "agent_name": "test"})
    return plugin


# ──────────────────────────────────────────────────────────────────────
# TemplateIndexer
# ──────────────────────────────────────────────────────────────────────


class TestTemplateIndexer:

    def test_empty_index_no_matches(self):
        idx = TemplateIndexer()
        assert idx.find_matches_in_text("anything goes here") == []

    def test_untagged_entry_does_not_surface(self):
        idx = TemplateIndexer()
        idx.index_entry(_entry("plain.tpl", tags=[]))
        assert idx.find_matches_in_text("plain template please") == []

    def test_single_component_tag_matches_word(self):
        idx = TemplateIndexer()
        idx.index_entry(_entry("rest.tpl", tags=["rest"]))
        assert [e.name for e in idx.find_matches_in_text(
            "Build a rest endpoint scaffold."
        )] == ["rest.tpl"]

    def test_single_component_tag_respects_word_boundaries(self):
        """`rest` must not match inside `arrest` or `restless`."""
        idx = TemplateIndexer()
        idx.index_entry(_entry("rest.tpl", tags=["rest"]))
        assert idx.find_matches_in_text("Police can arrest suspects.") == []
        assert idx.find_matches_in_text("Restless clients keep retrying.") == []

    def test_compound_tag_requires_components_in_one_segment(self):
        idx = TemplateIndexer()
        idx.index_entry(_entry("rest-controller.tpl", tags=["rest-controller"]))
        # All components in one sentence → match
        assert [e.name for e in idx.find_matches_in_text(
            "Generate a REST controller for orders."
        )] == ["rest-controller.tpl"]
        # Components split across sentences → no match
        assert idx.find_matches_in_text(
            "Build a REST endpoint. The controller is separate."
        ) == []

    def test_ranking_prefers_more_tag_hits(self):
        idx = TemplateIndexer()
        idx.index_entry(_entry("a.tpl", tags=["rest"]))
        idx.index_entry(_entry("b.tpl", tags=["rest", "controller"]))
        names = [
            e.name for e in idx.find_matches_in_text(
                "REST controller scaffold for new endpoint."
            )
        ]
        # b.tpl matches both 'rest' and 'controller'; a.tpl matches only 'rest'
        assert names[0] == "b.tpl"

    def test_triggering_tags_returns_only_matched_tags(self):
        idx = TemplateIndexer()
        e = _entry("multi.tpl", tags=["rest", "kafka", "mongo"])
        idx.index_entry(e)
        triggered = idx.triggering_tags(
            "Wire a rest endpoint to the kafka producer.", [e]
        )
        assert set(triggered) == {"rest", "kafka"}

    def test_remove_entry_deindexes_its_tags(self):
        idx = TemplateIndexer()
        idx.index_entry(_entry("alpha.tpl", tags=["alpha"]))
        idx.remove_entry("alpha.tpl")
        assert idx.find_matches_in_text("alpha topic") == []
        assert "alpha" not in idx.get_all_tags()

    def test_segmentation_size_cap_prevents_dump_false_positives(self):
        """A long unsegmented JSON dump must not trivially satisfy compound
        tag matching — the same false-positive guard memory and references
        rely on (250-char segment cap inside tag_coherence).  The dump
        must exceed SEGMENT_MAX_CHARS for the cap to trigger fallback to
        verbatim-only matching."""
        idx = TemplateIndexer()
        idx.index_entry(_entry("kafka-mongo.tpl", tags=["kafka-mongo"]))
        # Single-line dump well over the 250-char cap; words `kafka` and
        # `mongo` appear at opposite ends but the segment is too large
        # to trust for component coherence.
        compact = (
            '{"queues":[{"name":"orders","kind":"kafka","partitions":12,'
            '"retention":"7d","replicas":3},{"name":"events","kind":"sqs",'
            '"partitions":1,"retention":"4d","replicas":1}],"databases":'
            '[{"name":"profiles","kind":"mongo","shards":4,"replicas":3,'
            '"region":"us-east-1"},{"name":"audit","kind":"postgres",'
            '"shards":1,"replicas":2,"region":"eu-west-1"}],"caches":'
            '[{"name":"sessions","kind":"redis","ttl":900},{"name":"warm",'
            '"kind":"memcached","ttl":60}]}'
        )
        assert len(compact) > 250  # sanity: must exceed the cap
        assert idx.find_matches_in_text(compact) == []


# ──────────────────────────────────────────────────────────────────────
# TemplatePlugin.enrich_prompt — model input
# ──────────────────────────────────────────────────────────────────────


class TestEnrichPrompt:

    def test_no_tagged_templates_returns_prompt_untouched(self, tmp_path):
        plugin = _initialised_plugin(tmp_path)
        result = plugin.enrich_prompt("any prompt here")
        assert result.prompt == "any prompt here"
        assert result.metadata.get("template_matches") == 0

    def test_matching_template_appears_in_hint(self, tmp_path):
        plugin = _initialised_plugin(tmp_path)
        plugin._template_index["RestController.java.tpl"] = _entry(
            "RestController.java.tpl",
            tags=["rest", "controller"],
            description="Spring REST controller scaffold",
            variables=["entity", "package"],
        )
        plugin._indexer.build_index(list(plugin._template_index.values()))

        result = plugin.enrich_prompt(
            "Generate a rest controller for the Order entity."
        )
        assert "📦 **Available Templates**" in result.prompt
        assert "RestController.java.tpl" in result.prompt
        assert "Spring REST controller scaffold" in result.prompt
        assert "renderTemplateToFile" in result.prompt
        assert result.metadata["template_matches"] == 1
        assert result.metadata["matched_names"] == ["RestController.java.tpl"]
        assert "rest" in result.metadata["trigger_tags"]
        assert "controller" in result.metadata["trigger_tags"]

    def test_unrelated_prompt_does_not_surface_template(self, tmp_path):
        plugin = _initialised_plugin(tmp_path)
        plugin._template_index["RestController.java.tpl"] = _entry(
            "RestController.java.tpl",
            tags=["rest", "controller"],
            description="REST controller scaffold",
        )
        plugin._indexer.build_index(list(plugin._template_index.values()))

        result = plugin.enrich_prompt("How do I configure logging?")
        assert "📦 **Available Templates**" not in result.prompt
        assert result.metadata["template_matches"] == 0


# ──────────────────────────────────────────────────────────────────────
# TemplatePlugin.enrich_tool_result — model output
# ──────────────────────────────────────────────────────────────────────


class TestEnrichToolResultContextual:

    def test_tool_result_surfaces_matching_template(self, tmp_path):
        plugin = _initialised_plugin(tmp_path)
        plugin._template_index["Entity.java.tpl"] = _entry(
            "Entity.java.tpl",
            tags=["entity", "java"],
            description="JPA entity scaffold",
            variables=["name", "package"],
        )
        plugin._indexer.build_index(list(plugin._template_index.values()))

        # A tool result containing topic-relevant prose, no embedded templates
        tool_output = (
            "Found existing JPA entity classes in src/main/java/.../entity:\n"
            "  Order.java, Customer.java, Invoice.java"
        )
        result = plugin.enrich_tool_result("filesystem_query", tool_output)
        assert "📦 **Available Templates**" in result.result
        assert "Entity.java.tpl" in result.result
        assert result.metadata["template_matches"] == 1

    def test_tool_result_unchanged_when_no_match(self, tmp_path):
        plugin = _initialised_plugin(tmp_path)
        plugin._template_index["Entity.java.tpl"] = _entry(
            "Entity.java.tpl", tags=["entity", "java"]
        )
        plugin._indexer.build_index(list(plugin._template_index.values()))

        result = plugin.enrich_tool_result(
            "shell", "ls -la\ntotal 24\ndrwxr-xr-x  4 user  staff   128"
        )
        assert "📦 **Available Templates**" not in result.result
        assert result.metadata.get("template_matches") == 0


# ──────────────────────────────────────────────────────────────────────
# Persisted index loading — tags + dual-schema support
# ──────────────────────────────────────────────────────────────────────


class TestPersistedIndexTagLoading:

    def test_loads_tags_from_runtime_persist_schema(self, tmp_path):
        """``{"templates": {name: entry, ...}}`` schema (runtime persist)."""
        idx_path = tmp_path / ".jaato" / "templates" / "index.json"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps({
            "templates": {
                "x.tpl": {
                    "name": "x.tpl",
                    "source_path": "/tmp/x.tpl",
                    "syntax": "mustache",
                    "variables": [],
                    "origin": "standalone",
                    "tags": ["x-tag", "foo"],
                    "description": "X template",
                }
            }
        }))
        plugin = _initialised_plugin(tmp_path)
        entry = plugin._template_index["x.tpl"]
        assert entry.tags == ["x-tag", "foo"]
        assert entry.description == "X template"
        # Indexer was rebuilt from loaded entries
        assert "x-tag" in plugin._indexer.get_all_tags()

    def test_loads_tags_from_gen_references_schema(self, tmp_path):
        """``{"entries": [...]}`` schema (gen-references generated)."""
        idx_path = tmp_path / ".jaato" / "templates" / "index.json"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps({
            "schema": "template-index",
            "version": "1.0",
            "entries": [
                {
                    "name": "y.tpl",
                    "display_name": "Y template",
                    "source": "modules/y/y.tpl",
                    "syntax": "mustache",
                    "variables": [],
                    "tags": ["y-tag"],
                }
            ],
        }))
        plugin = _initialised_plugin(tmp_path)
        assert "y.tpl" in plugin._template_index
        assert plugin._template_index["y.tpl"].tags == ["y-tag"]
        # display_name fell through to description
        assert plugin._template_index["y.tpl"].description == "Y template"

    def test_legacy_entries_without_tags_load_unaffected(self, tmp_path):
        """Pre-tags entries must still load (no breakage for old indexes)."""
        idx_path = tmp_path / ".jaato" / "templates" / "index.json"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps({
            "templates": {
                "z.tpl": {
                    "name": "z.tpl",
                    "source_path": "/tmp/z.tpl",
                    "syntax": "mustache",
                }
            }
        }))
        plugin = _initialised_plugin(tmp_path)
        entry = plugin._template_index["z.tpl"]
        assert entry.tags == []
        assert entry.description == ""


# ──────────────────────────────────────────────────────────────────────
# Parity: prompt and tool-result surfaces share the same hint format
# ──────────────────────────────────────────────────────────────────────


class TestSurfaceParity:

    def test_prompt_and_tool_result_produce_same_hint_block(self, tmp_path):
        """Same input text → same hint, regardless of which surface called it.

        The two surfaces are exercised on fresh plugin instances so the
        session-level dedup (``_surfaced_template_names``) doesn't suppress
        the second call — parity is about format, not per-session
        frequency.
        """
        def _build():
            p = _initialised_plugin(tmp_path)
            p._template_index["circuit.tpl"] = _entry(
                "circuit.tpl",
                tags=["circuit-breaker"],
                description="Resilience4j circuit breaker",
            )
            p._indexer.build_index(list(p._template_index.values()))
            return p

        text = "Add a circuit breaker around the payment call."
        prompt_res = _build().enrich_prompt(text)
        tool_res = _build().enrich_tool_result("anything", text)

        # Both surfaces produced a hint, both reference the same template.
        assert "📦 **Available Templates**" in prompt_res.prompt
        assert "📦 **Available Templates**" in tool_res.result
        assert prompt_res.metadata["matched_names"] == \
            tool_res.metadata["matched_names"]


class TestContextualDedup:
    """Session-level dedup: once a template's contextual hint bullet has
    been injected into history, subsequent enrichments on matching text
    must NOT re-inject it — every tool call would otherwise re-append the
    same '📦 Available Templates' block."""

    def test_second_call_on_same_text_does_not_resurface(self, tmp_path):
        plugin = _initialised_plugin(tmp_path)
        plugin._template_index["circuit.tpl"] = _entry(
            "circuit.tpl",
            tags=["circuit-breaker"],
            description="Resilience4j circuit breaker",
        )
        plugin._indexer.build_index(list(plugin._template_index.values()))
        text = "Add a circuit breaker around the payment call."

        first = plugin.enrich_prompt(text)
        assert "📦 **Available Templates**" in first.prompt
        assert first.metadata["template_matches"] == 1

        second = plugin.enrich_tool_result("anything", text)
        assert "📦 **Available Templates**" not in second.result
        assert second.metadata["template_matches"] == 0
        assert second.metadata.get("suppressed_duplicates") == ["circuit.tpl"]

    def test_new_template_still_surfaces_alongside_known(self, tmp_path):
        """If a later enrichment would match multiple templates, only the
        NEW ones appear in the hint; already-surfaced ones are filtered
        out of the bullet list."""
        plugin = _initialised_plugin(tmp_path)
        plugin._template_index["rest.tpl"] = _entry(
            "rest.tpl", tags=["rest"], description="REST scaffold"
        )
        plugin._template_index["controller.tpl"] = _entry(
            "controller.tpl", tags=["controller"], description="Controller scaffold"
        )
        plugin._indexer.build_index(list(plugin._template_index.values()))

        first = plugin.enrich_prompt("Build a rest endpoint.")
        assert first.metadata["matched_names"] == ["rest.tpl"]

        # Second prompt matches both tags — but rest.tpl is already
        # surfaced, so only controller.tpl should appear.
        second = plugin.enrich_prompt("Generate a controller for the rest api.")
        assert "controller.tpl" in second.metadata["matched_names"]
        assert "rest.tpl" not in second.metadata["matched_names"]
        assert "controller.tpl" in second.prompt
        # rest.tpl must not reappear in the bullet list
        assert second.prompt.count("rest.tpl") == 0

    def test_on_history_cleared_resets_dedup(self, tmp_path):
        plugin = _initialised_plugin(tmp_path)
        plugin._template_index["circuit.tpl"] = _entry(
            "circuit.tpl",
            tags=["circuit-breaker"],
            description="Resilience4j circuit breaker",
        )
        plugin._indexer.build_index(list(plugin._template_index.values()))
        text = "Add a circuit breaker around the payment call."

        plugin.enrich_prompt(text)
        plugin.on_history_cleared()
        after_reset = plugin.enrich_prompt(text)
        assert "📦 **Available Templates**" in after_reset.prompt
        assert after_reset.metadata["template_matches"] == 1
