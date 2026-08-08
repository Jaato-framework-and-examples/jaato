"""Tests for memory enrichment on both prompts and tool results.

Covers:
1. Paragraph-coherence matching for compound tags: a tag's components
   must co-occur in the same paragraph for the memory to surface.
2. Whole-tag verbatim matching as the universal fallback.
3. Tool-result enrichment subscription parity with references plugin.
"""

import tempfile
from pathlib import Path

from shared.plugins.memory.indexer import MemoryIndexer
from shared.plugins.memory.models import Memory
from shared.plugins.memory.plugin import MemoryPlugin


def _make_memory(tags):
    return Memory(
        id=f"mem_{'_'.join(tags)[:20]}",
        content="irrelevant for index tests",
        description=f"Memory tagged with {tags}",
        tags=tags,
        timestamp="2026-04-14T10:00:00",
        maturity="raw",
        confidence=0.9,
        scope="project",
    )


class TestParagraphCoherenceMatching:

    def test_whole_tag_verbatim_in_text(self):
        """If the full compound tag appears verbatim, it matches."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        matches = idx.find_matches_in_text("Tell me about workspace-baseline.")
        assert len(matches) == 1

    def test_compound_tag_components_in_same_sentence(self):
        """All components co-occur in one sentence → match."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        text = "Show me the workspace baseline state."
        matches = idx.find_matches_in_text(text)
        assert len(matches) == 1

    def test_components_split_across_sentences_does_not_match(self):
        """Components must co-occur in the SAME sentence/segment.
        With sentence-scoped segmentation, even adjacent sentences
        about different concepts no longer pull in each other's tags."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        text = "What is the current workspace? Specifically, the baseline state."
        matches = idx.find_matches_in_text(text)
        assert len(matches) == 0

    def test_compound_tag_single_component_does_not_match(self):
        """Single-component mention is too loose — requires both."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        # Just mentions workspace, not baseline
        matches = idx.find_matches_in_text("Tell me about my workspace today.")
        assert len(matches) == 0

    def test_components_split_across_lines_does_not_match(self):
        """Components on different lines no longer co-occur in a segment."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        text = "Tell me about my workspace.\nSeparately, what is a baseline?"
        matches = idx.find_matches_in_text(text)
        assert len(matches) == 0

    def test_short_atomic_tag_matches_as_word(self):
        """Atomic tags like `api` match by mere presence."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["api"]))
        matches = idx.find_matches_in_text("Document the api endpoints")
        assert len(matches) == 1

    def test_long_compound_tag_requires_all_components(self):
        """All components are required (no majority leniency).
        With sentence-sized segments, requiring all components is the
        right strictness — it prevents large structured dumps from
        trivially matching long compound tags by chance."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["skill-mod-code-circuit"]))
        # Only 2 of 4 components → does NOT match
        matches = idx.find_matches_in_text("Implement the circuit breaker code pattern.")
        assert len(matches) == 0
        # All 4 components in one sentence → matches
        matches = idx.find_matches_in_text(
            "The skill mod implements the code for circuit breaker."
        )
        assert len(matches) == 1

    def test_colon_separator_treated_as_split(self):
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["agent:main"]))
        matches = idx.find_matches_in_text("The main agent is responsible.")
        assert len(matches) == 1

    def test_underscore_separator_treated_as_split(self):
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["list_memory_tags"]))
        text = "We need to list the memory tags somehow."
        matches = idx.find_matches_in_text(text)
        # 3 components: list, memory, tags → majority = 2
        # All three present → matches
        assert len(matches) == 1


class TestSegmentationAvoidsFalsePositives:
    """Regression tests for the issue where long tool-result dumps
    (JSON, CLI output, category lists) trivially satisfied compound-
    tag matching because the whole dump was treated as one paragraph."""

    def test_json_dump_does_not_trigger_false_positive(self):
        """A list_tools()-style category dump must not surface
        memory-system memories just because both `memory` and `system`
        appear as category labels in different lines."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["memory-system"]))
        json_dump = (
            "{\n"
            '  "categories": [\n'
            '    {"name": "system", "description": "Shell commands, environment, system operations"},\n'
            '    {"name": "memory", "description": "Persistent memory, notes, context storage across sessions"},\n'
            '    {"name": "filesystem", "description": "Read, write, search, and navigate files"}\n'
            '  ]\n'
            "}"
        )
        matches = idx.find_matches_in_text(json_dump)
        assert len(matches) == 0

    def test_cli_output_does_not_trigger_unrelated_tags(self):
        """A line-per-row listing must not surface tags whose
        components happen to appear on different lines."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["plan-lifecycle"]))
        cli_dump = (
            "drwxr-xr-x  2 user user 4096 Apr 14 21:00 plan/\n"
            "drwxr-xr-x  2 user user 4096 Apr 14 21:00 lifecycle/\n"
        )
        matches = idx.find_matches_in_text(cli_dump)
        assert len(matches) == 0

    def test_compact_json_dump_does_not_trigger_via_segment_cap(self):
        """A compact one-line JSON dump has no newlines or sentence
        terminators, so it would be one giant segment.  The segment-
        size cap should kick in and reject it for component coherence."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["memory-system"]))
        compact = (
            '{"categories":[{"name":"system","description":"Shell commands, '
            'environment, system operations"},{"name":"memory","description":'
            '"Persistent memory, notes, context storage across sessions"},'
            '{"name":"filesystem","description":"Read, write, search, '
            'and navigate files and directories"}]}'
        )
        matches = idx.find_matches_in_text(compact)
        assert len(matches) == 0

    def test_prose_paragraph_with_period_splits_into_sentences(self):
        """Two sentences in the same paragraph separated by a period
        are treated as separate segments — components in different
        sentences don't co-occur."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["api-design"]))
        text = "The API is great. Design matters too."  # api in s1, design in s2
        matches = idx.find_matches_in_text(text)
        assert len(matches) == 0
        # Same components in one sentence → match
        text = "The api design is great."
        matches = idx.find_matches_in_text(text)
        assert len(matches) == 1


class TestComponentExtraction:

    def test_skip_short_components(self):
        """Components shorter than 3 chars are dropped."""
        components = MemoryIndexer._tag_components("plan-v2")
        assert components == ["plan"]
        # `v2` filtered

    def test_atomic_short_tag_no_components(self):
        """A tag that's entirely too-short returns no components.
        It still matches via the whole-tag verbatim path."""
        components = MemoryIndexer._tag_components("gc")
        assert components == []

    def test_atomic_short_tag_still_matchable_via_whole_tag(self):
        """Atomic short tag matches by literal substring presence."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["gc"]))
        matches = idx.find_matches_in_text("Investigate gc behaviour.")
        assert len(matches) == 1


class TestFindMatchesLegacy:
    """The legacy keyword-based find_matches still works for callers
    (e.g. tests, programmatic use) that already have distinct keywords."""

    def test_exact_tag_match(self):
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        matches = idx.find_matches(["workspace-baseline"])
        assert len(matches) == 1

    def test_sub_token_does_not_match_via_legacy(self):
        """Legacy path is exact-tag only — sub-tokens don't surface
        compound tags any more (paragraph-coherence path covers that)."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        assert idx.find_matches(["workspace"]) == []


class TestToolResultEnrichment:

    def test_plugin_subscribes_to_tool_results(self):
        plugin = MemoryPlugin()
        assert plugin.subscribes_to_tool_result_enrichment() is True
        assert plugin.get_tool_result_enrichment_priority() == 80

    def test_tool_result_receives_hint_when_memory_matches(self, tmp_path):
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(tmp_path / "workspace_memories.jsonl"),
            "global_storage_path": str(tmp_path / "global_memories.jsonl"),
        })
        # Store a memory tagged around a specific topic + promote so
        # it enters the curated index that enrichment reads.
        store = plugin._execute_store({
            "content": "When debugging gpg-troubleshooting, re-warm the agent cache",
            "description": "GPG troubleshooting recipe",
            "tags": ["gpg-troubleshooting", "pass-store"],
            "confidence": 0.9,
        })
        plugin._execute_update({"id": store["memory_id"], "maturity": "validated"})
        # A tool returns output that mentions the topic
        tool_output = (
            "pass show jaato-knowledge-manager/github-token\n"
            "gpg: troubleshooting required"
        )
        result = plugin.enrich_tool_result("cli_based_tool", tool_output)
        assert "💡 **Available Memories**" in result.result
        assert result.metadata.get("memory_matches", 0) >= 1

    def test_tool_result_unchanged_when_no_match(self, tmp_path):
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(tmp_path / "workspace_memories.jsonl"),
            "global_storage_path": str(tmp_path / "global_memories.jsonl"),
        })
        tool_output = "random text with no tag overlap"
        result = plugin.enrich_tool_result("some_tool", tool_output)
        assert result.result == tool_output
        assert result.metadata.get("memory_matches") == 0

    def test_hint_includes_memory_ids_and_one_call_form(self, tmp_path):
        """The enrichment hint must show memory IDs and the one-shot
        retrieve_memories(ids=[...]) form so the agent doesn't fan out
        into one call per bullet."""
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(tmp_path / "workspace_memories.jsonl"),
            "global_storage_path": str(tmp_path / "global_memories.jsonl"),
        })
        store = plugin._execute_store({
            "content": "X",
            "description": "Workspace baseline lesson",
            "tags": ["workspace-baseline"],
            "confidence": 0.9,
        })
        plugin._execute_update({"id": store["memory_id"], "maturity": "validated"})
        result = plugin.enrich_prompt(
            "Tell me about the workspace baseline state."
        )
        # Must include the one-call form
        assert "retrieve_memories(ids=" in result.prompt
        # Must show memory IDs in the bullet list
        for mid in result.metadata["matched_ids"]:
            assert mid in result.prompt

    def test_retrieve_by_ids_fetches_specific_memories(self, tmp_path):
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(tmp_path / "workspace_memories.jsonl"),
            "global_storage_path": str(tmp_path / "global_memories.jsonl"),
        })
        ids = []
        for i in range(3):
            r = plugin._execute_store({
                "content": f"content {i}",
                "description": f"memory {i}",
                "tags": [f"tag-{i}", "common-tag"],
                "confidence": 0.9,
            })
            ids.append(r["memory_id"])
            plugin._execute_update({"id": r["memory_id"], "maturity": "validated"})
        # Fetch first and third only
        result = plugin._execute_retrieve({"ids": [ids[0], ids[2]]})
        assert result["status"] != "no_results"
        returned_ids = [m["id"] for m in result["memories"]]
        assert returned_ids == [ids[0], ids[2]]

    def test_retrieve_by_ids_unknown_returns_no_results(self, tmp_path):
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(tmp_path / "workspace_memories.jsonl"),
            "global_storage_path": str(tmp_path / "global_memories.jsonl"),
        })
        result = plugin._execute_retrieve({"ids": ["mem_does_not_exist"]})
        assert result["status"] == "no_results"

    def test_prompt_and_tool_result_use_same_core(self, tmp_path):
        """Both surfaces must produce the same hint format — verified by
        running each on a *fresh* plugin instance (reading the same
        storage) so the session-level dedup doesn't suppress the second
        call."""
        # Seed the storage once so both plugin instances see the same memory.
        seeder = MemoryPlugin()
        seeder.initialize({
            "storage_path": str(tmp_path / "workspace_memories.jsonl"),
            "global_storage_path": str(tmp_path / "global_memories.jsonl"),
        })
        store = seeder._execute_store({
            "content": "X",
            "description": "About workspace-baseline",
            "tags": ["workspace-baseline"],
            "confidence": 0.9,
        })
        seeder._execute_update({"id": store["memory_id"], "maturity": "validated"})

        def _reader():
            p = MemoryPlugin()
            p.initialize({
                "storage_path": str(tmp_path / "workspace_memories.jsonl"),
                "global_storage_path": str(tmp_path / "global_memories.jsonl"),
            })
            return p

        probe = "tell me about the workspace-baseline"
        prompt_result = _reader().enrich_prompt(probe)
        tool_result = _reader().enrich_tool_result("some_tool", probe)
        # Both must have surfaced the memory and appended the same hint block
        assert "💡 **Available Memories**" in prompt_result.prompt
        assert "💡 **Available Memories**" in tool_result.result
        assert prompt_result.metadata.get("memory_matches") == tool_result.metadata.get("memory_matches")

    def test_same_memory_not_resurfaced_across_tool_calls(self, tmp_path):
        """Once a memory's hint bullet has been injected into the session's
        history, subsequent enrichments on the same text must NOT re-inject
        the same block — otherwise every tool result inflates with
        duplicate '💡 Available Memories' sections."""
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(tmp_path / "workspace_memories.jsonl"),
            "global_storage_path": str(tmp_path / "global_memories.jsonl"),
        })
        store = plugin._execute_store({
            "content": "X",
            "description": "About workspace-baseline",
            "tags": ["workspace-baseline"],
            "confidence": 0.9,
        })
        plugin._execute_update({"id": store["memory_id"], "maturity": "validated"})

        probe = "tell me about the workspace-baseline"
        first = plugin.enrich_prompt(probe)
        assert "💡 **Available Memories**" in first.prompt
        assert first.metadata["memory_matches"] == 1

        # Second call on the same text must NOT re-inject the hint block.
        second = plugin.enrich_tool_result("some_tool", probe)
        assert second.result == probe
        assert second.metadata["memory_matches"] == 0
        assert second.metadata.get("suppressed_duplicates") == [store["memory_id"]]

        # A third enrichment on a DIFFERENT matching text still sees no
        # new memories to surface — the dedup is keyed on memory ID, not
        # on probe text.
        third = plugin.enrich_tool_result(
            "another_tool", "anything about the workspace-baseline here"
        )
        assert "💡 **Available Memories**" not in third.result

        # After on_history_cleared(), the memory can surface again.
        plugin.on_history_cleared()
        fourth = plugin.enrich_prompt(probe)
        assert "💡 **Available Memories**" in fourth.prompt
        assert fourth.metadata["memory_matches"] == 1
