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

    def test_compound_tag_components_in_same_paragraph(self):
        """All components co-occur in one paragraph → match."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        text = "What is the current workspace? Specifically, the baseline state."
        matches = idx.find_matches_in_text(text)
        assert len(matches) == 1

    def test_compound_tag_single_component_does_not_match(self):
        """Single-component mention is too loose — requires both."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        # Just mentions workspace, not baseline
        matches = idx.find_matches_in_text("Tell me about my workspace today.")
        assert len(matches) == 0

    def test_components_split_across_paragraphs_does_not_match(self):
        """Components must co-occur in the SAME paragraph."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        text = "Tell me about my workspace.\n\nSeparately, what is a baseline?"
        matches = idx.find_matches_in_text(text)
        assert len(matches) == 0

    def test_short_atomic_tag_matches_as_word(self):
        """Atomic tags like `api` match by mere presence."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["api"]))
        matches = idx.find_matches_in_text("Document the api endpoints")
        assert len(matches) == 1

    def test_long_compound_tag_uses_majority(self):
        """For ≥3 components, majority (ceil(n/2)) suffices."""
        idx = MemoryIndexer()
        # 4 components → majority = 2
        idx.index_memory(_make_memory(["skill-mod-code-circuit"]))
        # Mention 2 of the 4: code and circuit
        matches = idx.find_matches_in_text("Implement the circuit breaker code pattern.")
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
        # Store a memory tagged around a specific topic
        plugin._execute_store({
            "content": "When debugging gpg-troubleshooting, re-warm the agent cache",
            "description": "GPG troubleshooting recipe",
            "tags": ["gpg-troubleshooting", "pass-store"],
            "confidence": 0.9,
        })
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

    def test_prompt_and_tool_result_use_same_core(self, tmp_path):
        """Both surfaces must produce the same hint format when matching
        the same memory."""
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(tmp_path / "workspace_memories.jsonl"),
            "global_storage_path": str(tmp_path / "global_memories.jsonl"),
        })
        plugin._execute_store({
            "content": "X",
            "description": "About workspace-baseline",
            "tags": ["workspace-baseline"],
            "confidence": 0.9,
        })
        probe = "tell me about the workspace-baseline"
        prompt_result = plugin.enrich_prompt(probe)
        tool_result = plugin.enrich_tool_result("some_tool", probe)
        # Both must have surfaced the memory and appended the same hint block
        assert "💡 **Available Memories**" in prompt_result.prompt
        assert "💡 **Available Memories**" in tool_result.result
        assert prompt_result.metadata.get("memory_matches") == tool_result.metadata.get("memory_matches")
