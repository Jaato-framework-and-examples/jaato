"""Tests for memory enrichment on both prompts and tool results.

Covers:
1. Sub-token tag indexing (`workspace-baseline` reachable from the
   whole tag AND from `workspace` / `baseline` individually).
2. `min_overlap=1` default (single-tag matches surface).
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


class TestSubTokenIndexing:

    def test_whole_tag_still_indexed(self):
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        matches = idx.find_matches(["workspace-baseline"])
        assert len(matches) == 1

    def test_compound_tag_matched_from_single_sub_token(self):
        """The whole point: `workspace` alone should surface a memory
        tagged `workspace-baseline`."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["workspace-baseline"]))
        matches = idx.find_matches(["workspace"])
        assert len(matches) == 1

    def test_colon_tag_split_into_parts(self):
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["agent:main"]))
        assert len(idx.find_matches(["agent"])) == 1
        assert len(idx.find_matches(["main"])) == 1
        assert len(idx.find_matches(["agent:main"])) == 1

    def test_underscore_tag_split(self):
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["list_memory_tags"]))
        assert len(idx.find_matches(["memory"])) == 1
        assert len(idx.find_matches(["tags"])) == 1

    def test_short_sub_tokens_excluded_from_index(self):
        """Parts shorter than 3 chars should not clutter the index — so
        `v2` in `plan-v2` is not an index key by itself, but `plan` is."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["plan-v2"]))
        # Whole tag still reachable
        assert len(idx.find_matches(["plan-v2"])) == 1
        # `plan` works as sub-token
        assert len(idx.find_matches(["plan"])) == 1
        # `v2` was filtered out of the index, so no match
        assert len(idx.find_matches(["v2"])) == 0

    def test_single_tag_surfaces_at_min_overlap_one(self):
        """Default min_overlap is 1 — single-tag matches must surface."""
        idx = MemoryIndexer()
        idx.index_memory(_make_memory(["gpg-troubleshooting"]))
        matches = idx.find_matches(["gpg-troubleshooting"])
        assert len(matches) == 1


class TestKeywordExtraction:

    def test_compound_words_extracted_whole_and_split(self):
        idx = MemoryIndexer()
        keywords = idx.extract_keywords("How is the workspace-baseline today?")
        assert "workspace-baseline" in keywords
        assert "workspace" in keywords
        assert "baseline" in keywords

    def test_three_char_words_kept(self):
        idx = MemoryIndexer()
        keywords = idx.extract_keywords("API design for GPG")
        assert "api" in keywords
        assert "gpg" in keywords
        assert "design" in keywords


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
