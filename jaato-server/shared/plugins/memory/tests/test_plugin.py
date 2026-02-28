"""Tests for MemoryPlugin.

Covers core CRUD, prompt enrichment, tag validation, and the
knowledge-curation lifecycle fields (maturity, confidence, scope, evidence).
"""

import json
import tempfile
import unittest
from pathlib import Path

from ..models import (
    MATURITY_DISMISSED,
    MATURITY_ESCALATED,
    MATURITY_RAW,
    MATURITY_VALIDATED,
    SCOPE_PROJECT,
    SCOPE_UNIVERSAL,
    Memory,
)
from ..plugin import MemoryPlugin
from ..storage import MemoryStorage


class TestMemoryPlugin(unittest.TestCase):
    """Test cases for MemoryPlugin."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = str(Path(self.temp_dir) / "test_memories.jsonl")

        # Initialize plugin
        self.plugin = MemoryPlugin()
        self.plugin.initialize({
            "storage_path": self.storage_path
        })

    def tearDown(self):
        """Clean up after tests."""
        self.plugin.shutdown()
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plugin_name(self):
        """Test plugin has correct name."""
        self.assertEqual(self.plugin.name, "memory")

    def test_tool_schemas(self):
        """Test plugin provides expected tool schemas."""
        schemas = self.plugin.get_tool_schemas()
        tool_names = [s.name for s in schemas]

        self.assertIn("store_memory", tool_names)
        self.assertIn("retrieve_memories", tool_names)
        self.assertIn("list_memory_tags", tool_names)

    def test_store_memory_schema_has_lifecycle_params(self):
        """Test that store_memory schema includes confidence, scope, evidence."""
        schemas = self.plugin.get_tool_schemas()
        store_schema = next(s for s in schemas if s.name == "store_memory")
        props = store_schema.parameters["properties"]

        self.assertIn("confidence", props)
        self.assertIn("scope", props)
        self.assertIn("evidence", props)

    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving a memory."""
        # Store a memory
        store_result = self.plugin.get_executors()["store_memory"]({
            "content": "The Runtime/Session split allows efficient subagent spawning.",
            "description": "jaato architecture explanation",
            "tags": ["architecture", "runtime", "session"]
        })

        self.assertEqual(store_result["status"], "success")
        self.assertIn("memory_id", store_result)

        # Retrieve by tags
        retrieve_result = self.plugin.get_executors()["retrieve_memories"]({
            "tags": ["architecture"],
            "limit": 5
        })

        self.assertEqual(retrieve_result["status"], "success")
        self.assertEqual(retrieve_result["count"], 1)
        self.assertEqual(
            retrieve_result["memories"][0]["description"],
            "jaato architecture explanation"
        )

    def test_store_defaults_to_raw_maturity(self):
        """Test that newly stored memories default to raw maturity."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Test content",
            "description": "Test",
            "tags": ["testing"]
        })

        self.assertEqual(result["maturity"], MATURITY_RAW)
        self.assertEqual(result["confidence"], 0.5)
        self.assertEqual(result["scope"], SCOPE_PROJECT)

    def test_store_with_lifecycle_fields(self):
        """Test storing a memory with explicit lifecycle fields."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "PostgreSQL JSONB indexes don't support partial matching",
            "description": "JSONB indexing limitation",
            "tags": ["postgresql", "jsonb_indexing"],
            "confidence": 0.9,
            "scope": "universal",
            "evidence": "Tested with CREATE INDEX ... USING gin and partial match query returned seq scan"
        })

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["maturity"], MATURITY_RAW)
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["scope"], SCOPE_UNIVERSAL)

    def test_store_clamps_confidence(self):
        """Test that confidence is clamped to 0.0-1.0 range."""
        executors = self.plugin.get_executors()

        # Over 1.0
        result = executors["store_memory"]({
            "content": "Test",
            "description": "Test",
            "tags": ["test_clamp"],
            "confidence": 5.0,
        })
        self.assertEqual(result["confidence"], 1.0)

        # Below 0.0
        result2 = executors["store_memory"]({
            "content": "Test2",
            "description": "Test2",
            "tags": ["test_clamp2"],
            "confidence": -1.0,
        })
        self.assertEqual(result2["confidence"], 0.0)

    def test_store_invalid_scope_defaults_to_project(self):
        """Test that invalid scope values fall back to 'project'."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Test",
            "description": "Test",
            "tags": ["test_scope"],
            "scope": "galactic",
        })

        self.assertEqual(result["scope"], SCOPE_PROJECT)

    def test_retrieve_includes_lifecycle_fields(self):
        """Test that retrieved memories include maturity, confidence, scope."""
        executors = self.plugin.get_executors()

        executors["store_memory"]({
            "content": "Some insight",
            "description": "Test insight",
            "tags": ["lifecycle_test"],
            "confidence": 0.8,
            "scope": "universal",
        })

        result = executors["retrieve_memories"]({"tags": ["lifecycle_test"]})
        mem = result["memories"][0]

        self.assertEqual(mem["maturity"], MATURITY_RAW)
        self.assertEqual(mem["confidence"], 0.8)
        self.assertEqual(mem["scope"], SCOPE_UNIVERSAL)

    def test_list_tags(self):
        """Test listing memory tags."""
        # Store some memories
        executors = self.plugin.get_executors()
        executors["store_memory"]({
            "content": "Auth explanation",
            "description": "Authentication flow",
            "tags": ["auth", "security"]
        })
        executors["store_memory"]({
            "content": "DB explanation",
            "description": "Database schema",
            "tags": ["database", "schema"]
        })

        # List tags
        result = executors["list_memory_tags"]({})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["memory_count"], 2)
        self.assertIn("auth", result["tags"])
        self.assertIn("database", result["tags"])

    def test_prompt_enrichment(self):
        """Test prompt enrichment with memory hints."""
        # Store a memory first
        self.plugin.get_executors()["store_memory"]({
            "content": "Detailed subagent explanation",
            "description": "How to spawn subagents efficiently",
            "tags": ["subagent", "spawning", "efficiency"]
        })

        # Test enrichment
        result = self.plugin.enrich_prompt(
            "How do I create a subagent efficiently?"
        )

        # Should find the memory and add hints
        self.assertIn("💡 **Available Memories**", result.prompt)
        self.assertIn("subagent", result.prompt.lower())
        self.assertEqual(result.metadata["memory_matches"], 1)

    def test_escalated_memories_hidden_from_enrichment(self):
        """Test that escalated memories are NOT surfaced in prompt enrichment.

        Once a memory has been promoted to a reference by the advisor,
        it should no longer appear in memory hints — the reference takes over.
        """
        executors = self.plugin.get_executors()

        # Store and then manually escalate a memory
        result = executors["store_memory"]({
            "content": "Barrel imports prevent circular deps",
            "description": "Always use barrel file for model imports",
            "tags": ["barrel_imports", "circular_deps"]
        })

        # Manually escalate it via storage
        memory = self.plugin._storage.get_by_id(result["memory_id"])
        memory.maturity = MATURITY_ESCALATED
        self.plugin._storage.update(memory)

        # Rebuild index to pick up the change
        self.plugin._indexer.clear()
        self.plugin._indexer.build_index(self.plugin._storage.load_all())

        # Enrichment should NOT find the escalated memory
        enrichment = self.plugin.enrich_prompt(
            "How should I handle barrel imports to avoid circular deps?"
        )
        self.assertEqual(enrichment.metadata.get("memory_matches", 0), 0)

    def test_dismissed_memories_hidden_from_enrichment(self):
        """Test that dismissed memories are NOT surfaced in prompt enrichment."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Wrong lesson",
            "description": "Incorrect observation",
            "tags": ["wrong_lesson_test"]
        })

        memory = self.plugin._storage.get_by_id(result["memory_id"])
        memory.maturity = MATURITY_DISMISSED
        self.plugin._storage.update(memory)

        self.plugin._indexer.clear()
        self.plugin._indexer.build_index(self.plugin._storage.load_all())

        enrichment = self.plugin.enrich_prompt(
            "Tell me about wrong_lesson_test"
        )
        self.assertEqual(enrichment.metadata.get("memory_matches", 0), 0)

    def test_validated_memories_visible_in_enrichment(self):
        """Test that validated memories ARE surfaced in prompt enrichment."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Validated insight about patterns",
            "description": "Confirmed pattern usage",
            "tags": ["validated_pattern_test"]
        })

        memory = self.plugin._storage.get_by_id(result["memory_id"])
        memory.maturity = MATURITY_VALIDATED
        self.plugin._storage.update(memory)

        self.plugin._indexer.clear()
        self.plugin._indexer.build_index(self.plugin._storage.load_all())

        enrichment = self.plugin.enrich_prompt(
            "How to use validated_pattern_test?"
        )
        self.assertEqual(enrichment.metadata["memory_matches"], 1)

    def test_no_enrichment_when_no_matches(self):
        """Test that prompts without matches are not modified."""
        original_prompt = "What is the weather today?"

        result = self.plugin.enrich_prompt(original_prompt)

        # Prompt should be unchanged
        self.assertEqual(result.prompt, original_prompt)
        self.assertEqual(result.metadata["memory_matches"], 0)

    def test_auto_approved_tools(self):
        """Test that memory tools are auto-approved."""
        auto_approved = self.plugin.get_auto_approved_tools()

        self.assertIn("store_memory", auto_approved)
        self.assertIn("retrieve_memories", auto_approved)
        self.assertIn("list_memory_tags", auto_approved)

    def test_subscribes_to_enrichment(self):
        """Test plugin subscribes to prompt enrichment."""
        self.assertTrue(self.plugin.subscribes_to_prompt_enrichment())

    def test_usage_count_increments(self):
        """Test that usage count increments on retrieval."""
        executors = self.plugin.get_executors()

        # Store memory
        executors["store_memory"]({
            "content": "Test content",
            "description": "Test memory",
            "tags": ["test"]
        })

        # Retrieve once
        result1 = executors["retrieve_memories"]({"tags": ["test"]})
        self.assertEqual(result1["memories"][0]["usage_count"], 1)

        # Retrieve again
        result2 = executors["retrieve_memories"]({"tags": ["test"]})
        self.assertEqual(result2["memories"][0]["usage_count"], 2)

    def test_single_letter_tags_rejected(self):
        """Test that single-letter tags are rejected."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Some content",
            "description": "Test memory",
            "tags": ["a", "b", "c"]
        })

        self.assertEqual(result["status"], "error")
        self.assertIn("at least 2 characters", result["message"])

    def test_short_tags_filtered_valid_kept(self):
        """Test that short tags are filtered but valid tags are kept."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Authentication details",
            "description": "Auth flow",
            "tags": ["a", "authentication", "x", "oauth"]
        })

        # Should succeed with only the valid tags
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tags"], ["authentication", "oauth"])

    def test_validate_memory_schema_rejects_short_tags(self):
        """Test that _validate_memory_schema rejects short tags."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["a", "b"]
        })

        self.assertIsNotNone(error)
        self.assertIn("at least 2 characters", error)

    def test_validate_memory_schema_accepts_valid_tags(self):
        """Test that _validate_memory_schema accepts valid tags."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["authentication", "oauth"]
        })

        self.assertIsNone(error)

    def test_validate_memory_schema_rejects_invalid_maturity(self):
        """Test that _validate_memory_schema rejects invalid maturity."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["test_tag"],
            "maturity": "legendary",
        })

        self.assertIsNotNone(error)
        self.assertIn("maturity", error)

    def test_validate_memory_schema_accepts_valid_maturity(self):
        """Test that _validate_memory_schema accepts valid maturity values."""
        for maturity in (MATURITY_RAW, MATURITY_VALIDATED, MATURITY_ESCALATED, MATURITY_DISMISSED):
            error = self.plugin._validate_memory_schema({
                "content": "test",
                "description": "test",
                "tags": ["test_tag"],
                "maturity": maturity,
            })
            self.assertIsNone(error, f"maturity={maturity} should be valid")

    def test_validate_memory_schema_rejects_invalid_confidence(self):
        """Test that _validate_memory_schema rejects out-of-range confidence."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["test_tag"],
            "confidence": 2.0,
        })

        self.assertIsNotNone(error)
        self.assertIn("confidence", error)

    def test_validate_memory_schema_rejects_invalid_scope(self):
        """Test that _validate_memory_schema rejects invalid scope."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["test_tag"],
            "scope": "galactic",
        })

        self.assertIsNotNone(error)
        self.assertIn("scope", error)

    def test_store_result_includes_telemetry_dict(self):
        """Test that store_memory result includes _telemetry for span enrichment."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Test telemetry emission",
            "description": "Telemetry test",
            "tags": ["telemetry_test"],
            "confidence": 0.8,
            "scope": "universal",
            "evidence": "Observed during testing",
        })

        self.assertIn("_telemetry", result)
        telem = result["_telemetry"]
        self.assertEqual(telem["jaato.memory.operation"], "store")
        self.assertEqual(telem["jaato.memory.maturity"], "raw")
        self.assertEqual(telem["jaato.memory.confidence"], 0.8)
        self.assertEqual(telem["jaato.memory.scope"], "universal")
        self.assertTrue(telem["jaato.memory.has_evidence"])
        self.assertEqual(telem["jaato.memory.tag_count"], 1)

    def test_retrieve_result_includes_telemetry_dict(self):
        """Test that retrieve_memories result includes _telemetry for span enrichment."""
        executors = self.plugin.get_executors()

        executors["store_memory"]({
            "content": "Insight A",
            "description": "First insight",
            "tags": ["telem_retrieve_test"],
            "confidence": 0.7,
            "scope": "project",
        })
        executors["store_memory"]({
            "content": "Insight B",
            "description": "Second insight",
            "tags": ["telem_retrieve_test"],
            "confidence": 0.9,
            "scope": "universal",
        })

        result = executors["retrieve_memories"]({"tags": ["telem_retrieve_test"]})

        self.assertIn("_telemetry", result)
        telem = result["_telemetry"]
        self.assertEqual(telem["jaato.memory.operation"], "retrieve")
        self.assertEqual(telem["jaato.memory.count_retrieved"], 2)
        self.assertIn("raw", telem["jaato.memory.maturities_retrieved"])
        self.assertIn("project", telem["jaato.memory.scopes_retrieved"])
        self.assertIn("universal", telem["jaato.memory.scopes_retrieved"])
        self.assertAlmostEqual(telem["jaato.memory.avg_confidence"], 0.8, places=2)

    def test_list_tags_result_includes_telemetry_dict(self):
        """Test that list_memory_tags result includes _telemetry for span enrichment."""
        executors = self.plugin.get_executors()

        executors["store_memory"]({
            "content": "Test",
            "description": "Test",
            "tags": ["telem_list_test"],
        })

        result = executors["list_memory_tags"]({})

        self.assertIn("_telemetry", result)
        telem = result["_telemetry"]
        self.assertEqual(telem["jaato.memory.operation"], "list_tags")
        self.assertEqual(telem["jaato.memory.total_count"], 1)
        self.assertGreaterEqual(telem["jaato.memory.tag_count"], 1)
        self.assertEqual(telem["jaato.memory.count_raw"], 1)

    def test_source_agent_captured(self):
        """Test that source_agent is captured from plugin config."""
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(Path(self.temp_dir) / "agent_test.jsonl"),
            "agent_name": "analyst-codebase-documentation",
        })

        result = plugin.get_executors()["store_memory"]({
            "content": "Test from named agent",
            "description": "Agent source test",
            "tags": ["agent_test"],
        })

        memory = plugin._storage.get_by_id(result["memory_id"])
        self.assertEqual(memory.source_agent, "analyst-codebase-documentation")
        plugin.shutdown()


class TestMemoryStorageBackwardCompat(unittest.TestCase):
    """Test backward compatibility when loading old JSONL without lifecycle fields."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = str(Path(self.temp_dir) / "compat_test.jsonl")

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_old_format_loads_with_defaults(self):
        """Test that JSONL lines without lifecycle fields load with defaults."""
        # Write an old-format memory (no maturity, confidence, scope, etc.)
        old_memory = {
            "id": "mem_old_001",
            "content": "Old format memory",
            "description": "Before lifecycle fields",
            "tags": ["legacy", "compat"],
            "timestamp": "2024-01-01T00:00:00",
            "usage_count": 3,
            "last_accessed": "2024-06-01T00:00:00",
        }
        with open(self.storage_path, 'w') as f:
            f.write(json.dumps(old_memory) + '\n')

        storage = MemoryStorage(self.storage_path)
        memories = storage.load_all()

        self.assertEqual(len(memories), 1)
        mem = memories[0]
        self.assertEqual(mem.id, "mem_old_001")
        self.assertEqual(mem.content, "Old format memory")
        # Lifecycle fields should have defaults
        self.assertEqual(mem.maturity, MATURITY_RAW)
        self.assertEqual(mem.confidence, 0.5)
        self.assertEqual(mem.scope, SCOPE_PROJECT)
        self.assertIsNone(mem.evidence)
        self.assertIsNone(mem.source_agent)
        self.assertIsNone(mem.source_session)

    def test_unknown_keys_ignored(self):
        """Test that unknown JSON keys don't crash loading."""
        weird_memory = {
            "id": "mem_weird_001",
            "content": "Has extra keys",
            "description": "Unknown fields test",
            "tags": ["weird"],
            "timestamp": "2024-01-01T00:00:00",
            "usage_count": 0,
            "some_future_field": "should be ignored",
            "another_unknown": 42,
        }
        with open(self.storage_path, 'w') as f:
            f.write(json.dumps(weird_memory) + '\n')

        storage = MemoryStorage(self.storage_path)
        memories = storage.load_all()

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, "mem_weird_001")

    def test_maturity_queries(self):
        """Test maturity-based storage queries."""
        storage = MemoryStorage(self.storage_path)

        # Write memories with different maturities
        for i, maturity in enumerate([MATURITY_RAW, MATURITY_RAW, MATURITY_VALIDATED, MATURITY_ESCALATED]):
            mem = Memory(
                id=f"mem_mat_{i}",
                content=f"Memory {i}",
                description=f"Maturity test {i}",
                tags=["maturity_test"],
                timestamp=f"2024-01-0{i+1}T00:00:00",
                maturity=maturity,
            )
            storage.save(mem)

        # get_pending_curation should return only raw
        pending = storage.get_pending_curation()
        self.assertEqual(len(pending), 2)
        for m in pending:
            self.assertEqual(m.maturity, MATURITY_RAW)

        # count_by_maturity
        counts = storage.count_by_maturity()
        self.assertEqual(counts[MATURITY_RAW], 2)
        self.assertEqual(counts[MATURITY_VALIDATED], 1)
        self.assertEqual(counts[MATURITY_ESCALATED], 1)

        # search_by_maturity
        validated = storage.search_by_maturity({MATURITY_VALIDATED})
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].maturity, MATURITY_VALIDATED)

    def test_search_by_tags_active_only(self):
        """Test that search_by_tags respects active_only filter."""
        storage = MemoryStorage(self.storage_path)

        # Store one active and one escalated memory with same tags
        storage.save(Memory(
            id="mem_active",
            content="Active memory",
            description="Active",
            tags=["shared_tag"],
            timestamp="2024-01-02T00:00:00",
            maturity=MATURITY_RAW,
        ))
        storage.save(Memory(
            id="mem_escalated",
            content="Escalated memory",
            description="Escalated",
            tags=["shared_tag"],
            timestamp="2024-01-01T00:00:00",
            maturity=MATURITY_ESCALATED,
        ))

        # active_only=True (default) should return only the active one
        active = storage.search_by_tags(["shared_tag"])
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].id, "mem_active")

        # active_only=False should return both
        all_mems = storage.search_by_tags(["shared_tag"], active_only=False)
        self.assertEqual(len(all_mems), 2)


if __name__ == "__main__":
    unittest.main()
