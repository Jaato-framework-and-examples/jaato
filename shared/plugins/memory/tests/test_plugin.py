"""Tests for MemoryPlugin."""

import tempfile
import unittest
from pathlib import Path

from ..plugin import MemoryPlugin


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


if __name__ == "__main__":
    unittest.main()
