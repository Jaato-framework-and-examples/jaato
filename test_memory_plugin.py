#!/usr/bin/env python3
"""Quick verification script for memory plugin.

Run this from your venv after installing with: pip install -e .
"""

import tempfile
from pathlib import Path

from shared.plugins.memory.plugin import MemoryPlugin


def test_memory_plugin():
    """Test basic memory plugin functionality."""
    print("Testing Memory Plugin...")
    print("=" * 60)

    # Create temp storage
    temp_dir = tempfile.mkdtemp()
    storage_path = str(Path(temp_dir) / "test_memories.jsonl")

    # Initialize plugin
    plugin = MemoryPlugin()
    plugin.initialize({"storage_path": storage_path})

    print(f"✓ Plugin initialized: {plugin.name}")

    # Check tool schemas
    schemas = plugin.get_tool_schemas()
    tool_names = [s.name for s in schemas]
    print(f"✓ Tools available: {tool_names}")

    assert "store_memory" in tool_names
    assert "retrieve_memories" in tool_names
    assert "list_memory_tags" in tool_names

    # Test storing
    executors = plugin.get_executors()
    result = executors["store_memory"]({
        "content": "The Runtime/Session split allows efficient subagent spawning.",
        "description": "jaato architecture explanation",
        "tags": ["architecture", "runtime", "session"]
    })

    print(f"✓ Store result: {result['status']}")
    assert result["status"] == "success"

    # Test retrieving
    result = executors["retrieve_memories"]({
        "tags": ["architecture"],
        "limit": 5
    })

    print(f"✓ Retrieve result: {result['status']}, found {result['count']} memories")
    assert result["status"] == "success"
    assert result["count"] == 1

    # Test prompt enrichment
    enrichment = plugin.enrich_prompt("How do I work with runtime and session?")
    print(f"✓ Prompt enrichment: {enrichment.metadata['memory_matches']} matches")
    assert enrichment.metadata['memory_matches'] > 0

    # Test list tags
    result = executors["list_memory_tags"]({})
    print(f"✓ List tags: {result['tags']}")
    assert "architecture" in result["tags"]

    # Cleanup
    plugin.shutdown()
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 60)
    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_memory_plugin()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
