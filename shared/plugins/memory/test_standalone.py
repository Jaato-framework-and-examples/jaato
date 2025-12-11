"""Standalone test script for memory plugin (no external dependencies)."""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.plugins.memory.plugin import MemoryPlugin


def test_basic_functionality():
    """Test basic plugin functionality."""
    print("Testing Memory Plugin...")
    print("-" * 60)

    # Create temp storage
    temp_dir = tempfile.mkdtemp()
    storage_path = str(Path(temp_dir) / "test_memories.jsonl")

    # Initialize plugin
    print("✓ Creating plugin instance")
    plugin = MemoryPlugin()

    print("✓ Initializing plugin")
    plugin.initialize({"storage_path": storage_path})

    # Check basic properties
    print(f"✓ Plugin name: {plugin.name}")
    assert plugin.name == "memory", f"Expected name 'memory', got {plugin.name}"

    # Check tool schemas
    print("✓ Checking tool schemas")
    schemas = plugin.get_tool_schemas()
    tool_names = [s.name for s in schemas]
    print(f"  Available tools: {tool_names}")
    assert "store_memory" in tool_names
    assert "retrieve_memories" in tool_names
    assert "list_memory_tags" in tool_names

    # Check prompt enrichment
    print("✓ Checking prompt enrichment subscription")
    assert plugin.subscribes_to_prompt_enrichment()

    # Test storing a memory
    print("✓ Testing store_memory")
    executors = plugin.get_executors()
    result = executors["store_memory"]({
        "content": "The Runtime/Session split allows efficient subagent spawning.",
        "description": "jaato architecture explanation",
        "tags": ["architecture", "runtime", "session"]
    })
    print(f"  Store result: {result['status']}")
    print(f"  Memory ID: {result['memory_id']}")
    assert result["status"] == "success"

    # Test retrieving
    print("✓ Testing retrieve_memories")
    result = executors["retrieve_memories"]({
        "tags": ["architecture"],
        "limit": 5
    })
    print(f"  Retrieve result: {result['status']}")
    print(f"  Found {result['count']} memories")
    assert result["status"] == "success"
    assert result["count"] == 1

    # Test listing tags
    print("✓ Testing list_memory_tags")
    result = executors["list_memory_tags"]({})
    print(f"  Found {result['count']} unique tags")
    print(f"  Tags: {result['tags']}")
    assert result["status"] == "success"
    assert "architecture" in result["tags"]

    # Test prompt enrichment
    print("✓ Testing prompt enrichment")
    enrichment_result = plugin.enrich_prompt(
        "How do I work with the runtime and session?"
    )
    print(f"  Matches found: {enrichment_result.metadata['memory_matches']}")
    if enrichment_result.metadata['memory_matches'] > 0:
        print("  Enriched prompt preview:")
        preview = enrichment_result.prompt[-200:] if len(enrichment_result.prompt) > 200 else enrichment_result.prompt
        print(f"  ...{preview}")

    # Test no matches
    print("✓ Testing prompt with no matches")
    result = plugin.enrich_prompt("What is the weather today?")
    print(f"  Matches found: {result.metadata['memory_matches']}")
    assert result.metadata['memory_matches'] == 0

    # Cleanup
    print("✓ Shutting down plugin")
    plugin.shutdown()

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("-" * 60)
    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
