# Memory Plugin

A plugin that enables the model to build and maintain a persistent knowledge base across sessions through self-curation.

## Overview

The Memory plugin allows the AI model to:
- **Store** valuable explanations, insights, and architectural knowledge for future reference
- **Retrieve** stored memories when relevant to current conversations
- **Build** a persistent, project-specific knowledge base over time

Unlike the `references` plugin (which provides access to external documentation), the memory plugin stores **model-generated content** that the model itself deems valuable for future sessions.

## Architecture

### Two-Phase Retrieval System

1. **Prompt Enrichment (Phase 1)**
   - Plugin analyzes user prompts for keywords
   - Injects lightweight hints about available memories
   - Model sees: "💡 **Available Memories** (use retrieve_memories to access)"
   - No prompt bloat - only metadata, not full content

2. **Model-Driven Retrieval (Phase 2)**
   - Model decides if memories are relevant
   - Calls `retrieve_memories` to fetch full content
   - Model has full control over what to retrieve

### Components

```
memory/
├── models.py       # Memory and MemoryMetadata data classes
├── storage.py      # JSONL-based storage backend
├── indexer.py      # Keyword extraction and tag indexing
├── plugin.py       # Main MemoryPlugin class
└── tests/          # Unit tests
```

## Usage

### Initialization

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.expose_plugin("memory", config={
    "storage_path": ".jaato/memories.jsonl"  # Optional, this is the default
})
```

### Model Tools

The plugin provides three tools for the model:

#### 1. `store_memory`

Store information for future sessions.

```json
{
  "content": "The Runtime/Session split allows subagents to share provider config...",
  "description": "jaato Runtime/Session architecture and efficient subagent spawning",
  "tags": ["architecture", "runtime", "session", "subagents"]
}
```

**When the model should use this:**
- After providing comprehensive explanations
- When documenting project-specific patterns
- After analyzing complex architectures

**When NOT to use:**
- Ephemeral responses to simple questions
- User-specific information
- Temporary troubleshooting steps

#### 2. `retrieve_memories`

Retrieve stored memories by tags.

```json
{
  "tags": ["subagents", "spawning"],
  "limit": 3
}
```

**Response format:**
```json
{
  "status": "success",
  "count": 1,
  "memories": [
    {
      "id": "mem_20231211_143022",
      "description": "jaato Runtime/Session architecture...",
      "content": "The Runtime/Session split allows...",
      "tags": ["architecture", "runtime", "session"],
      "stored": "2023-12-11T14:30:22",
      "usage_count": 3
    }
  ]
}
```

#### 3. `list_memory_tags`

Discover what has been stored.

```json
{}
```

**Response format:**
```json
{
  "status": "success",
  "tags": ["architecture", "auth", "database", "runtime", "session"],
  "count": 5,
  "memory_count": 12
}
```

## Example Flows

### Session 1: Storing Knowledge

```
User: "Explain how the Runtime/Session architecture works in jaato"

Model: [Generates comprehensive explanation]

       The architecture separates shared resources (Runtime) from
       per-agent state (Session). This allows subagents to spawn
       quickly by reusing the parent's provider config...

       [Model calls store_memory with full explanation and tags]
```

### Session 2: Retrieving Knowledge

```
User: "How do I create a subagent efficiently?"

[Plugin enriches prompt:]
  💡 **Available Memories** (use retrieve_memories to access):
    - [architecture, runtime, session, subagents]: jaato Runtime/Session
      architecture and efficient subagent spawning

Model: [Sees hint about available memory]
       [Calls retrieve_memories with tags=["subagents", "runtime"]]
       [Receives stored explanation from Session 1]

       To efficiently create subagents, use runtime.create_session().
       This is lightweight because...
```

## Storage Format

Memories are stored in JSONL (JSON Lines) format:

```jsonl
{"id": "mem_20231211_143022", "content": "...", "description": "...", "tags": [...], "timestamp": "2023-12-11T14:30:22", "usage_count": 3, "last_accessed": "2023-12-11T15:45:00"}
{"id": "mem_20231211_150133", "content": "...", "description": "...", "tags": [...], "timestamp": "2023-12-11T15:01:33", "usage_count": 0, "last_accessed": null}
```

Default location: `.jaato/memories.jsonl` (per-project)

## Indexing

The plugin maintains an in-memory index for efficient lookup:

- **Tag Index**: Maps tags to memory IDs
- **Metadata Cache**: Stores lightweight metadata (no full content)

**Matching Strategy:**
1. Exact tag matches (keyword == tag)
2. Partial matches (keyword in tag or tag in keyword)
3. Results sorted by recency (most recent first)

**Keyword Extraction:**
- Extracts alphanumeric words from prompts
- Filters common stopwords
- Filters short words (< 4 characters)

## Best Practices

### For Model Behavior

**Store memories when:**
- Providing deep architectural explanations
- Documenting project conventions
- Analyzing complex code patterns
- Explaining non-obvious design decisions

**Use specific, searchable tags:**
- ✅ Good: `["oauth2", "jwt_auth", "token_refresh"]`
- ❌ Bad: `["auth", "stuff", "things"]`

**Write clear descriptions:**
- ✅ Good: "OAuth2 flow with JWT refresh tokens and Redis storage"
- ❌ Bad: "Authentication thing we discussed"

### For Users

**Storage location:**
- Keep `.jaato/memories.jsonl` in `.gitignore` (project-specific knowledge)
- Or commit it for team-shared knowledge base

**Maintenance:**
- Periodically review stored memories
- Remove outdated or incorrect information
- Consider memory lifecycle (currently append-only)

## Configuration

```python
{
    "storage_path": ".jaato/memories.jsonl",  # Path to storage file
    "enrichment_limit": 5                      # Max hints in prompt (future)
}
```

## Future Enhancements

Potential improvements:

1. **Vector Embeddings**: Semantic similarity matching (beyond keyword matching)
2. **Memory Updates**: Allow model to update/invalidate old memories
3. **Global Memories**: Cross-project knowledge sharing
4. **Expiration**: TTL for stale memories
5. **Categorization**: Memory types (architecture, patterns, bugs, decisions)
6. **Compression**: Summarize old memories to reduce storage
7. **Analytics**: Usage statistics, most valuable memories

## Testing

Run the test suite:

```bash
python3 -m pytest shared/plugins/memory/tests/ -v
```

Or with unittest:

```bash
python3 -m unittest discover -s shared/plugins/memory/tests -v
```

## License

Same as jaato project.
