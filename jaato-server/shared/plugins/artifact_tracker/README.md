# Artifact Tracker Plugin

The Artifact Tracker plugin helps the AI model track artifacts (documents, tests, configs, code) created during sessions and reminds it to review related artifacts when source files change.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Model / Agent                         │
│   Uses tools: trackArtifact, notifyChange, checkRelated...  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ArtifactTrackerPlugin                      │
│  • Tracks artifacts with relationships                       │
│  • Flags dependents for review on changes                    │
│  • Auto-discovers dependencies via LSP                       │
│  • Persists state to JSON file                              │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────┐          ┌─────────────────────┐
│  .artifact_tracker  │          │    LSP Plugin       │
│  .json              │          │  (for dependency    │
│  (persistence)      │          │   discovery)        │
└─────────────────────┘          └─────────────────────┘
```

## Key Features

1. **Artifact Tracking**: Register files as artifacts with metadata, tags, and relationships
2. **Dependency Tracking**: Link artifacts to their source files via `related_to`
3. **Review Flagging**: Automatically flag dependent artifacts when source files change
4. **LSP Integration**: Auto-discover file dependencies using Language Server Protocol
5. **State Persistence**: Save and load artifact registry to/from JSON file

## Configuration

```python
registry.expose_all({
    'artifact_tracker': {
        'storage_path': '.jaato/.artifact_tracker.json',  # default
        'auto_load': True  # load existing state on init
    }
})
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `storage_path` | string | `.jaato/.artifact_tracker.json` | Path to JSON persistence file |
| `auto_load` | boolean | `true` | Load existing state on initialization |

## Tools

### trackArtifact

Register a new artifact with optional dependencies.

```json
{
  "path": "tests/test_api.py",
  "type": "test",
  "description": "Unit tests for the API module",
  "related_to": ["src/api.py"],
  "tags": ["unit-test", "api"]
}
```

### updateArtifact

Modify artifact metadata, add/remove relations or tags.

```json
{
  "path": "tests/test_api.py",
  "add_related_to": ["src/utils.py"],
  "add_tags": ["integration"]
}
```

### listArtifacts

Show tracked artifacts, filterable by type, tag, or review status.

```json
{
  "filter_type": "test",
  "filter_tag": "api",
  "filter_review_status": "needs_review"
}
```

### checkRelated

Preview artifacts before modifying a file (impact analysis).

```json
{
  "path": "src/api.py"
}
```

Returns list of artifacts that would be flagged if this file changes.

### notifyChange

Flag dependent artifacts after modifying a source file.

```json
{
  "path": "src/api.py",
  "reason": "Updated API endpoint signatures"
}
```

### acknowledgeReview

Mark artifact as reviewed (clears review flag).

```json
{
  "path": "tests/test_api.py"
}
```

### flagForReview

Manually flag an artifact for review.

```json
{
  "path": "docs/api-reference.md",
  "reason": "API changed, docs may need update"
}
```

### removeArtifact

Stop tracking an artifact.

```json
{
  "path": "tests/test_old_feature.py"
}
```

## LSP Integration

When files are modified via `updateFile`, `writeNewFile`, `lsp_rename_symbol`, or `lsp_apply_code_action`, the artifact tracker automatically:

1. Uses LSP to discover which files depend on the modified file (via `find_references`)
2. Also checks the tracked artifact registry for previously known relationships
3. Flags those dependent files for review
4. Shows a notification listing flagged files

### Dual-Source Dependency Discovery

The plugin uses two sources to find dependents:

| Source | Purpose | Finds |
|--------|---------|-------|
| **LSP** | Real-time code analysis | New files that import/reference the modified file |
| **Artifact Registry** | Tracked relationships | Previously known dependents, including deleted files |

This ensures that:
- New dependencies are discovered automatically via LSP
- Deleted files that were previously tracked show as "(missing)" in the notification
- No dependent is missed due to stale LSP index or deleted files

```
File A.py modified via updateFile
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  LSP Plugin (priority 30)                                    │
│  • Runs diagnostics                                          │
│  • Notification: "checked A.py, no issues found"             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Artifact Tracker (priority 50)                              │
│  • Calls lsp.get_file_dependents("A.py")                     │
│  • Discovers: B.py, C.py import from A.py                    │
│  • Auto-tracks B.py, C.py as artifacts if not already        │
│  • Flags them for review                                     │
│  • Notification: "flagged for review: B.py, C.py"            │
└─────────────────────────────────────────────────────────────┘
```

### User Notification

When dependencies are discovered, users see:

```
  ╭ result ← lsp: checked api.py, no issues found
  ╰ result ← artifact_tracker: flagged for review: test_api.py, handler.py
```

If a previously tracked dependent file has been deleted:

```
  ╰ result ← artifact_tracker: flagged for review: test_api.py, old_handler.py (missing)
```

The "(missing)" marker alerts the user that a dependent file no longer exists.

### System Instructions

The plugin injects dynamic system instructions that include:
- Artifacts currently flagged for review
- Total count of tracked artifacts
- Workflow guidance for using the tools

## Artifact Types

| Type | Description |
|------|-------------|
| `document` | Documentation, READMEs, guides |
| `test` | Test files (unit, integration, e2e) |
| `config` | Configuration files |
| `code` | Source code files |
| `schema` | API schemas, database schemas |
| `script` | Build scripts, automation |
| `data` | Data files, fixtures |
| `other` | Anything else |

## Review Statuses

| Status | Meaning |
|--------|---------|
| `current` | Artifact is up to date |
| `needs_review` | Source dependency changed, needs review |
| `in_review` | Currently being reviewed |
| `acknowledged` | Review acknowledged, may still need work |

## Example Workflow

```
1. Model creates README.md → tracks with trackArtifact
2. Model creates tests/test_api.py → tracks it, relates to src/api.py
3. Model modifies src/api.py:
   a. LSP automatically discovers test_api.py depends on api.py
   b. Artifact tracker flags test_api.py for review
   c. User sees notification: "flagged for review: test_api.py"
4. Next turn's system instructions remind model:
   "⚠️ Artifacts needing review: test_api.py (dependency api.py was modified)"
5. Model reviews test_api.py, updates if needed
6. Model calls acknowledgeReview to clear the flag
```

## Persistence Format

The artifact registry is persisted as JSON:

```json
{
  "artifacts": [
    {
      "id": "uuid-1234",
      "path": "tests/test_api.py",
      "artifact_type": "test",
      "description": "Unit tests for API module",
      "related_to": ["src/api.py"],
      "tags": ["unit-test", "api"],
      "review_status": "needs_review",
      "review_reason": "Dependency src/api.py was modified",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T14:20:00Z"
    }
  ]
}
```

## Auto-Approved Tools

All artifact tracker tools are auto-approved (no permission prompts) because they are read-only tracking operations that don't modify actual files.
