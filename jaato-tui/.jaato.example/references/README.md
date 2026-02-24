# References

Reference sources provide documentation and context that agents can access
during a session. Each JSON file here defines one reference entry.

## Reference Schema

```json
{
  "id": "unique-identifier",
  "name": "Human-Readable Name",
  "description": "Brief description of what this reference contains.",
  "type": "local",
  "path": "./docs/architecture.md",
  "mode": "selectable",
  "tags": ["architecture", "design"],
  "fetchHint": "Read the Overview section first."
}
```

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier (lowercase, hyphens) |
| `name` | Yes | Display name |
| `description` | Yes | What the reference contains |
| `type` | Yes | `"local"` (file), `"url"` (web), `"inline"` (embedded text) |
| `path` | For local | Relative or absolute path to the file/directory |
| `url` | For url | Web URL to fetch |
| `content` | For inline | The reference text itself |
| `mode` | No | `"auto"` (always loaded) or `"selectable"` (user chooses). Default: `"selectable"` |
| `tags` | No | Tags for filtering and organization |
| `fetchHint` | No | Hint for the agent on what to read first |

## Auto-Discovery

Files placed in `.jaato/references/` are automatically discovered. Each `.json`
file should contain a single reference object (not an array).

## Modes

- **auto**: Reference is always included in the agent's context. Use sparingly
  for foundational docs that every task needs.
- **selectable**: Agent can choose to load the reference when relevant. Better
  for large reference sets.

## Examples

- **project-docs.json** -- Example reference pointing to project documentation
