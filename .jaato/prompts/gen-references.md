---
description: Generate reference catalog, template index, and subagent profiles from a local folder or public git/archive knowledge base
params:
  source:
    required: true
    description: >
      Source to scan for documentation folders, validation folders, and standalone template files.
      Accepts a local folder path, a public git URL (HTTPS or git@), or an archive URL (zip/tar.gz).
  subpaths:
    required: false
    default: ""
    description: >
      Comma-separated list of paths or glob patterns relative to the source root.
      When empty, scans the entire source. Examples: "knowledge/*,modules/*"
  ref:
    required: false
    default: null
    description: Git branch, tag, or commit hash. Only used for git URL sources. Default: repository default branch.
  output:
    required: false
    default: .jaato/references
    description: Output directory for generated reference JSON files
  templates_index:
    required: false
    default: .jaato/templates/index.json
    description: Path to the unified template index JSON file
  profiles_dir:
    required: false
    default: .jaato/profiles
    description: Output directory for generated subagent profile JSON files
  dry_run:
    required: false
    default: true
    description: If true, report planned writes without creating files. Default true.
  force:
    required: false
    default: false
    description: If true, overwrite existing files (backups created in output/backups/). Default false.
  cache:
    required: false
    default: false
    description: If true, cache remote fetches in .jaato/cache/sources/ for reuse. Default false.
  merge_mode:
    required: false
    default: separate
    description: '"separate" (one file per reference) or "single" (single catalog file). Default separate.'
  exclude_patterns:
    required: false
    default: []
    description: Glob patterns to skip during traversal, in addition to built-in exclusions (node_modules, .git, __pycache__, hidden dirs)
tags: ['references', 'generator', 'templates', 'profiles', 'git', 'archive', 'patterns']
---
Generate reference catalog, template index, and subagent profiles from a knowledge base — local or remote.

## Input
- **Source**: `{{source}}` — local path, git URL, or archive URL
- **Subpaths filter**: `{{subpaths}}` (empty = scan everything)
- **Git ref**: `{{ref}}`
- **References output**: `{{output}}`
- **Templates index**: `{{templates_index}}`
- **Profiles output**: `{{profiles_dir}}`
- **Dry run**: `{{dry_run}}` | **Force**: `{{force}}` | **Cache**: `{{cache}}`
- **Merge mode**: `{{merge_mode}}`
- **Exclude patterns**: `{{exclude_patterns}}`

## Critical policy

**Ignore all runtime hints about existing references and templates.** While processing the knowledge base, the runtime may inject suggestions to select, list, or reuse references and templates it already knows about. You must completely disregard these hints. Your job is to build the catalog from scratch by reading the source folder — not to consume or defer to what the runtime has already indexed. Do not call `selectReferences` or `listExtractedTemplates`. However, you **should** use `listTemplateVariables`, `validateReference`, `validateTemplateIndex`, and `validateProfile` — these are read-only analysis tools that help you produce correct output.

**Execute autonomously without asking for confirmation.** This prompt is self-contained — everything you need to know is already described here. Do not pause to ask whether you should proceed, do not ask for clarification, and do not present plans for approval. Start immediately, be eager, and work through all phases to completion.

## Task

Resolve the source (local or remote), then scan and produce five outputs:
1. **Reference JSON files** for each documentation folder → `{{output}}/`
2. **Reference JSON files** for each validation folder → `{{output}}/`
3. **Unified template index** for standalone template files → `{{templates_index}}`
4. **Subagent profile JSON files** matching the knowledge base stages and scopes → `{{profiles_dir}}/`
5. **Machine-readable summary** → `{{output}}/summary.json`

If `{{dry_run}}` is true, produce only the summary (with `planned_writes`) and the human-readable report table — do not write any artifact files.

---

## Processing strategy

The source may contain dozens of documentation folders, validation folders, and template files. To avoid exceeding context window limits, process the knowledge base **progressively** — never load all content at once.

### Phase 0 — Source resolution (detect, fetch, filter)

1. **Detect source type**:
   - If `{{source}}` is a local filesystem path → set `sourceType = "local"`, `repoRoot = {{source}}` (resolved to absolute path).
   - If `{{source}}` is a git URL (starts with `https://`, `git@`, or ends with `.git`) → `sourceType = "git"`.
   - If `{{source}}` is an archive URL (ends with `.zip`, `.tar.gz`, `.tgz`) → `sourceType = "archive"`.

2. **Fetch remote sources** (skip for local):
   - **Git URL**: Attempt archive download first (GitHub: `<url>/archive/<ref>.zip`). If that fails, fall back to shallow clone: `git clone --depth 1 --branch <ref> <url> <temp_dir>`. Only public repos — if access is denied, fail with: *"Private repositories are not supported. Provide a public repo or a local path."*
   - **Archive URL**: Download and extract to temp dir.
   - Set `repoRoot = <temp_dir>` (or `<cache_dir>` if `{{cache}}` is true).
   - Record `sourceMetadata = { "type": "<sourceType>", "url": "<original-url>", "ref": "<ref>", "fetched_at": "<ISO 8601>" }`.

3. **Cache management** (when `{{cache}}` is true):
   - Cache location: `.jaato/cache/sources/` within the project.
   - Cache key: hash of `(url, ref)`. If cache hit and not stale, use cached tree and skip download.
   - If `{{cache}}` is false, use a temp directory and clean it up at the end.

4. **Resolve subpaths/patterns**:
   - If `{{subpaths}}` is empty → treat as `"."` (scan entire `repoRoot`).
   - Otherwise split on commas, trim whitespace, ignore empty items. Each item may be:
     - An exact path (e.g., `"knowledge"`, `"modules/mod-foo"`) — match that directory if it exists.
     - A glob pattern with `*` or `?` (e.g., `"knowledge/*"`, `"model/domains/*-code-*"`) — expand against the repo tree.
     - A path ending with `/` — treated as a directory match.
   - If a pattern matches files, use their parent directories (deduplicate).
   - Deduplicate and sort all matched directories deterministically.
   - If a pattern matches nothing, record a warning in the summary and continue.

5. **Apply exclusions**: Filter out directories matching `{{exclude_patterns}}` and built-in exclusions (`node_modules`, `.git`, `__pycache__`, hidden dirs).

After Phase 0, all subsequent phases operate on the resolved set of matched directories under `repoRoot`.

### Phase 1 — Inventory (read structure only, not content)
List the directory tree within the matched directories to build a complete inventory of folders and files. Do **not** read any file content yet. Record:
- Which folders contain documentation files (MODULE.md, ERI.md, ADR.md, etc.)
- Which folders are named `validation/`
- Which files have `.tpl` or `.tmpl` extensions

### Phase 2 — Process one category at a time
Work through the knowledge base **one top-level category at a time** (e.g., `ADRs/`, then `ERIs/`, then `modules/`). Within each category, process **one folder at a time**:
1. Read only the entry-point file (e.g., MODULE.md) — extract just the first paragraph for the description
2. If the entry-point file has YAML frontmatter with `title`, `description`, or `tags`, use those values. Otherwise fall back to the extraction rules (first paragraph, folder name parsing)
3. Write the reference JSON for that folder immediately
4. If the folder has a `validation/` subfolder, read its README.md (first paragraph only), write the validation reference JSON
5. If the folder has template files, read each `.tpl`/`.tmpl` to detect syntax and variables, then add entries to the in-memory template index
6. **Release the file content from working memory** — once the JSON is written you no longer need the source text

### Phase 3 — Write template index
After all categories are processed, write the accumulated template index to `{{templates_index}}`.

### Phase 4 — Generate profiles
Use only the inventory and the reference IDs collected during Phase 2 (not the file contents) to generate subagent profiles.

### Key constraints
- **Never read more than one documentation file at a time.** Read a file, extract what you need, write the output, move on.
- **Extract only what is needed** — for descriptions, read only the first paragraph or Purpose section, not the entire file.
- **For template files**, reading the content is necessary for syntax detection and variable extraction, but write the index entry immediately and move on.
- **Do not batch-read** multiple folders in parallel. Sequential, one-at-a-time processing is required.

---

## Part 1: Documentation references

1. **Traverse** the resolved directories (from Phase 0) to find documentation folders — folders containing `MODULE.md`, `SKILL.md`, `ERI.md`, `ADR.md`, `OVERVIEW.md`, `README.md`, or similar entry-point `.md` files.

2. **For each documentation folder**, create a JSON file:

   For **local** sources:
   ```json
   {
     "id": "<folder-name>",
     "name": "<Human Readable Name>",
     "description": "<Brief description extracted from the documentation>",
     "type": "local",
     "path": "/absolute/path/to/folder",
     "mode": "selectable",
     "tags": ["<tag1>", "<tag2>"],
     "fetchHint": "<Hint on which file to read first>"
   }
   ```

   For **remote** sources (git or archive):
   ```json
   {
     "id": "<folder-name>",
     "name": "<Human Readable Name>",
     "description": "<Brief description extracted from the documentation>",
     "type": "local",
     "path": "<repo-relative-path-to-folder>",
     "mode": "selectable",
     "tags": ["<tag1>", "<tag2>"],
     "fetchHint": "<Hint on which file to read first>",
     "source": {
       "type": "<git|archive>",
       "url": "<original-source-url>",
       "ref": "<ref-if-applicable>",
       "subpath": "<matched-subpath>",
       "fetched_at": "<ISO 8601>"
     }
   }
   ```

   **Note on `"type"` field**: Always `"local"` — it describes how the reference is consumed at runtime (content is on local disk, either at the source path or in the cache). The `"source"` object carries provenance for remote origins.

   **Note on `"path"` field**: For local sources, this is an absolute POSIX path. For remote sources, this is the path relative to the repo root (e.g., `modules/mod-code-001-.../`) — stable across machines and sessions. When `{{cache}}` is true, the cached checkout provides the local content; the `"source"` object allows re-fetching if needed.

3. **Derive the reference properties**:
   - **id**: Folder name as-is (e.g., `mod-code-001-circuit-breaker-java-resilience4j`)
   - **name**: Parse folder name into human-readable title. If YAML frontmatter provides a `title`, use it. Otherwise:
     - `mod-code-NNN-xxx` → `MOD-NNN: Xxx (formatted)`
     - `skill-NNN-xxx` → `SKILL-NNN: Xxx (formatted)`
     - `eri-code-NNN-xxx` → `ERI-NNN: Xxx (formatted)`
     - `adr-NNN-xxx` → `ADR-NNN: Xxx (formatted)`
     - Other patterns → convert hyphens to spaces and title-case
   - **description**: If YAML frontmatter has `description`, use it. Otherwise read the main documentation file and extract the first paragraph or summary
   - **type**: Always `"local"`
   - **path**: For local sources: absolute POSIX path (see "Computing paths" below). For remote sources: repo-relative path
   - **mode**: Default `"selectable"`. Use `"auto"` only for foundational references that should always be loaded
   - **tags**: If YAML frontmatter has `tags`, use those first. Then augment with: folder path components (e.g., `modules` → `module`), technology keywords in name (e.g., `java`, `spring`, `resilience4j`), and content-type indicators (e.g., `circuit-breaker`, `persistence`). Deduplicate.
   - **fetchHint**: Main file to read (e.g., `"Read MODULE.md for templates"`, `"Read ERI.md for implementation requirements"`)

4. **Save** as `{{output}}/<id>.json` (or add to single catalog if `{{merge_mode}}` is `"single"`).

---

## Part 2: Validation references

5. **Find validation folders** — folders named `validation` anywhere within the resolved directories.
   - These typically live inside module folders (e.g., `modules/mod-code-001-.../validation/`)
   - Each contains a `README.md` describing checks and one or more shell scripts.

6. **For each validation folder**, create a JSON file:
   ```json
   {
     "id": "<parent-folder-name>-validation",
     "name": "<Parent Name> - Validation",
     "description": "<Brief description from validation README.md>",
     "type": "local",
     "path": "<absolute-or-repo-relative-path-to-validation-folder>",
     "mode": "selectable",
     "tags": ["validation", "<inherited-tags>"],
     "fetchHint": "Read README.md for validation checks and usage"
   }
   ```

   For remote sources, include the `"source"` provenance object as described in Part 1.

7. **Derive the validation reference properties**:
   - **id**: Parent folder name + `-validation` (e.g., `mod-code-001-circuit-breaker-java-resilience4j-validation`)
   - **name**: Parent's human-readable name + ` - Validation` (e.g., `MOD-001: Circuit Breaker - Java/Resilience4j - Validation`)
   - **description**: Extract the Purpose section or first paragraph from the validation `README.md`
   - **path**: For local sources: absolute POSIX path to the `validation/` folder. For remote sources: repo-relative path.
   - **tags**: Always include `"validation"`, plus technology and domain tags inherited from the parent folder name
   - **fetchHint**: Always `"Read README.md for validation checks and usage"`

8. **Save** as `{{output}}/<id>.json` (or add to single catalog if `{{merge_mode}}` is `"single"`).

---

## Part 3: Standalone template index

9. **Find standalone template files** — files with `.tpl` or `.tmpl` extensions anywhere within the resolved directories.

10. **For each template file**, use `listTemplateVariables` to extract syntax and variables:

    Call `listTemplateVariables(template_name=<absolute-path-to-template-file>)`. The tool reads the file, auto-detects the syntax (Jinja2 vs Mustache), and returns the complete, deduplicated variable list using proper parsing (Jinja2 AST analysis or Mustache regex). Use its `syntax` and `variables` output directly for the index entry — do **not** manually parse template variables yourself.

    **Name resolution** (to handle collisions like multiple `Entity.java.tpl`):
    - If filename is unique across all discovered templates → use as-is (e.g., `Application.java.tpl`)
    - If duplicated → prefix with parent folder path relative to the containing module's `templates/` directory (e.g., `domain/Entity.java.tpl`, `application/dto/Response.java.tpl`)

11. **Build the unified index** at `{{templates_index}}`:

    For **local** sources:
    ```json
    {
      "generated_at": "<ISO 8601 timestamp>",
      "template_count": <number>,
      "templates": {
        "<template-name>": {
          "name": "<template-name>",
          "source_path": "/absolute/path/to/template.tpl",
          "syntax": "mustache|jinja2",
          "variables": ["var1", "var2"],
          "origin": "standalone"
        }
      }
    }
    ```

    For **remote** sources, `source_path` is repo-relative and includes a `"source"` provenance object:
    ```json
    {
      "<template-name>": {
        "name": "<template-name>",
        "source_path": "modules/.../templates/domain/Entity.java.tpl",
        "syntax": "mustache",
        "variables": ["Entity", "basePackage", "fields"],
        "origin": "standalone",
        "source": { "type": "git", "url": "...", "ref": "main", "fetched_at": "..." }
      }
    }
    ```

    - **origin**: Always `"standalone"` (as opposed to `"embedded"` which is used at runtime for templates extracted from code blocks)
    - **source_path**: For local sources: absolute POSIX path. For remote sources: repo-relative path.
    - If `{{templates_index}}` already exists, preserve existing entries with `"origin": "embedded"` — merge standalone entries alongside them

---

## Part 4: Subagent profiles

12. **Analyze the knowledge base structure** discovered in Parts 1–3 and generate subagent profiles that match the stages and scopes of the knowledge base. Each profile is a JSON file in `{{profiles_dir}}/`.

    The profile JSON schema:
    ```json
    {
      "name": "<unique-identifier>",
      "description": "<when to use, triggers, scope>",
      "plugins": ["<plugin1>", "<plugin2>"],
      "plugin_configs": {
        "references": {
          "preselected": ["<reference-id-1>", "<reference-id-2>"],
          "exclude_tools": ["selectReferences"]
        }
      },
      "system_instructions": "<custom prompt for the subagent>",
      "max_turns": <integer>,
      "auto_approved": <boolean>,
      "icon_name": "<predefined-icon>"
    }
    ```

    **Required fields**: `name`, `description`

    **Optional fields with defaults**: `plugins` (default `[]` = inherit parent), `plugin_configs` (default `{}`), `system_instructions` (default `null` = inherit parent), `model` (default `null` = inherit parent), `provider` (default `null` = inherit parent), `max_turns` (default `10`), `auto_approved` (default `false`), `icon` (3-line ASCII art array or `null`), `icon_name` (`null`), `gc` (`null`)

    For long-running profiles (max_turns > 15), add GC budget config to prevent context window exhaustion. The `"budget"` GC type removes content in priority order: enrichment → ephemeral → oldest conversation turns → preservable (only under pressure). LOCKED entries are never removed.

    ```json
    "gc": {
      "type": "budget",
      "threshold_percent": 80.0,
      "target_percent": 60.0,
      "pressure_percent": 0,
      "preserve_recent_turns": 5,
      "notify_on_gc": false
    }
    ```

    | Field | Default | Description |
    |-------|---------|-------------|
    | `type` | `"truncate"` | GC strategy. Use `"budget"` for policy-aware removal. |
    | `threshold_percent` | `80.0` | Trigger GC when context usage exceeds this %. |
    | `target_percent` | `60.0` | Target context usage after GC. |
    | `pressure_percent` | `90.0` | When exceeded, PRESERVABLE content may be removed. Set to `0` for **continuous mode** — GC runs every turn above target, PRESERVABLE never touched. |
    | `preserve_recent_turns` | `5` | Number of recent turns always kept. |
    | `notify_on_gc` | `true` | Inject a notification into history after GC. |

    **Guidelines for choosing values:**
    - Skill profiles with many tool calls: use `pressure_percent: 0` (continuous mode) to prevent sudden large evictions
    - Validator profiles: use defaults (`pressure_percent: 90`) — shorter runs rarely hit pressure
    - Set `notify_on_gc: false` for auto_approved profiles to avoid noise

13. **Generate profiles for these categories**:

    ### a) Skill profiles (one per skill/module that represents an actionable flow)
    - **When**: A module or skill folder describes a concrete code-generation or code-modification flow (e.g., "add circuit breaker", "generate microservice")
    - **Name**: Match the knowledge folder id (e.g., `skill-code-001-add-circuit-breaker-java-resilience4j`)
    - **Description**: Start with `[ADD Flow]` or `[GENERATE Flow]` tag, describe when to use, triggers, and scope
    - **Plugins**: `["artifact_tracker", "background", "cli", "filesystem_query", "lsp", "mcp", "memory", "references", "template", "todo"]`
    - **plugin_configs.references.preselected**: Include the ERI, module, and any dependency references needed for the skill. Always include the enablement knowledge base if one exists
    - **plugin_configs.references.exclude_tools**: `["selectReferences"]` (enforce preselected knowledge)
    - **plugin_configs.lsp.config_path**: `"${workspaceRoot}/.lsp.json"`
    - **system_instructions**: Instruct the subagent to read the enablement knowledge base first, then follow the skill specification and apply templates from module dependencies
    - **max_turns**: 15–20 depending on complexity. Add `gc` budget config if > 15 (use `pressure_percent: 0` for continuous mode)
    - **auto_approved**: `false`
    - **icon_name**: Choose from `"circuit_breaker"`, `"microservice"`, `"api"`, `"document"` based on domain

    ### b) Validator profiles (tiered validation matching validation folders)
    - **Tier 1 — Universal**: Basic quality gates for all code (syntax, formatting, secrets, security)
      - **preselected**: Only the enablement knowledge base
      - **max_turns**: 5, **auto_approved**: true
    - **Tier 2 — Technology**: Language/framework-specific checks (e.g., Java/Spring conventions)
      - **preselected**: Enablement + technology-scoped ADRs
      - **max_turns**: 10, **auto_approved**: true
    - **Tier 3 — Pattern compliance**: Verify implementations match skill/module templates
      - **preselected**: Enablement + relevant ADRs + skills + validation references discovered in Part 2
      - **Plugins**: Include `"lsp"` and `"todo"` with config
      - **max_turns**: 15, **auto_approved**: true
    - **Tier 4 — CI/CD**: Build and integration validation
      - **preselected**: Enablement only
      - **max_turns**: 10, **auto_approved**: true
    - All validators use **icon_name**: `"validator"`

    ### c) Analyst profiles (research and documentation)
    - **When**: The knowledge base has a `model/` folder or high-level documentation
    - **Name**: e.g., `analyst-codebase-documentation`
    - **Plugins**: `["cli", "filesystem_query", "memory", "references", "todo", "web_search"]`
    - **max_turns**: 15, **auto_approved**: false, **icon_name**: `"document"`, **gc**: budget config with defaults

    ### d) Investigator profiles (web research)
    - **Name**: e.g., `investigator-web-research`
    - **Plugins**: `["memory", "todo", "web_search", "web_fetch"]`
    - **max_turns**: 10, **auto_approved**: false, **icon_name**: `"search"`

14. **Save each profile** as `{{profiles_dir}}/<name>.json`.

15. **When updating existing profiles**: If a profile file already exists in `{{profiles_dir}}/`, read it first and preserve any manual customizations (e.g., `system_instructions`, `gc` config, `icon`). Only update the `plugin_configs.references.preselected` list and `description` to reflect the current state of the knowledge base. Do not overwrite fields that have been manually tuned.

---

## Common rules

16. **Skip folders that**:
    - Don't contain any documentation, validation, or template files
    - Are hidden (start with `.`)
    - Are named `node_modules`, `__pycache__`, `.git`, etc.
    - Match any pattern in `{{exclude_patterns}}`

17. **Computing paths**:

    **For local sources** — compute absolute filesystem paths, normalize to POSIX forward slashes (`/`), remove trailing slashes:

    Python:
    ```python
    import os
    abs_posix = os.path.abspath(folder_path).replace('\\', '/').rstrip('/')
    ```

    Bash:
    ```bash
    abs=$(realpath "$folder" 2>/dev/null || python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$folder")
    abs_posix=$(echo "$abs" | sed -e 's#\\#/#g' -e 's:/*$::')
    ```

    **For remote sources** — compute repo-relative paths (relative to `repoRoot`), normalize to POSIX forward slashes, remove leading `./` and trailing slashes:

    Python:
    ```python
    import os
    rel = os.path.relpath(folder_path, repo_root).replace('\\', '/').rstrip('/')
    ```

18. **Overwrite protection** (when `{{dry_run}}` is false):
    - If an output file already exists and `{{force}}` is false → skip it, record in summary as `"skipped"` with reason `"exists"`.
    - If `{{force}}` is true → create a timestamped backup in `{{output}}/backups/` before overwriting.

19. **Merge mode**:
    - `"separate"` (default): Write one `<id>.json` file per reference into `{{output}}/`.
    - `"single"`: Accumulate all references into a single `{{output}}/references.json` array.

20. **Report results** — After all artifacts have been written (or planned), produce two outputs:

    **a) Human-readable summary table** (always produced, shown as agent output):

    | # | Type | Artifact file | Source folder | Status |
    |---|------|---------------|---------------|--------|
    | 1 | doc-ref | `{{output}}/mod-code-001-....json` | `modules/mod-code-001-...` | created / updated |
    | 2 | validation-ref | `{{output}}/mod-code-001-...-validation.json` | `modules/mod-code-001-.../validation` | created |
    | 3 | template | `domain/Entity.java.tpl` (in index) | `modules/.../templates/domain/Entity.java.tpl` | indexed |
    | 4 | profile | `{{profiles_dir}}/skill-code-001-....json` | — | created / preserved |
    | ... | | | | |

    **Type** column values: `doc-ref`, `validation-ref`, `template`, `profile`

    **Status** column values:
    - `created` — new file written
    - `updated` — existing file overwritten with changes
    - `preserved` — existing file kept (profile with manual customizations)
    - `indexed` — template added to the index (not copied)
    - `skipped` — folder found but skipped (explain why in a footnote)
    - `planned` — dry run, would be written

    Follow the table with a one-line summary count per type, e.g.:
    > **Totals**: 10 doc-refs, 6 validation-refs, 19 templates indexed, 9 profiles (5 created, 4 preserved)

    **b) Machine-readable summary** (`{{output}}/summary.json` — written unless `{{dry_run}}` is true, in which case it is returned inline):

    ```json
    {
      "source": {
        "type": "<local|git|archive>",
        "url": "<original source>",
        "ref": "<ref or null>",
        "fetched_at": "<ISO 8601 or null>"
      },
      "requested_patterns": ["<subpath patterns or '.' if none>"],
      "matched_directories": ["<resolved dirs>"],
      "generated": ["<filenames written>"],
      "skipped": [{"path": "...", "reason": "..."}],
      "warnings": [{"pattern": "...", "message": "..."}],
      "counts": {
        "doc_refs": 0,
        "validation_refs": 0,
        "templates": 0,
        "profiles_created": 0,
        "profiles_preserved": 0,
        "skipped": 0
      },
      "planned_writes": ["<only present when dry_run=true>"],
      "timestamp": "<ISO 8601>"
    }
    ```

21. **Cleanup** — After all phases complete:
    - If source was remote and `{{cache}}` is false: remove temporary clone/extract directories.
    - If `{{cache}}` is true: keep the cached tree in `.jaato/cache/sources/`.

---

## Examples

### Local source (current behavior, fully backwards-compatible)

```
source="/home/user/project/knowledge"
```

Scans the entire folder recursively, produces absolute paths, no `"source"` object on references.

### Local source with subpath filtering

```
source="/home/user/project/knowledge"
subpaths="modules/*,ERIs/*"
```

Only scans directories matching `modules/*` and `ERIs/*` under the knowledge folder.

### Public GitHub repository

```
source="https://github.com/owner/knowledge-repo"
subpaths="docs/*,knowledge"
ref="main"
cache=true
dry_run=false
```

Downloads the archive of the `main` branch, scans `docs/*` and `knowledge/` directories, caches the checkout, produces repo-relative paths with `"source"` provenance objects.

### Full example folder structure

For a folder structure:
```
knowledge/
├── model/
│   └── ENABLEMENT-MODEL-v3.0.md
├── modules/
│   ├── mod-code-001-circuit-breaker-java-resilience4j/
│   │   ├── MODULE.md
│   │   ├── templates/
│   │   └── validation/
│   │       ├── README.md
│   │       └── circuit-breaker-check.sh
│   └── mod-code-015-hexagonal-base-java-spring/
│       ├── MODULE.md
│       ├── templates/
│       │   ├── Application.java.tpl
│       │   ├── domain/
│       │   │   └── Entity.java.tpl
│       │   └── adapter/
│       │       └── RestController.java.tpl
│       └── validation/
│           └── README.md
├── ERIs/
│   └── eri-code-008-circuit-breaker-java-resilience4j/
│       └── ERI.md
└── ADRs/
    └── adr-004-resilience-patterns/
        └── ADR.md
```

**Generated reference files** in `{{output}}/`:
- `mod-code-001-circuit-breaker-java-resilience4j.json` (documentation)
- `mod-code-001-circuit-breaker-java-resilience4j-validation.json` (validation)
- `mod-code-015-hexagonal-base-java-spring.json` (documentation)
- `mod-code-015-hexagonal-base-java-spring-validation.json` (validation)
- `eri-code-008-circuit-breaker-java-resilience4j.json` (documentation)
- `adr-004-resilience-patterns.json` (documentation)

**Generated template index** at `{{templates_index}}`:
```json
{
  "generated_at": "2026-02-24T12:00:00",
  "template_count": 3,
  "templates": {
    "Application.java.tpl": {
      "name": "Application.java.tpl",
      "source_path": "/.../templates/Application.java.tpl",
      "syntax": "mustache",
      "variables": ["ServiceName", "basePackage", "serviceName"],
      "origin": "standalone"
    },
    "domain/Entity.java.tpl": {
      "name": "domain/Entity.java.tpl",
      "source_path": "/.../templates/domain/Entity.java.tpl",
      "syntax": "mustache",
      "variables": ["Entity", "basePackage", "fields"],
      "origin": "standalone"
    },
    "adapter/RestController.java.tpl": {
      "name": "adapter/RestController.java.tpl",
      "source_path": "/.../templates/adapter/RestController.java.tpl",
      "syntax": "mustache",
      "variables": ["Entity", "basePackage"],
      "origin": "standalone"
    }
  }
}
```

**Generated profiles** in `{{profiles_dir}}/`:
- `skill-code-001-add-circuit-breaker-java-resilience4j.json` (skill, preselects eri-008 + mod-001)
- `validator-tier1-universal.json` (basic quality gates)
- `validator-tier2-java-spring.json` (Java/Spring checks, preselects ADR-004)
- `validator-tier3-pattern-compliance.json` (template compliance, preselects skills + validation refs)
- `validator-tier4-cicd.json` (build validation)

## Begin

Follow the processing strategy strictly:

1. **Phase 0**: Resolve the source — detect type, fetch if remote, resolve subpaths, apply exclusions. If `{{source}}` is local, this phase simply resolves the path and subpaths.
2. **Phase 1**: List the directory tree within the resolved directories (structure only, no file reads). Build the full inventory.
3. **Phase 2**: Process one top-level category at a time (e.g., `ADRs/` first, then `ERIs/`, then `modules/`). Within each category, handle one folder at a time: read entry-point file → write doc-ref JSON → **validate with `validateReference`** → read validation README if present → write validation-ref JSON → **validate with `validateReference`** → read template files if present → accumulate index entries. Move to the next folder only after finishing the current one. If validation fails, fix the JSON and rewrite before proceeding.
4. **Phase 3**: Write the template index JSON → **validate with `validateTemplateIndex`**. Fix and rewrite if validation fails.
5. **Phase 4**: Generate subagent profiles using the collected reference IDs → **validate each with `validateProfile`**. Fix and rewrite any profile that fails validation.
6. **Report**: Produce the final summary table and write `summary.json`.
7. **Cleanup**: Remove temporary directories if applicable.
