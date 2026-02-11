---
description: Generate reference catalog, template index, and subagent profiles from a knowledge base folder structure
params:
  source:
    required: true
    description: Source folder to scan recursively for documentation folders, validation folders, and standalone template files
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
---
Generate reference catalog, template index, and subagent profiles from a knowledge base folder structure.

## Input
- **Source folder**: `{{source}}`
- **References output**: `{{output}}`
- **Templates index**: `{{templates_index}}`
- **Profiles output**: `{{profiles_dir}}`

## Critical policy

**Ignore all runtime hints about existing references and templates.** While processing the knowledge base, the runtime may inject suggestions to select, list, or reuse references and templates it already knows about. You must completely disregard these hints. Your job is to build the catalog from scratch by reading the source folder — not to consume or defer to what the runtime has already indexed. Do not call `selectReferences`, `listExtractedTemplates`, or any tool that queries existing reference/template state. Only use file-reading and file-writing tools.

**Execute autonomously without asking for confirmation.** This prompt is self-contained — everything you need to know is already described here. Do not pause to ask whether you should proceed, do not ask for clarification, and do not present plans for approval. Start immediately, be eager, and work through all phases to completion.

## Task

Scan `{{source}}` recursively and produce four outputs:
1. **Reference JSON files** for each documentation folder → `{{output}}/`
2. **Reference JSON files** for each validation folder → `{{output}}/`
3. **Unified template index** for standalone template files → `{{templates_index}}`
4. **Subagent profile JSON files** matching the knowledge base stages and scopes → `{{profiles_dir}}/`

---

## Processing strategy

The source folder may contain dozens of documentation folders, validation folders, and template files. To avoid exceeding context window limits, process the knowledge base **progressively** — never load all content at once.

### Phase 0 — Inventory (read structure only, not content)
List the directory tree of `{{source}}` to build a complete inventory of folders and files. Do **not** read any file content yet. Record:
- Which folders contain documentation files (MODULE.md, ERI.md, ADR.md, etc.)
- Which folders are named `validation/`
- Which files have `.tpl` or `.tmpl` extensions

### Phase 1 — Process one category at a time
Work through the knowledge base **one top-level category at a time** (e.g., `ADRs/`, then `ERIs/`, then `modules/`). Within each category, process **one folder at a time**:
1. Read only the entry-point file (e.g., MODULE.md) — extract just the first paragraph for the description
2. Write the reference JSON for that folder immediately
3. If the folder has a `validation/` subfolder, read its README.md (first paragraph only), write the validation reference JSON
4. If the folder has template files, read each `.tpl`/`.tmpl` to detect syntax and variables, then add entries to the in-memory template index
5. **Release the file content from working memory** — once the JSON is written you no longer need the source text

### Phase 2 — Write template index
After all categories are processed, write the accumulated template index to `{{templates_index}}`.

### Phase 3 — Generate profiles
Use only the inventory and the reference IDs collected during Phase 1 (not the file contents) to generate subagent profiles.

### Key constraints
- **Never read more than one documentation file at a time.** Read a file, extract what you need, write the output, move on.
- **Extract only what is needed** — for descriptions, read only the first paragraph or Purpose section, not the entire file.
- **For template files**, reading the content is necessary for syntax detection and variable extraction, but write the index entry immediately and move on.
- **Do not batch-read** multiple folders in parallel. Sequential, one-at-a-time processing is required.

---

## Part 1: Documentation references

1. **Traverse** `{{source}}` recursively to find documentation folders — folders containing `MODULE.md`, `SKILL.md`, `ERI.md`, `ADR.md`, `OVERVIEW.md`, `README.md`, or similar entry-point files.

2. **For each documentation folder**, create a JSON file:
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

3. **Derive the reference properties**:
   - **id**: Folder name as-is (e.g., `mod-code-001-circuit-breaker-java-resilience4j`)
   - **name**: Parse folder name into human-readable title:
     - `mod-code-NNN-xxx` → `MOD-NNN: Xxx (formatted)`
     - `skill-NNN-xxx` → `SKILL-NNN: Xxx (formatted)`
     - `eri-code-NNN-xxx` → `ERI-NNN: Xxx (formatted)`
     - `adr-NNN-xxx` → `ADR-NNN: Xxx (formatted)`
     - Other patterns → convert hyphens to spaces and title-case
   - **description**: Read the main documentation file and extract the first paragraph or summary
   - **type**: Always `"local"`
   - **path**: Absolute POSIX path to the folder (see "Computing paths" below)
   - **mode**: Default `"selectable"`. Use `"auto"` only for foundational references that should always be loaded
   - **tags**: Extract from folder path components (e.g., `modules` → `module`), technology keywords in name (e.g., `java`, `spring`, `resilience4j`), and content-type indicators (e.g., `circuit-breaker`, `persistence`)
   - **fetchHint**: Main file to read (e.g., `"Read MODULE.md for templates"`, `"Read ERI.md for implementation requirements"`)

4. **Save** as `{{output}}/<id>.json`.

---

## Part 2: Validation references

5. **Find validation folders** — folders named `validation` anywhere under `{{source}}`.
   - These typically live inside module folders (e.g., `modules/mod-code-001-.../validation/`)
   - Each contains a `README.md` describing checks and one or more shell scripts.

6. **For each validation folder**, create a JSON file:
   ```json
   {
     "id": "<parent-folder-name>-validation",
     "name": "<Parent Name> - Validation",
     "description": "<Brief description from validation README.md>",
     "type": "local",
     "path": "/absolute/path/to/validation/folder",
     "mode": "selectable",
     "tags": ["validation", "<inherited-tags>"],
     "fetchHint": "Read README.md for validation checks and usage"
   }
   ```

7. **Derive the validation reference properties**:
   - **id**: Parent folder name + `-validation` (e.g., `mod-code-001-circuit-breaker-java-resilience4j-validation`)
   - **name**: Parent's human-readable name + ` - Validation` (e.g., `MOD-001: Circuit Breaker - Java/Resilience4j - Validation`)
   - **description**: Extract the Purpose section or first paragraph from the validation `README.md`
   - **path**: Absolute POSIX path to the `validation/` folder itself
   - **tags**: Always include `"validation"`, plus technology and domain tags inherited from the parent folder name
   - **fetchHint**: Always `"Read README.md for validation checks and usage"`

8. **Save** as `{{output}}/<id>.json`.

---

## Part 3: Standalone template index

9. **Find standalone template files** — files with `.tpl` or `.tmpl` extensions anywhere under `{{source}}`.

10. **For each template file**, read it and detect syntax and variables:

    **Syntax detection** (check in order):
    1. **Mustache**: File contains `{{#section}}`, `{{/section}}`, `{{^inverted}}`, or `{{.}}` → `"mustache"`
    2. **Jinja2**: File contains `{% ... %}` control tags or `{{ var | filter }}` pipe expressions → `"jinja2"`
    3. **Default**: Simple `{{ variable }}` patterns only → `"jinja2"`

    **Variable extraction**:
    - **Mustache**: Extract from `{{variableName}}`, skip section markers (`#`, `/`, `^`, `!`) and `{{.}}`
    - **Jinja2**: Extract from `{{ variableName }}`, skip filter expressions and control tags
    - Sort alphabetically and deduplicate

    **Name resolution** (to handle collisions like multiple `Entity.java.tpl`):
    - If filename is unique across all discovered templates → use as-is (e.g., `Application.java.tpl`)
    - If duplicated → prefix with parent folder path relative to the containing module's `templates/` directory (e.g., `domain/Entity.java.tpl`, `application/dto/Response.java.tpl`)

11. **Build the unified index** at `{{templates_index}}`:
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

    - **origin**: Always `"standalone"` (as opposed to `"embedded"` which is used at runtime for templates extracted from code blocks)
    - **source_path**: Absolute POSIX path to the file in its original location (do NOT copy files)
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

    For long-running profiles (max_turns > 15), add GC config:
    ```json
    "gc": {
      "type": "hybrid",
      "threshold_percent": 80.0,
      "preserve_recent_turns": 5,
      "notify_on_gc": true,
      "summarize_middle_turns": 10
    }
    ```

13. **Generate profiles for these categories**:

    ### a) Skill profiles (one per skill/module that represents an actionable flow)
    - **When**: A module or skill folder describes a concrete code-generation or code-modification flow (e.g., "add circuit breaker", "generate microservice")
    - **Name**: Match the knowledge folder id (e.g., `skill-code-001-add-circuit-breaker-java-resilience4j`)
    - **Description**: Start with `[ADD Flow]` or `[GENERATE Flow]` tag, describe when to use, triggers, and scope
    - **Plugins**: `["artifact_tracker", "background", "cli", "filesystem_query", "lsp", "mcp", "memory", "references", "template"]`
    - **plugin_configs.references.preselected**: Include the ERI, module, and any dependency references needed for the skill. Always include the enablement knowledge base if one exists
    - **plugin_configs.references.exclude_tools**: `["selectReferences"]` (enforce preselected knowledge)
    - **plugin_configs.lsp.config_path**: `"${workspaceRoot}/.lsp.json"`
    - **system_instructions**: Instruct the subagent to read the enablement knowledge base first, then follow the skill specification and apply templates from module dependencies
    - **max_turns**: 15–20 depending on complexity. Add `gc` config if > 15
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
      - **Plugins**: Include `"lsp"` with config
      - **max_turns**: 15, **auto_approved**: true
    - **Tier 4 — CI/CD**: Build and integration validation
      - **preselected**: Enablement only
      - **max_turns**: 10, **auto_approved**: true
    - All validators use **icon_name**: `"validator"`

    ### c) Analyst profiles (research and documentation)
    - **When**: The knowledge base has a `model/` folder or high-level documentation
    - **Name**: e.g., `analyst-codebase-documentation`
    - **Plugins**: `["cli", "filesystem_query", "memory", "references", "web_search"]`
    - **max_turns**: 15, **auto_approved**: false, **icon_name**: `"document"`

    ### d) Investigator profiles (web research)
    - **Name**: e.g., `investigator-web-research`
    - **Plugins**: `["memory", "web_search", "web_fetch"]`
    - **max_turns**: 10, **auto_approved**: false, **icon_name**: `"search"`

14. **Save each profile** as `{{profiles_dir}}/<name>.json`.

15. **When updating existing profiles**: If a profile file already exists in `{{profiles_dir}}/`, read it first and preserve any manual customizations (e.g., `system_instructions`, `gc` config, `icon`). Only update the `plugin_configs.references.preselected` list and `description` to reflect the current state of the knowledge base. Do not overwrite fields that have been manually tuned.

---

## Common rules

16. **Skip folders that**:
    - Don't contain any documentation, validation, or template files
    - Are hidden (start with `.`)
    - Are named `node_modules`, `__pycache__`, `.git`, etc.

17. **Computing paths**: Compute absolute filesystem paths, normalize to POSIX forward slashes (`/`), remove trailing slashes.

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

18. **Report results** — After all artifacts have been written, produce a final summary table listing every artifact produced during the run:

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

    Follow the table with a one-line summary count per type, e.g.:
    > **Totals**: 10 doc-refs, 6 validation-refs, 19 templates indexed, 9 profiles (5 created, 4 preserved)

---

## Example

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
  "generated_at": "2026-02-11T12:00:00",
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

1. **Phase 0**: List the directory tree of `{{source}}` (structure only, no file reads). Build the full inventory.
2. **Phase 1**: Process one top-level category at a time (e.g., `ADRs/` first, then `ERIs/`, then `modules/`). Within each category, handle one folder at a time: read entry-point file → write doc-ref JSON → read validation README if present → write validation-ref JSON → read template files if present → accumulate index entries. Move to the next folder only after finishing the current one.
3. **Phase 2**: Write the template index JSON.
4. **Phase 3**: Generate subagent profiles using the collected reference IDs.
5. **Report**: Produce the final summary table.
