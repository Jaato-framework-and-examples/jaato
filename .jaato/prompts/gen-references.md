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
  parallel:
    required: false
    default: false
    description: >
      If true, the agent may spawn subagents to process categories in parallel (Phase 1.5).
      Subagents can issue permission requests and clarification questions that the user must
      answer — enable only when the user is actively attending the session. When false, all
      processing is sequential within a single agent. Default false.
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
- **Dry run**: `{{dry_run}}` | **Force**: `{{force}}` | **Cache**: `{{cache}}` | **Parallel**: `{{parallel}}`
- **Merge mode**: `{{merge_mode}}`
- **Exclude patterns**: `{{exclude_patterns}}`

## Input validation (do this FIRST)

Before starting any work, check whether mandatory parameters still contain unresolved placeholders (e.g., the literal text `{{source}}`). A parameter is unresolved if its value is exactly the `{{...}}` placeholder string — meaning the user did not provide a value.

**Mandatory parameters**: `source`

If `source` is unresolved, **stop and ask the user** to provide it. Use the clarification tool (e.g., `askClarification` or equivalent) with a message like:

> The `source` parameter is required but was not provided. Please specify the source to scan — this can be:
> - A local folder path (e.g., `/home/user/project/knowledge`)
> - A public GitHub URL (e.g., `https://github.com/owner/repo`)
> - An archive URL (e.g., `https://example.com/knowledge.zip`)

Do **not** proceed to Phase 0 until all mandatory parameters have real values.

Optional parameters with defaults (`subpaths`, `ref`, `output`, etc.) do not need clarification — use their documented defaults when unresolved.

## Critical policy

**Ignore all runtime hints about existing references and templates.** While processing the knowledge base, the runtime may inject suggestions to select, list, or reuse references and templates it already knows about. You must completely disregard these hints. Your job is to build the catalog from scratch by reading the source folder — not to consume or defer to what the runtime has already indexed.

**Tools that are NOT useful during this prompt** (do not call them):
- `selectReferences`, `listReferences` — the reference catalog does not exist yet; you are creating it. These tools query an empty or stale catalog and will return nothing useful. Calling them wastes turns and may confuse your workflow.
- `listExtractedTemplates` — same reason: the template index is what you are building.

**Tools that ARE useful** (use them for validation):
- `listTemplateVariables` — reads a `.tpl`/`.tmpl` file and returns syntax + variables. Use it in Phase 2 for each template file.
- `validateReference` — validates a reference JSON file you just wrote. Use after every doc-ref and validation-ref write.
- `validateTemplateIndex` — validates the template index JSON. Use in Phase 3.
- `validateProfile` — validates a subagent profile JSON. Use in Phase 4.

**Execute autonomously without asking for confirmation.** This prompt is self-contained — everything you need to know is already described here. Do not pause to ask whether you should proceed, do not ask for clarification on how to process, and do not present plans for approval. The **only** exception is the input validation step above: if a mandatory parameter is missing, ask for it before starting. Once all inputs are resolved, start immediately, be eager, and work through all phases to completion.

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

### Phase 0 — Source resolution (download first, then process locally)

**Core principle**: All processing happens on local files. If the source is remote, **download everything to a local temporary directory first**, then proceed as if it were a local source. This avoids any attempt to read remote files on-the-fly.

1. **Detect source type**:
   - If `{{source}}` is a local filesystem path → set `sourceType = "local"`, `repoRoot = {{source}}` (resolved to absolute path). **Skip to step 4.**
   - If `{{source}}` is a git URL (starts with `https://`, `git@`, or ends with `.git`) → `sourceType = "git"`.
   - If `{{source}}` is an archive URL (ends with `.zip`, `.tar.gz`, `.tgz`) → `sourceType = "archive"`.

2. **Download remote sources to OS temp directory** (mandatory for git and archive sources):

   **You must complete this step before doing anything else.** Do not attempt to list, read, or process any remote files without downloading them first.

   **Which tools to use for downloading:**
   - Use your **shell/CLI tool** (e.g., `call_service` with the `cli` plugin, or the bash/shell tool) to run `curl`, `git clone`, `unzip`, and other shell commands.
   - Alternatively, if you have a **`web_fetch`** tool available, you can use it to download the zip archive URL directly.
   - Do **not** try to use `Read`, `Glob`, or file-reading tools on a URL — those only work on local paths. The source must be downloaded to disk first.

   a) Create a temporary directory using the OS temp location:
      ```bash
      WORK_DIR=$(mktemp -d /tmp/jaato-source-XXXXXX)
      ```

   b) Download the content:
      - **GitHub URL** (contains `github.com`):
        - Build the archive URL: `https://github.com/<owner>/<repo>/archive/<ref>.zip` (use `{{ref}}` if provided, otherwise `HEAD`).
        - Download with `curl -fsSL -o "$WORK_DIR/source.zip" "<archive_url>"`.
        - If curl fails (404, 403, etc.), fall back to shallow clone: `git clone --depth 1 --branch <ref> <url> "$WORK_DIR/repo"`.
        - If both fail, stop with: *"Could not download repository. Check that the URL is correct and the repo is public."*
      - **Other git URL**: `git clone --depth 1 --branch <ref> <url> "$WORK_DIR/repo"`.
      - **Archive URL** (.zip/.tar.gz/.tgz): `curl -fsSL -o "$WORK_DIR/source.archive" "<url>"`.

   c) Extract if downloaded as archive:
      ```bash
      cd "$WORK_DIR"
      unzip -q source.zip       # for .zip
      # or: tar xzf source.archive  # for .tar.gz/.tgz
      ```
      GitHub zip archives extract to a folder named `<repo>-<ref>/`. Identify this folder:
      ```bash
      EXTRACTED=$(ls -d "$WORK_DIR"/*/  | head -1)
      ```

   d) Set `repoRoot = "$EXTRACTED"` (the extracted directory, or `"$WORK_DIR/repo"` if cloned).

   e) **Verify the download succeeded**: List the top-level contents of `repoRoot` to confirm files are present. If empty or missing, fail immediately.

   f) Record `sourceMetadata = { "type": "<sourceType>", "url": "<original-url>", "ref": "<ref>", "fetched_at": "<ISO 8601>" }`.

3. **Cache management** (when `{{cache}}` is true):
   - Cache location: `.jaato/cache/sources/` within the project.
   - Cache key: hash of `(url, ref)`. If cache hit and not stale, use cached tree as `repoRoot` and skip download.
   - If `{{cache}}` is false, the temp directory from step 2 will be cleaned up at the end.

4. **Resolve subpaths/patterns** (operates on local `repoRoot` — always local at this point):
   - If `{{subpaths}}` is empty → treat as `"."` (scan entire `repoRoot`).
   - Otherwise split on commas, trim whitespace, ignore empty items. Each item may be:
     - An exact path (e.g., `"knowledge"`, `"modules/mod-foo"`) — match that directory if it exists.
     - A glob pattern with `*` or `?` (e.g., `"knowledge/*"`, `"model/domains/*-code-*"`) — expand against the repo tree.
     - A path ending with `/` — treated as a directory match.
   - If a pattern matches files, use their parent directories (deduplicate).
   - Deduplicate and sort all matched directories deterministically.
   - If a pattern matches nothing, record a warning in the summary and continue.

5. **Apply exclusions**: Filter out directories matching `{{exclude_patterns}}` and built-in exclusions (`node_modules`, `.git`, `__pycache__`, hidden dirs).

After Phase 0, `repoRoot` always points to a local directory on disk. All subsequent phases operate on local files under `repoRoot` using standard file-reading tools.

### Phase 1 — Inventory (read structure only, not content)
List the directory tree within the matched directories to build a complete inventory of folders and files. Do **not** read any file content yet. Record:
- Which folders contain documentation entry-point files (see "Recognized entry-point files" below)
- Which folders are named `validation/` **and are subfolders of a documentation folder** (post-implementation validation scripts). A top-level or standalone `validation/` directory matched by a subpath pattern is a documentation folder, not a validation subfolder.
- Which folders are named `templates/` (standalone template files)
- Which folders are named `policies/` (implementation constraint documents)
- Which folders are named `scripts/` (helper scripts for implementation)
- Which files have `.tpl` or `.tmpl` extensions

### Phase 1.5 — Evaluate parallelization opportunity

**This phase is only available when `{{parallel}}` is true.** If `{{parallel}}` is false, skip directly to Phase 2 (sequential processing).

After building the inventory, assess whether the work can be split across **parallel subagents** to finish faster.

**Decision criteria — parallelize when ALL of the following are true:**
- `{{parallel}}` is `true`
- The inventory contains **3 or more top-level categories** (e.g., `ADRs/`, `ERIs/`, `modules/`, `skills/`), OR the total number of documentation folders exceeds **10**
- `{{dry_run}}` is `false` (dry runs are fast enough sequentially)

**Do NOT parallelize when ANY of the following are true:**
- `{{parallel}}` is `false` (user has not opted in)
- There are fewer than 3 categories and fewer than 10 folders total (overhead outweighs the benefit)
- `{{dry_run}}` is `true`

#### Subagent interaction warning

Subagents run as independent agents that may need user interaction:
- **Permission requests**: Subagents executing file writes or shell commands will prompt the user for approval. The user must be actively attending the session to respond, or subagents will block indefinitely.
- **Clarification questions**: If a subagent encounters ambiguous content (e.g., cannot determine the entry-point file), it may ask the user for guidance.
- **Multiple simultaneous prompts**: With N subagents running in parallel, the user may receive multiple permission/clarification prompts at the same time.

Before spawning subagents, **inform the user** that subagents will run in parallel and may require their attention for permission approvals or clarifications. For example:

> Spawning N subagents to process categories in parallel. Subagents may prompt you for file-write permissions — please keep the session attended.

**How to split:**
1. Partition the matched directories into **groups** — typically one group per top-level category (e.g., all `ADRs/*` folders in one group, all `modules/*` in another). If one category is much larger than the rest, split it further (e.g., `modules/mod-001..mod-010` and `modules/mod-011..mod-020`).
2. Spawn one **subagent per group**. Each subagent receives:
   - The `repoRoot` path (local, read-only — all subagents share the same downloaded directory)
   - Its assigned list of directories to process
   - The `sourceType` and `sourceMetadata` (for provenance on remote sources)
   - The output directory `{{output}}/` (no conflicts — each folder produces a unique file ID)
   - The same extraction and validation rules from Phase 2 (Parts 1–2 of this prompt)
3. Each subagent executes Phase 2 for its group: reads entry-point files, writes doc-ref and validation-ref JSON files, validates each with `validateReference`, and collects template index entries.
4. Each subagent returns to the coordinator:
   - The list of reference IDs it generated
   - The template index entries it collected (template name → metadata)
   - Any warnings or skipped folders
5. The **coordinator** (this agent) waits for all subagents to complete, then:
   - Merges template index entries from all subagents → proceeds to Phase 3
   - Merges reference ID lists from all subagents → proceeds to Phase 4
   - Merges warnings and skipped lists → proceeds to the final report

**Subagent instructions template** — when spawning each subagent, you **must** include the full artifact schemas inline. Subagents do not have access to this prompt — they only see the instructions you give them. If you omit the schemas, subagents will invent their own field names and the output will fail validation.

Provide instructions equivalent to the following (substitute the `<placeholders>` but keep all schemas verbatim):

> Process the following directories under `repoRoot` = `<path>`:
> - `<dir1>`, `<dir2>`, ...
>
> sourceType = `<local|git|archive>`, sourceMetadata = `<metadata dict if remote>`, knowledgeDir = `<.jaato/knowledge/hash>`
>
> ---
>
> **Entry-point priority** (use the first match):
> MODULE.md > SKILL.md > ERI.md > ADR.md > DOMAIN.md > CAPABILITY.md > FLOW.md > OVERVIEW.md > README.md
>
> ---
>
> **For each directory**, produce a documentation reference JSON file at `<output>/<id>.json`:
>
> ```json
> {
>   "id": "<folder-name>",
>   "name": "<Human Readable Name>",
>   "description": "<First paragraph from entry-point file>",
>   "type": "local",
>   "path": "/absolute/path/to/folder",
>   "mode": "selectable",
>   "tags": ["<tag1>", "<tag2>"],
>   "fetchHint": "<Hint on which file to read first>",
>   "contents": {
>     "templates": "templates/",
>     "validation": "validation/",
>     "policies": null,
>     "scripts": null
>   }
> }
> ```
>
> For remote sources, add a `"source"` object:
> ```json
> "source": {
>   "type": "<git|archive>",
>   "url": "<original-source-url>",
>   "ref": "<ref-if-applicable>",
>   "subpath": "<matched-subpath>",
>   "fetched_at": "<ISO 8601>"
> }
> ```
>
> Property rules:
> - **id**: Folder name as-is (e.g., `mod-code-001-circuit-breaker-java-resilience4j`)
> - **name**: Parse folder name → human-readable. `mod-code-NNN-xxx` → `MOD-NNN: Xxx`, `eri-code-NNN-xxx` → `ERI-NNN: Xxx`, `adr-NNN-xxx` → `ADR-NNN: Xxx`, `skill-NNN-xxx` → `SKILL-NNN: Xxx`. Others: hyphens to spaces, title-case. Use YAML frontmatter `title` if available.
> - **description**: YAML frontmatter `description` if available, otherwise first paragraph from entry-point file.
> - **type**: Always `"local"`.
> - **path**: Absolute POSIX path. For remote sources: point to materialized copy under `.jaato/knowledge/`.
> - **mode**: Default `"selectable"`.
> - **tags**: YAML frontmatter `tags` first, then augment with folder path components, technology keywords, content-type indicators. Deduplicate.
> - **fetchHint**: e.g., `"Read MODULE.md for templates"`, `"Read ERI.md for implementation requirements"`.
> - **contents** (required, exact field name, must be an object with exactly four keys):
>   - `"templates"`: relative path (e.g., `"templates/"`) if present, else `null`
>   - `"validation"`: relative path (e.g., `"validation/"`) if present, else `null`
>   - `"policies"`: relative path (e.g., `"policies/"`) if present, else `null`
>   - `"scripts"`: relative path (e.g., `"scripts/"`) if present, else `null`
>
> ---
>
> **Validation references** — if a folder has a `validation/` subfolder, also write `<output>/<parent-id>-validation.json`:
>
> ```json
> {
>   "id": "<parent-folder-name>-validation",
>   "name": "<Parent Name> - Validation",
>   "description": "<First paragraph from validation/README.md>",
>   "type": "local",
>   "path": "/absolute/path/to/validation-folder",
>   "mode": "selectable",
>   "tags": ["validation", "<inherited-tags>"],
>   "fetchHint": "Read README.md for validation checks and usage"
> }
> ```
>
> For remote sources, include the `"source"` provenance object as above.
>
> ---
>
> **Template index entries** — if the folder contains `.tpl`/`.tmpl` files, call `listTemplateVariables(template_name=<absolute-path>)` for each and collect entries in this shape:
>
> ```json
> {
>   "<reference-id>/<relative-path-from-templates-dir>": {
>     "name": "<reference-id>/<relative-path-from-templates-dir>",
>     "source_path": "/absolute/path/to/template.tpl",
>     "syntax": "mustache|jinja2",
>     "variables": ["var1", "var2"],
>     "origin": "standalone"
>   }
> }
> ```
>
> Template name = `<reference-id>/<relative-path-from-templates-dir>` (e.g., `mod-code-015-hexagonal-base-java-spring/domain/Entity.java.tpl`). For remote sources, add `"source": { "type": "...", "url": "...", "ref": "...", "fetched_at": "..." }` to each entry.
>
> ---
>
> If the source is remote, **copy the folder to `<knowledge_dir>/<repo-relative-path>/` before writing the reference** — the reference `path` must point to this materialized copy. Materialize validation subfolders too.
>
> Validate every reference JSON with `validateReference`. Fix and rewrite if validation fails.
>
> Return: `{ "reference_ids": [...], "template_entries": {...}, "warnings": [...], "skipped": [...] }`

If parallelization is not warranted (even with `{{parallel}}` true), skip this step and proceed with sequential Phase 2.

### Phase 2 — Process one category at a time
Work through the knowledge base **one top-level category at a time** (e.g., `ADRs/`, then `ERIs/`, then `modules/`, then `model/domains/`, then `model/standards/`). Within each category, process **one folder at a time**:
1. Identify the entry-point file using the priority order from "Recognized entry-point files" above. Read only that file — extract just the first paragraph for the description
2. If the entry-point file has YAML frontmatter with `title`, `description`, or `tags`, use those values. Otherwise fall back to the extraction rules (first paragraph, folder name parsing)
3. **If source is remote**: Copy the folder from the temp download to a stable workspace location (see "Materializing remote content" below) **before** writing the reference JSON. The reference `path` must point to this permanent copy, not to the temp directory.
4. **Build the `"contents"` object** — check if the **current documentation folder** contains any of these **immediate child** directories:
   - `templates/` — contains `.tpl`/`.tmpl` files (authoritative standalone templates)
   - `validation/` — contains shell scripts for post-implementation checks
   - `policies/` — contains markdown documents with implementation constraints
   - `scripts/` — contains helper scripts for use during implementation

   For each, set the corresponding key in the `"contents"` object to the relative path (e.g., `"templates/"`) if present, or `null` if absent. **Always include all four keys.** The result must be an object — not an array, not a list of names. Example: `{"templates": "templates/", "validation": null, "policies": null, "scripts": null}`.

   **Important**: This detection applies only to **immediate children** of the documentation folder being processed. A directory named `validation/` that was itself matched by a subpath pattern (e.g., `model/standards/validation/` matched by `model/standards/*`) is a **documentation folder in its own right**, not a validation subfolder. Only treat `validation/` as a typed subfolder when it appears **inside** another documentation folder (e.g., `modules/mod-code-001-.../validation/`).
5. Write the reference JSON for that folder immediately. The JSON **must** include the `"contents"` property exactly as built in step 4 (an object with four keys, not an array or renamed field like `"subfolders"`)
6. If the folder has a `validation/` subfolder, read its README.md (first paragraph only), copy the validation folder to the workspace location if remote, write the validation reference JSON
7. If the folder has template files, read each `.tpl`/`.tmpl` to detect syntax and variables, then add entries to the in-memory template index
8. **Release the file content from working memory** — once the JSON is written you no longer need the source text

#### Materializing remote content

When the source is remote (git or archive), the downloaded content lives in a temporary directory that will be deleted after this prompt finishes. References with `"type": "local"` require their `path` to point to **permanent, readable files on disk**. Therefore, you must copy each referenced folder to a stable location within the workspace.

**Target location**: `.jaato/knowledge/<source-hash>/` where `<source-hash>` is a short deterministic identifier derived from the source URL (e.g., first 8 chars of the SHA-256 of the URL). This creates a structure like:
```
.jaato/knowledge/a1b2c3d4/
├── modules/
│   ├── mod-code-001-circuit-breaker-java-resilience4j/
│   │   ├── MODULE.md
│   │   ├── templates/
│   │   └── validation/
│   └── mod-code-015-hexagonal-base-java-spring/
├── ERIs/
│   └── eri-code-008-.../
└── ADRs/
    └── adr-004-.../
```

**How to copy**: For each documentation folder being processed, copy it (preserving directory structure relative to `repoRoot`) from the temp download to the knowledge directory:
```bash
KNOWLEDGE_DIR=".jaato/knowledge/<source-hash>"
mkdir -p "$KNOWLEDGE_DIR/$(dirname '<repo-relative-path>')"
cp -r "$repoRoot/<repo-relative-path>" "$KNOWLEDGE_DIR/<repo-relative-path>"
```

**Reference `path`**: Set to the **absolute path** of the materialized copy (e.g., `/home/user/project/.jaato/knowledge/a1b2c3d4/modules/mod-code-001-.../`). This ensures the reference works at runtime regardless of whether the temp download still exists.

**When subagents are used (Phase 1.5)**: Each subagent must also materialize its folders before writing references. Pass the `KNOWLEDGE_DIR` path to each subagent so all copies land in the same tree.

### Phase 3 — Write template index
After all categories are processed (sequentially or via subagents), merge all collected template index entries and write the unified template index to `{{templates_index}}`. If subagents were used, combine the `template_entries` dicts returned by each subagent before writing.

### Phase 4 — Generate profiles
Use only the inventory and the full list of reference IDs collected during Phase 2 — either directly or merged from subagent results — to generate subagent profiles. Do not use file contents.

### Key constraints
- **Never read more than one documentation file at a time** (per agent). Read a file, extract what you need, write the output, move on.
- **Extract only what is needed** — for descriptions, read only the first paragraph or Purpose section, not the entire file.
- **For template files**, reading the content is necessary for syntax detection and variable extraction, but write the index entry immediately and move on.
- **Within a single agent, process folders sequentially** — one at a time. Parallelism is achieved by splitting work across **subagents** (Phase 1.5), not by reading multiple files in one agent.

---

## Recognized entry-point files

A folder is a "documentation folder" if it contains at least one of these entry-point files. When a folder contains **multiple** entry-point files, use the **first match** from this priority-ordered list:

1. `MODULE.md`
2. `SKILL.md`
3. `ERI.md`
4. `ADR.md`
5. `DOMAIN.md`
6. `CAPABILITY.md`
7. `FLOW.md`
8. `OVERVIEW.md`
9. `README.md`

`README.md` has the **lowest** priority because it often serves as an index or container description rather than the primary documentation. When a folder contains both `README.md` and a more specific entry-point (e.g., `DOMAIN.md`), always prefer the specific one.

**This list is exhaustive.** If a folder contains only `.md` files not in this list (e.g., `TAG-TAXONOMY.md`, `ASSET-STANDARDS-v1.4.md`), it is still a valid documentation folder if it has a `README.md`. Standalone `.md` files that are not entry-points are treated as supplementary content within the folder.

---

## Part 1: Documentation references

1. **Traverse** the resolved directories (from Phase 0) to find documentation folders — folders containing any of the recognized entry-point files listed above.

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
     "fetchHint": "<Hint on which file to read first>",
     "contents": {
       "templates": "templates/",
       "validation": "validation/",
       "policies": null,
       "scripts": null
     }
   }
   ```

   For **remote** sources (git or archive):
   ```json
   {
     "id": "<folder-name>",
     "name": "<Human Readable Name>",
     "description": "<Brief description extracted from the documentation>",
     "type": "local",
     "path": "/absolute/path/to/.jaato/knowledge/<source-hash>/<repo-relative-path>",
     "mode": "selectable",
     "tags": ["<tag1>", "<tag2>"],
     "fetchHint": "<Hint on which file to read first>",
     "contents": {
       "templates": "templates/",
       "validation": "validation/",
       "policies": null,
       "scripts": null
     },
     "source": {
       "type": "<git|archive>",
       "url": "<original-source-url>",
       "ref": "<ref-if-applicable>",
       "subpath": "<matched-subpath>",
       "fetched_at": "<ISO 8601>"
     }
   }
   ```

   **Note on `"type"` field**: Always `"local"` — it describes how the reference is consumed at runtime (content is on local disk). For remote sources, the content has been materialized (copied) into the workspace. The `"source"` object carries provenance for re-fetching if needed.

   **Note on `"path"` field**: Always an **absolute POSIX path** pointing to readable content on disk. For local sources, this points directly to the original folder. For remote sources, this points to the materialized copy under `.jaato/knowledge/<source-hash>/` (see "Materializing remote content" in Phase 2). The `"source"` object records the original URL for provenance and re-fetching.

3. **Derive the reference properties**:
   - **id**: Folder name as-is (e.g., `mod-code-001-circuit-breaker-java-resilience4j`)
   - **name**: Parse folder name into human-readable title. If YAML frontmatter provides a `title`, use it. Otherwise:
     - `mod-code-NNN-xxx` → `MOD-NNN: Xxx (formatted)`
     - `skill-NNN-xxx` → `SKILL-NNN: Xxx (formatted)`
     - `eri-code-NNN-xxx` → `ERI-NNN: Xxx (formatted)`
     - `adr-NNN-xxx` → `ADR-NNN: Xxx (formatted)`
     - Other patterns (e.g., `code`, `authoring`, `traceability`) → convert hyphens to spaces and title-case
   - **description**: If YAML frontmatter has `description`, use it. Otherwise read the main documentation file and extract the first paragraph or summary
   - **type**: Always `"local"`
   - **path**: Always an absolute POSIX path (see "Computing paths" below). For local sources: points to the original folder. For remote sources: points to the materialized copy under `.jaato/knowledge/`
   - **mode**: Default `"selectable"`. Use `"auto"` only for foundational references that should always be loaded
   - **tags**: If YAML frontmatter has `tags`, use those first. Then augment with: folder path components (e.g., `modules` → `module`), technology keywords in name (e.g., `java`, `spring`, `resilience4j`), and content-type indicators (e.g., `circuit-breaker`, `persistence`). Deduplicate.
   - **fetchHint**: Main file to read (e.g., `"Read MODULE.md for templates"`, `"Read ERI.md for implementation requirements"`, `"Read DOMAIN.md for domain definition"`)
   - **contents** (**required**, exact field name `"contents"`, must be an object — never an array or renamed field):
     An object declaring which typed subfolders exist in this reference directory. **Always include all four keys** — set each to the relative path string if the subfolder exists, or `null` if absent:
     - `"templates"`: e.g., `"templates/"` if the folder contains a `templates/` subfolder with `.tpl`/`.tmpl` files. These are authoritative templates the model must use via `renderTemplateToFile` — the runtime suppresses extraction of embedded templates from documentation when this is set.
     - `"validation"`: e.g., `"validation/"` if the folder contains a `validation/` subfolder with shell scripts that must be run as post-implementation checks.
     - `"policies"`: e.g., `"policies/"` if the folder contains a `policies/` subfolder with markdown documents defining implementation constraints.
     - `"scripts"`: e.g., `"scripts/"` if the folder contains a `scripts/` subfolder with helper scripts the model can use during implementation.

     Correct: `"contents": {"templates": "templates/", "validation": "validation/", "policies": null, "scripts": null}`
     Wrong:   `"subfolders": ["templates", "validation"]` — wrong field name, wrong type (array instead of object), missing null keys

4. **Save** as `{{output}}/<id>.json` (or add to single catalog if `{{merge_mode}}` is `"single"`).

---

## Part 2: Validation references

5. **Find validation subfolders** — folders named `validation` that are **immediate children of a documentation folder** processed in Part 1.
   - These typically live inside module folders (e.g., `modules/mod-code-001-.../validation/`)
   - Each contains a `README.md` describing checks and one or more shell scripts.
   - **Do NOT treat a directory as a validation subfolder if it was matched directly by a subpath pattern** (e.g., `model/standards/validation/` matched by `model/standards/*`). Such directories are documentation folders and were already processed in Part 1.

6. **For each validation folder**, create a JSON file:
   ```json
   {
     "id": "<parent-folder-name>-validation",
     "name": "<Parent Name> - Validation",
     "description": "<Brief description from validation README.md>",
     "type": "local",
     "path": "/absolute/path/to/validation-folder",
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
   - **path**: Always an absolute POSIX path to the `validation/` folder. For remote sources: points to the materialized copy under `.jaato/knowledge/`.
   - **tags**: Always include `"validation"`, plus technology and domain tags inherited from the parent folder name
   - **fetchHint**: Always `"Read README.md for validation checks and usage"`

8. **Save** as `{{output}}/<id>.json` (or add to single catalog if `{{merge_mode}}` is `"single"`).

---

## Part 3: Standalone template index

9. **Find standalone template files** — files with `.tpl` or `.tmpl` extensions anywhere within the resolved directories.

10. **For each template file**, use `listTemplateVariables` to extract syntax and variables:

    Call `listTemplateVariables(template_name=<absolute-path-to-template-file>)`. The tool reads the file, auto-detects the syntax (Jinja2 vs Mustache), and returns the complete, deduplicated variable list using proper parsing (Jinja2 AST analysis or Mustache regex). Use its `syntax` and `variables` output directly for the index entry — do **not** manually parse template variables yourself.

    **Name resolution** — always namespace by the owning reference ID to avoid cross-module collisions:
    - Template name = `<reference-id>/<relative-path-from-templates-dir>` (e.g., `mod-code-015-hexagonal-base-java-spring/domain/Entity.java.tpl`)
    - The `<reference-id>` is the `id` of the documentation folder that contains the `templates/` directory
    - The `<relative-path-from-templates-dir>` is the template file's path relative to that `templates/` directory (e.g., `domain/Entity.java.tpl`, `Application.java.tpl`)
    - This guarantees uniqueness even when multiple modules have identically-named templates with different content (e.g., `mod-code-001-.../templates/config/Config.java.tpl` vs `mod-code-015-.../templates/config/Config.java.tpl`)

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

    For **remote** sources, `source_path` points to the materialized copy and includes a `"source"` provenance object:
    ```json
    {
      "mod-code-015-hexagonal-base-java-spring/domain/Entity.java.tpl": {
        "name": "mod-code-015-hexagonal-base-java-spring/domain/Entity.java.tpl",
        "source_path": "/absolute/path/to/.jaato/knowledge/<hash>/modules/mod-code-015-.../templates/domain/Entity.java.tpl",
        "syntax": "mustache",
        "variables": ["Entity", "basePackage", "fields"],
        "origin": "standalone",
        "source": { "type": "git", "url": "...", "ref": "main", "fetched_at": "..." }
      }
    }
    ```

    - **origin**: Always `"standalone"` (as opposed to `"embedded"` which is used at runtime for templates extracted from code blocks)
    - **source_path**: Always an absolute POSIX path. For remote sources, this points to the materialized copy under `.jaato/knowledge/` (the content was copied there during Phase 2).
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

13. **Subagent interaction model** — When a subagent with `auto_approved: false` executes a tool that requires permission, or calls `request_clarification`, the request is **forwarded to the parent agent** as an XML injection (`<permission_request>` or `<clarification_request>`). The subagent **blocks** until the parent responds. If the parent agent doesn't know this will happen, it will appear to hang — the parent waits for the subagent to finish, the subagent waits for the parent to respond, deadlock.

    To prevent this, generated profiles must address both sides of the interaction:

    **In the profile's `description`** — include an interaction hint so the orchestrating agent knows what to expect:
    - For `auto_approved: false` profiles: append `"⚠ This subagent may request permissions or clarifications — monitor and respond promptly to avoid blocking."`
    - For `auto_approved: true` profiles: no hint needed (permissions are auto-granted)

    **In the profile's `system_instructions`** — include guidance for the subagent about how its requests are handled:
    - For `auto_approved: false`: `"When you need permission to execute a tool or need clarification from the user, your request will be forwarded to the orchestrating agent. Proceed with other work if possible while waiting, but do not assume the request was denied if there is a delay."`
    - For `auto_approved: true`: `"Tool permissions are auto-approved for this profile. If a tool fails, diagnose the error rather than requesting manual intervention."`

    **In validator profiles** that may need to download/install tools (e.g., linters, formatters): include `web_fetch` in the plugin list so the subagent can fetch installers or tool binaries without blocking on the parent for a manual download step.

14. **Generate profiles for these categories**:

    ### a) Skill profiles (one per skill/module that represents an actionable flow)
    - **When**: A module or skill folder describes a concrete code-generation or code-modification flow (e.g., "add circuit breaker", "generate microservice")
    - **Name**: Match the knowledge folder id (e.g., `skill-code-001-add-circuit-breaker-java-resilience4j`)
    - **Description**: Start with `[ADD Flow]` or `[GENERATE Flow]` tag, describe when to use, triggers, and scope. End with interaction hint (see item 13).
    - **Plugins**: `["artifact_tracker", "background", "cli", "environment", "file_edit", "filesystem_query", "lsp", "mcp", "memory", "references", "template", "todo", "waypoint"]`
    - **plugin_configs.references.preselected**: Include the ERI, module, and any dependency references needed for the skill. Always include the enablement knowledge base if one exists
    - **plugin_configs.references.exclude_tools**: `["selectReferences"]` (enforce preselected knowledge)
    - **plugin_configs.lsp.config_path**: `"${workspaceRoot}/.lsp.json"`
    - **system_instructions**: Instruct the subagent to read the enablement knowledge base first, then follow the skill specification and apply templates from module dependencies. Include the `auto_approved: false` interaction guidance from item 13.
    - **max_turns**: 15–20 depending on complexity. Add `gc` budget config if > 15 (use `pressure_percent: 0` for continuous mode)
    - **auto_approved**: `false`
    - **icon_name**: Choose from `"circuit_breaker"`, `"microservice"`, `"api"`, `"document"` based on domain

    ### b) Validator profiles (tiered validation matching validation folders)
    - **system_instructions** (all tiers): Include the `auto_approved: true` interaction guidance from item 13.
    - **Tier 1 — Universal**: Basic quality gates for all code (syntax, formatting, secrets, security)
      - **Plugins**: `["cli", "environment", "filesystem_query", "references", "todo", "web_fetch"]`
      - **preselected**: Only the enablement knowledge base
      - **max_turns**: 5, **auto_approved**: true
    - **Tier 2 — Technology**: Language/framework-specific checks (e.g., Java/Spring conventions)
      - **Plugins**: `["cli", "environment", "filesystem_query", "references", "todo", "web_fetch"]`
      - **preselected**: Enablement + technology-scoped ADRs
      - **max_turns**: 10, **auto_approved**: true
    - **Tier 3 — Pattern compliance**: Verify implementations match skill/module templates
      - **Plugins**: `["cli", "environment", "file_edit", "filesystem_query", "lsp", "references", "todo", "waypoint", "web_fetch"]`
      - **plugin_configs.lsp.config_path**: `"${workspaceRoot}/.lsp.json"`
      - **preselected**: Enablement + relevant ADRs + skills + validation references discovered in Part 2
      - **max_turns**: 15, **auto_approved**: true
    - **Tier 4 — CI/CD**: Build and integration validation
      - **Plugins**: `["cli", "environment", "filesystem_query", "references", "todo", "web_fetch"]`
      - **preselected**: Enablement only
      - **max_turns**: 10, **auto_approved**: true
    - All validators use **icon_name**: `"validator"`
    - **Note**: `web_fetch` is included in all tiers so validators can download and install tools (linters, formatters, security scanners) autonomously without blocking on the parent for manual steps.

    ### c) Analyst profiles (research and documentation)
    - **When**: The knowledge base has a `model/` folder or high-level documentation
    - **Name**: e.g., `analyst-codebase-documentation`
    - **Description**: End with interaction hint (see item 13) since `auto_approved: false`.
    - **Plugins**: `["cli", "environment", "filesystem_query", "memory", "references", "todo", "web_fetch", "web_search"]`
    - **system_instructions**: Include the `auto_approved: false` interaction guidance from item 13.
    - **max_turns**: 15, **auto_approved**: false, **icon_name**: `"document"`, **gc**: budget config with defaults

    ### d) Investigator profiles (web research)
    - **Name**: e.g., `investigator-web-research`
    - **Description**: End with interaction hint (see item 13) since `auto_approved: false`.
    - **Plugins**: `["memory", "todo", "web_search", "web_fetch"]`
    - **system_instructions**: Include the `auto_approved: false` interaction guidance from item 13.
    - **max_turns**: 10, **auto_approved**: false, **icon_name**: `"search"`

15. **Save each profile** as `{{profiles_dir}}/<name>.json`.

16. **When updating existing profiles**: If a profile file already exists in `{{profiles_dir}}/`, read it first and preserve any manual customizations (e.g., `system_instructions`, `gc` config, `icon`). Only update the `plugin_configs.references.preselected` list and `description` to reflect the current state of the knowledge base. Do not overwrite fields that have been manually tuned.

---

## Common rules

17. **Skip folders that**:
    - Don't contain any documentation, validation, or template files
    - Are hidden (start with `.`)
    - Are named `node_modules`, `__pycache__`, `.git`, etc.
    - Match any pattern in `{{exclude_patterns}}`

18. **Computing paths**:

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

    **For remote sources** — compute the absolute path to the **materialized copy** under `.jaato/knowledge/<source-hash>/`, not the temporary download. The folder must have been copied there during Phase 2 (see "Materializing remote content"):

    Python:
    ```python
    import os
    rel = os.path.relpath(folder_path, repo_root).replace('\\', '/').rstrip('/')
    materialized = os.path.join(knowledge_dir, rel)  # knowledge_dir = .jaato/knowledge/<hash>
    abs_posix = os.path.abspath(materialized).replace('\\', '/').rstrip('/')
    ```

    Bash:
    ```bash
    rel=$(python -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$folder" "$repo_root")
    abs_posix=$(realpath "$KNOWLEDGE_DIR/$rel" 2>/dev/null)
    ```

19. **Overwrite protection** (when `{{dry_run}}` is false):
    - If an output file already exists and `{{force}}` is false → skip it, record in summary as `"skipped"` with reason `"exists"`.
    - If `{{force}}` is true → create a timestamped backup in `{{output}}/backups/` before overwriting.

20. **Merge mode**:
    - `"separate"` (default): Write one `<id>.json` file per reference into `{{output}}/`.
    - `"single"`: Accumulate all references into a single `{{output}}/references.json` array.

21. **Report results** — After all artifacts have been written (or planned), produce two outputs:

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

22. **Cleanup** — After all phases complete:
    - If source was remote: the referenced content has already been materialized to `.jaato/knowledge/<source-hash>/` during Phase 2. The temporary download directory (`$WORK_DIR`) is now disposable — remove it: `rm -rf "$WORK_DIR"`.
    - If `{{cache}}` is true: additionally copy the **full** downloaded tree to `.jaato/cache/sources/` before removing `$WORK_DIR`, so future runs with the same URL + ref can skip the download entirely. (The `.jaato/knowledge/` copy is always kept — it's what the references point to.)

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

### Multi-level subpath patterns

```
source="https://github.com/owner/knowledge-repo"
subpaths="knowledge/*,model/domains/*,model/standards/*,modules/*"
dry_run=false
```

Subpath patterns can be multi-level (e.g., `model/domains/*`, `model/standards/*`). These are resolved relative to the source root just like single-level patterns. The `*` expands the **last** path component:
- `knowledge/*` → matches `knowledge/ADRs/`, `knowledge/ERIs/`, etc.
- `model/domains/*` → matches `model/domains/code/`, `model/domains/design/`, etc. (each with `DOMAIN.md`)
- `model/standards/*` → matches `model/standards/authoring/`, `model/standards/traceability/`, `model/standards/validation/`
- `modules/*` → matches `modules/mod-code-001-...`, `modules/mod-code-015-...`, etc.

All matched directories are then traversed to find documentation folders with recognized entry-point files.

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

For a folder structure (with subpaths `knowledge/*,model/domains/*,model/standards/*,modules/*`):
```
knowledge/
├── ADRs/
│   └── adr-004-resilience-patterns/
│       └── ADR.md
├── ERIs/
│   └── eri-code-008-circuit-breaker-java-resilience4j/
│       └── ERI.md
model/
├── domains/
│   ├── README.md
│   ├── code/
│   │   ├── DOMAIN.md              ← entry-point: DOMAIN.md
│   │   ├── TAG-TAXONOMY.md
│   │   └── capabilities/
│   ├── design/
│   │   └── DOMAIN.md              ← entry-point: DOMAIN.md
│   ├── governance/
│   │   └── DOMAIN.md
│   └── qa/
│       └── DOMAIN.md
├── standards/
│   ├── ASSET-STANDARDS-v1.4.md    ← standalone file, parent = standards/
│   ├── DETERMINISM-RULES.md       ← standalone file, parent = standards/
│   ├── authoring/
│   │   ├── README.md              ← entry-point: README.md (has MODULE.md, ADR.md etc. too)
│   │   ├── ADR.md
│   │   ├── ERI.md
│   │   └── MODULE.md
│   ├── traceability/
│   │   ├── README.md              ← entry-point: README.md
│   │   └── profiles/
│   └── validation/                ← documentation folder (NOT a validation subfolder)
│       └── README.md              ← entry-point: README.md
modules/
├── mod-code-001-circuit-breaker-java-resilience4j/
│   ├── MODULE.md
│   ├── templates/
│   ├── policies/
│   │   └── naming-conventions.md
│   ├── scripts/
│   │   └── generate-config.sh
│   └── validation/                ← validation subfolder (inside a doc folder)
│       ├── README.md
│       └── circuit-breaker-check.sh
└── mod-code-015-hexagonal-base-java-spring/
    ├── MODULE.md
    ├── templates/
    │   ├── Application.java.tpl
    │   ├── domain/
    │   │   └── Entity.java.tpl
    │   └── adapter/
    │       └── RestController.java.tpl
    └── validation/
        └── README.md
```

**Generated reference files** in `{{output}}/`:
- `adr-004-resilience-patterns.json` (documentation, no contents)
- `eri-code-008-circuit-breaker-java-resilience4j.json` (documentation, no contents)
- `code.json` (documentation from `model/domains/code/`, entry-point: DOMAIN.md)
- `design.json` (documentation from `model/domains/design/`, entry-point: DOMAIN.md)
- `governance.json` (documentation from `model/domains/governance/`, entry-point: DOMAIN.md)
- `qa.json` (documentation from `model/domains/qa/`, entry-point: DOMAIN.md)
- `authoring.json` (documentation from `model/standards/authoring/`, entry-point: README.md)
- `traceability.json` (documentation from `model/standards/traceability/`, entry-point: README.md)
- `validation.json` (documentation from `model/standards/validation/`, entry-point: README.md — this is a doc folder, NOT a validation subfolder)
- `mod-code-001-circuit-breaker-java-resilience4j.json` (documentation, contents: templates + policies + scripts + validation)
- `mod-code-001-circuit-breaker-java-resilience4j-validation.json` (validation subfolder)
- `mod-code-015-hexagonal-base-java-spring.json` (documentation, contents: templates + validation)
- `mod-code-015-hexagonal-base-java-spring-validation.json` (validation subfolder)

**Generated template index** at `{{templates_index}}`:
```json
{
  "generated_at": "2026-02-24T12:00:00",
  "template_count": 3,
  "templates": {
    "mod-code-015-hexagonal-base-java-spring/Application.java.tpl": {
      "name": "mod-code-015-hexagonal-base-java-spring/Application.java.tpl",
      "source_path": "/.../mod-code-015-hexagonal-base-java-spring/templates/Application.java.tpl",
      "syntax": "mustache",
      "variables": ["ServiceName", "basePackage", "serviceName"],
      "origin": "standalone"
    },
    "mod-code-015-hexagonal-base-java-spring/domain/Entity.java.tpl": {
      "name": "mod-code-015-hexagonal-base-java-spring/domain/Entity.java.tpl",
      "source_path": "/.../mod-code-015-hexagonal-base-java-spring/templates/domain/Entity.java.tpl",
      "syntax": "mustache",
      "variables": ["Entity", "basePackage", "fields"],
      "origin": "standalone"
    },
    "mod-code-015-hexagonal-base-java-spring/adapter/RestController.java.tpl": {
      "name": "mod-code-015-hexagonal-base-java-spring/adapter/RestController.java.tpl",
      "source_path": "/.../mod-code-015-hexagonal-base-java-spring/templates/adapter/RestController.java.tpl",
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

1. **Phase 0**: Resolve the source — detect type. If remote, **download the full repository zip to an OS temp directory and extract it before doing anything else**. Then resolve subpaths and apply exclusions. After this phase, `repoRoot` is always a local directory on disk.
2. **Phase 1**: List the directory tree within the resolved directories (structure only, no file reads). Build the full inventory.
3. **Phase 1.5**: Evaluate parallelization — if 3+ categories or 10+ folders, spawn subagents to process category groups in parallel. Otherwise proceed sequentially.
4. **Phase 2**: Process categories (sequentially or via subagents). Each category/folder: read entry-point file → write doc-ref JSON → **validate with `validateReference`** → read validation README if present → write validation-ref JSON → **validate with `validateReference`** → read template files if present → accumulate index entries. If validation fails, fix the JSON and rewrite before proceeding.
5. **Phase 3**: Merge template entries (from subagents if parallel), write the template index JSON → **validate with `validateTemplateIndex`**. Fix and rewrite if validation fails.
6. **Phase 4**: Generate subagent profiles using the merged reference IDs → **validate each with `validateProfile`**. Fix and rewrite any profile that fails validation.
7. **Report**: Produce the final summary table and write `summary.json`.
8. **Cleanup**: Remove temporary download directories (`$WORK_DIR`) if source was remote and cache is disabled.
