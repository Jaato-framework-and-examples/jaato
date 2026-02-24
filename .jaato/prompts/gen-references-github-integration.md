---
description: Generate reference catalog JSON files from a local folder or public git/archive sources. Subpaths may be a comma-separated list of paths or glob patterns. Defaults to dry_run=true and does not support private repos.
tags: ['references', 'generator', 'git', 'archive', 'patterns']
---

Title: gen-references

Purpose:
Generate reference catalog JSON files from a source that may be local or remote (git/HTTP archive). Each documentation folder found under the source becomes one reference JSON entry.

Inputs:
- source (required): either
  - a local path to scan (e.g., "./.jaato/docs"), OR
  - a git/HTTPS URL for a public repository (e.g., "https://github.com/owner/repo", "git@github.com:owner/repo.git"),
  - or an archive URL (zip/tar).
- subpaths (optional): comma-separated list of paths or glob patterns to scan inside the source (e.g., "knowledge/*,model/domains/*,model/domains/standards/*,modules/*"). Patterns are relative to the repository root or local source root. Default: root of the source (equivalent to "./").
  - Behavior: split on commas, trim whitespace, ignore empty items. Each item may be:
    - an exact path (e.g., "knowledge" or "modules/mod-foo"), or
    - a glob pattern using '*' and '?' (e.g., "knowledge/*", "model/domains/*-code-*"), or
    - a path ending with '/' (treated as a directory match).
  - Matching semantics: expand patterns against the repository tree (or local file tree). For exact paths, match that directory if it exists. For glob patterns, match directories whose path matches the pattern. If a pattern matches files, the containing directory will be used.
- ref (optional): git branch, tag, or commit hash. Default: repository default branch.
- output_dir (optional): directory to write references (default: ".jaato/references")
- merge_mode (optional): "separate" (one file per reference) or "single" (single catalog references.json). Default: "separate"
- dry_run (optional): boolean. If true, do not write files; just report planned writes. Default: true
- force (optional): boolean. If true, overwrite existing files. Default: false
- cache (optional): boolean. If true, cache clones/archives for reuse. Default: false
- exclude_patterns (optional): list of glob patterns to skip (e.g., ["node_modules", "__pycache__"]). These apply after expansion of subpaths.
- auth_token_env (optional): environment variable name holding a token for private repo access (default: "GITHUB_TOKEN"). NOTE: private repos are NOT supported by this prompt; this parameter is accepted but ignored.
- verbose (optional): boolean. If true, produce human-readable progress output. Default: false

High-level behavior:
1. Resolve the source:
   - If source is a local path: treat as localRoot = source.
   - If source is a git URL:
     - Attempt to download a public archive (GitHub: /archive/<ref>.zip) and extract to temp_dir. Do not attempt authenticated access. If archive download fails due to access restrictions, return a clear error indicating private repositories are not supported.
     - If the archive approach fails and git is available, perform a shallow clone: git clone --depth 1 --branch <ref> <url> <temp_dir> (public repos only).
   - If source is an archive URL: download and extract to temp_dir.
   - After extraction/clone, set repoRoot = temp_dir (or the local source root). Then resolve subpaths (see next step).
2. Resolve subpaths/patterns:
   - If subpaths is empty or unspecified: treat as a single pattern "./" (scan entire repoRoot).
   - Otherwise parse the comma-separated list into individual patterns.
   - For each pattern:
     - If it's an exact path and exists as a directory under repoRoot, add it to the list of matched directories.
     - If it contains glob wildcards, expand it against the repo tree and add all matching directories.
     - If the pattern matches files, use their parent directories (deduplicate results).
   - Deduplicate matched directories and sort deterministically.
   - If a pattern matches nothing, record a warning for that pattern in the summary and continue.
3. Traverse each matched directory recursively to find documentation folders:
   - Consider documentation files case-insensitively: MODULE.md, SKILL.md, ERI.md, OVERVIEW.md, README.md, *.md, *.rst, *.adoc
   - Skip hidden folders and exclude_patterns; always skip known vendored directories (node_modules, .git, __pycache__).
   - Each folder that contains at least one documentation file becomes a candidate reference entry.
4. For each candidate folder found:
   - Extract metadata:
     - YAML frontmatter (if present): use title, description, tags when available.
     - Fallback: first paragraph of the preferred doc file for description.
     - Tags: frontmatter tags first, then path components + tokens from folder name (lowercased, deduped).
     - Name: generate from frontmatter title if present, else apply pattern rules to the folder name:
       - mod-code-001-foo → "MOD-001: Foo"
       - skill-001-foo → "SKILL-001: Foo"
       - otherwise, hyphens → spaces, Title Case
     - id: folder name slug (lowercase, hyphens). Validate slug character set and length.
   - Build reference object:
     {
       "id": "<folder-name>",
       "name": "<Human Readable Name>",
       "description": "<Brief description>",
       "type": "local",
       "path": "./<relative-path-to-folder>/",
       "localPath": "<absolute-or-temp-local-path>",
       "source": { "type": "<local|git|archive>", "url": "<original-source-url-or-path>", "ref": "<ref-if-applicable>", "subpath": "<subpath-if-any>" },
       "mode": "selectable",
       "tags": ["<tag1>", "<tag2>"],
       "fetchHint": "<preferred-doc-file e.g., MODULE.md>"
     }
5. Output:
   - The generator will include in the summary an object mapping: requested_patterns → resolved_matched_directories (so callers can see how patterns expanded).
   - If merge_mode == "separate": write <id>.json into output_dir (filename: <id>.json)
   - If merge_mode == "single": add items to a single references.json array and write to output_dir
   - Always include a summary.json in output_dir with:
     {
       "requested_patterns": ["..."],
       "matched_directories": ["..."],
       "generated": [<filenames>],
       "skipped": [{"path": "...", "reason": "..."}],
       "warnings": [{"pattern": "...", "message": "no matches"}],
       "counts": {"found": N, "generated": M, "skipped": K},
       "timestamp": "<ISO8601>"
     }
   - If dry_run=true: do not write files; produce planned_writes in the summary and return success.
   - If an output file already exists:
     - If force=true: overwrite and create a backup copy (timestamped) in output_dir/backups/
     - If force=false: skip and record in summary.skipped with reason "exists"
6. Reporting and diagnostics:
   - Return a machine-readable report (summary.json). If verbose=true, also print a short human summary.
   - Record warnings (parsing failures, missing frontmatter, unmatched patterns) per-folder/pattern in the summary.
7. Cleanup:
   - Remove temporary clone/extract directories unless cache=true (cache dir location configurable).
8. Errors:
   - For private repositories or access-denied errors, fail with a clear message: "Private repositories are not supported by this prompt. Provide a public repo or a local path."
   - For network issues, include the HTTP/git error in summary and abort gracefully.

Examples:
- Subpaths example (patterns):
  subpaths="knowledge/*,model/domains/*,model/domains/standards/*,modules/*"
  This will expand each pattern against the repo tree and scan all matched directories.

- Local scan (single path):
  source="./.jaato/docs"
  subpaths="modules"
  dry_run=false

- GitHub public repo:
  source="https://github.com/owner/repo"
  subpaths="docs/*,knowledge"
  ref="main"

Notes:
- The generator will prefer YAML frontmatter for title/description/tags when available.
- For remote sources the generated reference.path should still be a relative path (e.g., "./modules/mod-code-001.../") representing layout inside the source; the reference.source object will contain the canonical origin.
- The prompt will not store or print tokens; tokens are used only for the HTTP/Git operation and not written to outputs.

Begin:
- Validate that required inputs are present.
- Resolve the source as above (download archive or clone if remote).
- Expand subpaths/patterns and resolve matched directories.
- Proceed with the traversal and generation steps over the matched directories.
- When complete, write outputs (unless dry_run) and produce the summary.

Execution behavior (overrides):
- If missing required inputs, use the clarification tool to request them.
- By default run in dry_run=true to avoid accidental overwrites. Callers may change dry_run/force options.
