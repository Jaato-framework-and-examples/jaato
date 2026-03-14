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
  kb_manager:
    required: false
    default: null
    description: >
      Base URL of a running Knowledge Manager instance (e.g. "http://localhost:3001").
      When provided, Phase 5 publishes the discovered knowledge structure to the KB
      Manager API. When null, Phase 5 is skipped entirely.
tags: ['references', 'generator', 'templates', 'profiles', 'git', 'archive', 'patterns', 'kb-manager']
---
Generate reference catalog, template index, and subagent profiles from a knowledge base — local or remote.

Input
Source: {{source}} — local path, git URL, or archive URL
Subpaths filter: {{subpaths}} (empty = scan everything)
Git ref: {{ref}}
References output: {{output}}
Templates index: {{templates_index}}
Profiles output: {{profiles_dir}}
Dry run: {{dry_run}} | Force: {{force}} | Cache: {{cache}} | Parallel: {{parallel}}
Merge mode: {{merge_mode}}
Exclude patterns: {{exclude_patterns}}
KB Manager: {{kb_manager}}

This prompt is intended to be used with the `gen-references` profile, which provides the required plugins: references (compute_embedding, validateReference), template (listTemplateVariables, validateTemplateIndex), subagent (validateProfile, subscribeToTasks), service_connector (discover_service, call_service for KB Manager), cli, file_edit, filesystem_query, web_fetch, clarification, introspection, todo, and permission.

See the canonical gen-references prompt in the knowledge base repository for the full operational instructions covering all six phases (source resolution, inventory, parallel evaluation, folder processing, template/embedding/profile generation, and KB Manager publication).
