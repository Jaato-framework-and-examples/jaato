# Tool Telemetry Enrichment Assessment

Assessment of which tool executors and enrichment plugins could benefit from the
`_telemetry` convention introduced in #744 (memory plugin).

## How the Convention Works

Tool executors include a `"_telemetry"` dict in their result.  The session's
tool execution path (both sequential and parallel) extracts each entry and
forwards it as a span attribute on the enclosing `tool_span`.  The `_telemetry`
key is stripped by `_build_tool_result` (which removes all `_`-prefixed keys)
before results reach the model.

```python
# In any executor:
return {
    "status": "success",
    # ...model-visible fields...
    "_telemetry": {
        "jaato.myplugin.some_metric": value,
    },
}
```

The session picks it up automatically — no coupling to the telemetry plugin.

---

## Tier 1 — High Value (rich domain metrics, direct observability wins)

### 1. CLI Plugin (`cli/plugin.py`)

**Tools:** `cli_based_tool`

Every shell command already returns `stdout`, `stderr`, `returncode`.
Surfacing these as span attributes enables dashboards for command health.

```python
"_telemetry": {
    "jaato.cli.command": cmd_name,          # e.g. "git", "npm", "pytest"
    "jaato.cli.returncode": returncode,
    "jaato.cli.stdout_bytes": len(stdout),
    "jaato.cli.stderr_bytes": len(stderr),
    "jaato.cli.shell_mode": bool,           # shell vs direct exec
    "jaato.cli.timeout_used": bool,
    "jaato.cli.cwd": cwd,
}
```

**Why high value:** CLI is the most-called tool.  Knowing which commands fail,
how much output they produce, and whether they time out answers the most common
"what is the agent actually doing" questions.

### 2. File Edit Plugin (`file_edit/plugin.py`)

**Tools:** `readFile`, `updateFile`, `writeNewFile`, `removeFile`, `moveFile`,
`multiFileEdit`, `findAndReplace`, `undoFileChange`, `restoreFile`, `listBackups`

Write operations already return `path`, `size`, `lines`.

```python
# updateFile
"_telemetry": {
    "jaato.file.operation": "update",
    "jaato.file.path": normalized_path,
    "jaato.file.size_bytes": len(new_content),
    "jaato.file.lines": line_count,
    "jaato.file.mode": "targeted" | "full_replacement",
    "jaato.file.had_backup": bool,
}

# readFile
"_telemetry": {
    "jaato.file.operation": "read",
    "jaato.file.path": normalized_path,
    "jaato.file.lines": total_lines,
    "jaato.file.chunked": bool,
    "jaato.file.has_more": bool,
}

# multiFileEdit
"_telemetry": {
    "jaato.file.operation": "multi_edit",
    "jaato.file.files_count": len(operations),
    "jaato.file.total_edits": total_edit_count,
    "jaato.file.all_succeeded": bool,
}

# findAndReplace
"_telemetry": {
    "jaato.file.operation": "find_replace",
    "jaato.file.files_matched": files_matched,
    "jaato.file.replacements": replacement_count,
    "jaato.file.dry_run": bool,
}
```

**Why high value:** File operations are the primary way the agent modifies code.
Tracking file sizes, operation types, and backup usage gives direct insight into
agent editing patterns, file churn, and whether undo/restore is being used.

### 3. MCP Plugin (`mcp/plugin.py`)

**Tools:** Dynamic (one per MCP tool from connected servers)

The MCP plugin wraps calls to external MCP servers.  The executor already knows
the server name and original tool name.

```python
"_telemetry": {
    "jaato.mcp.server_name": server_name,
    "jaato.mcp.original_tool": original_tool_name,
    "jaato.mcp.response_type": "text" | "image" | "resource",
    "jaato.mcp.content_items": len(result.content),
}
```

**Why high value:** MCP tools are opaque — the framework has no visibility into
what external servers do.  Server name + response shape is the minimum needed
for debugging MCP integration issues.

### 4. Subagent Plugin (`subagent/plugin.py`)

**Tools:** `spawn_subagent`, `send_to_subagent`, `close_subagent`,
`cancel_subagent`, `list_active_subagents`

```python
# spawn_subagent
"_telemetry": {
    "jaato.subagent.operation": "spawn",
    "jaato.subagent.agent_id": agent_id,
    "jaato.subagent.profile": profile_name or "default",
    "jaato.subagent.model": model_name,
    "jaato.subagent.has_inline_config": bool,
    "jaato.subagent.active_count": current_active_count,
}

# close_subagent / cancel_subagent
"_telemetry": {
    "jaato.subagent.operation": "close" | "cancel",
    "jaato.subagent.agent_id": agent_id,
    "jaato.subagent.was_running": bool,
    "jaato.subagent.turns_used": turns,
}
```

**Why high value:** Subagent lifecycle is one of the hardest things to debug.
Trace data showing spawn profiles, active counts, and whether agents are
cancelled vs closed cleanly would be extremely useful for multi-agent debugging.

### 5. Interactive Shell Plugin (`interactive_shell/plugin.py`)

**Tools:** `shell_spawn`, `shell_input`, `shell_read`, `shell_control`,
`shell_close`, `shell_list`

```python
# shell_spawn
"_telemetry": {
    "jaato.shell.operation": "spawn",
    "jaato.shell.session_id": session_id,
    "jaato.shell.command": command,
    "jaato.shell.backend": backend_name,       # pexpect vs popen_spawn
    "jaato.shell.is_alive": bool,
    "jaato.shell.active_sessions": active_count,
}

# shell_input / shell_control
"_telemetry": {
    "jaato.shell.operation": "input" | "control",
    "jaato.shell.session_id": session_id,
    "jaato.shell.output_bytes": len(output),
    "jaato.shell.is_alive": bool,
}

# shell_close
"_telemetry": {
    "jaato.shell.operation": "close",
    "jaato.shell.session_id": session_id,
    "jaato.shell.exit_status": exit_status,
    "jaato.shell.lifetime_seconds": lifetime,
}
```

**Why high value:** Interactive shells are long-lived and stateful — the hardest
tool type to observe.  Session lifetime, exit status, and backend choice are
critical debugging signals.

---

## Tier 2 — Medium Value (useful context, moderate observability benefit)

### 6. Filesystem Query Plugin (`filesystem_query/plugin.py`)

**Tools:** `glob_files`, `grep_content`

```python
# glob_files
"_telemetry": {
    "jaato.fsquery.operation": "glob",
    "jaato.fsquery.pattern": pattern,
    "jaato.fsquery.matches": total_count,
    "jaato.fsquery.truncated": total_count > max_results,
}

# grep_content
"_telemetry": {
    "jaato.fsquery.operation": "grep",
    "jaato.fsquery.pattern": pattern,
    "jaato.fsquery.matches": match_count,
    "jaato.fsquery.files_searched": files_searched,
    "jaato.fsquery.truncated": bool,
}
```

**Why:** Search patterns and hit counts help understand how the agent navigates
code.  Detecting high truncation rates flags over-broad searches.

### 7. Web Fetch Plugin (`web_fetch/plugin.py`)

**Tools:** `web_fetch`

```python
"_telemetry": {
    "jaato.web.operation": "fetch",
    "jaato.web.url_host": parsed_url.netloc,
    "jaato.web.mode": mode,                    # markdown | structured | raw
    "jaato.web.status_code": status_code,
    "jaato.web.content_type": content_type,
    "jaato.web.content_bytes": content_length,
    "jaato.web.cache_hit": bool,
    "jaato.web.redirected": bool,
    "jaato.web.used_proxy": bool,
}
```

**Why:** Network calls are latency-sensitive and failure-prone.  Cache hit rate,
status codes, and content sizes drive performance tuning decisions.

### 8. Web Search Plugin (`web_search/plugin.py`)

**Tools:** `web_search`

```python
"_telemetry": {
    "jaato.web.operation": "search",
    "jaato.web.query_length": len(query),
    "jaato.web.result_count": len(results),
    "jaato.web.timed_out": bool,
}
```

**Why:** Search quality and timeout rates are direct signals of whether the
agent is getting useful external information.

### 9. Notebook Plugin (`notebook/plugin.py`)

**Tools:** `notebook_execute`, `notebook_create`, `notebook_variables`,
`notebook_reset`, `notebook_list`, `notebook_backends`

```python
# notebook_execute
"_telemetry": {
    "jaato.notebook.operation": "execute",
    "jaato.notebook.notebook_id": notebook_id,
    "jaato.notebook.backend": backend_name,
    "jaato.notebook.code_lines": code.count('\n') + 1,
    "jaato.notebook.has_output": bool,
    "jaato.notebook.execution_count": exec_count,
    "jaato.notebook.sandbox_mode": sandbox_mode,
}
```

**Why:** Code execution backends, sandbox modes, and execution counts help
profile computational workloads and detect runaway execution patterns.

### 10. Todo Plugin (`todo/plugin.py`)

**Tools:** `createPlan`, `startPlan`, `setStepStatus`, `getPlanStatus`,
`completePlan`, `addStep`, `subscribeToTasks`, etc.

```python
# createPlan
"_telemetry": {
    "jaato.plan.operation": "create",
    "jaato.plan.step_count": len(steps),
    "jaato.plan.agent": agent_name,
}

# setStepStatus
"_telemetry": {
    "jaato.plan.operation": "set_status",
    "jaato.plan.status": status_str,
    "jaato.plan.step_sequence": step.sequence,
    "jaato.plan.progress_pct": progress["percent"],
}

# completePlan
"_telemetry": {
    "jaato.plan.operation": "complete",
    "jaato.plan.final_status": status_str,
    "jaato.plan.total_steps": len(plan.steps),
    "jaato.plan.completed_steps": progress["completed"],
    "jaato.plan.failed_steps": progress.get("failed", 0),
}
```

**Why:** Plan lifecycle telemetry answers "how well does the agent follow its
own plans?" — completion rates, failure rates, and plan sizes.

### 11. Template Plugin (`template/plugin.py`)

**Tools:** `writeFileFromTemplate`, `listAvailableTemplates`,
`listTemplateVariables`, `validateTemplateIndex`

```python
# writeFileFromTemplate
"_telemetry": {
    "jaato.template.operation": "write",
    "jaato.template.name": template_name,
    "jaato.template.syntax": "jinja2" | "mustache",
    "jaato.template.origin": "standalone" | "embedded",
    "jaato.template.variable_count": len(variables),
    "jaato.template.output_path": output_path,
}
```

**Why:** Template usage patterns show which templates are popular, whether
embedded vs standalone is preferred, and which syntax is dominant.

---

## Tier 3 — Lower Priority (niche or low-frequency tools)

### 12. AST Search Plugin (`ast_search/plugin.py`)

**Tools:** `ast_search`

```python
"_telemetry": {
    "jaato.ast.pattern": pattern,
    "jaato.ast.language": language,
    "jaato.ast.match_count": count,
    "jaato.ast.files_searched": files_count,
}
```

### 13. Clarification Plugin (`clarification/plugin.py`)

**Tools:** `request_clarification`

```python
"_telemetry": {
    "jaato.clarification.question_count": len(questions),
    "jaato.clarification.all_answered": bool,
    "jaato.clarification.skipped_count": skipped,
}
```

### 14. Calculator Plugin (`calculator/plugin.py`)

**Tools:** `add`, `subtract`, `multiply`, `divide`, `calculate`

Minimal value — these are trivial computations with no interesting metrics.

### 15. Environment Plugin (`environment/plugin.py`)

**Tools:** `get_environment`

```python
"_telemetry": {
    "jaato.env.aspect": aspect,
}
```

### 16. Introspection Plugin (`introspection/plugin.py`)

**Tools:** `list_tools`, `get_tool_schemas`

```python
"_telemetry": {
    "jaato.introspection.operation": operation,
    "jaato.introspection.tools_returned": count,
    "jaato.introspection.categories_requested": categories,
}
```

### 17. Prompt Library Plugin (`prompt_library/plugin.py`)

**Tools:** `listPrompts`, `usePrompt`, `savePrompt`, `deletePrompt`

```python
"_telemetry": {
    "jaato.prompt.operation": operation,
    "jaato.prompt.name": prompt_name,
}
```

### 18. References Plugin (`references/plugin.py`)

**Tools:** `selectReferences`

```python
"_telemetry": {
    "jaato.references.selected_count": len(selected),
    "jaato.references.source_types": source_types,
}
```

### 19. LSP Plugin (`lsp/plugin.py`)

**Tools:** Dynamic (diagnostics, hover, definition, references, code_actions)

```python
"_telemetry": {
    "jaato.lsp.operation": operation,     # diagnostics | hover | definition | ...
    "jaato.lsp.language": language_id,
    "jaato.lsp.result_count": count,
}
```

---

## Beyond Tool Executors — Enrichment Pipeline Telemetry

The `_telemetry` convention is designed for tool executor results.  But several
plugins operate outside the executor path — they enrich prompts, system
instructions, or tool results.  These cannot use `_telemetry` directly, but
they **could** benefit from analogous span instrumentation.

### Prompt Enrichment Plugins

These run inside `enrich_prompt()` before the LLM call, within the turn span
but outside any tool span.

| Plugin | What it does | Telemetry opportunity |
|--------|-------------|----------------------|
| **Memory** | Injects hint about matching memories | `memory.prompt_matches`, `memory.matched_tags` |
| **References** | Resolves `@ref` mentions, pins sources | `references.resolved_count`, `references.pinned_count` |
| **Waypoint** | Injects pending restore notifications | `waypoint.restore_pending` |
| **Reliability (nudge)** | Injects behavioral nudges | `reliability.nudge_severity`, `reliability.pattern_type` |

**Recommendation:** Add span events (not attributes) on the turn span for
enrichment activity.  The registry's `enrich_prompt()` already collects
metadata from each plugin — this metadata could be forwarded to the turn span
via a similar convention.

### Tool Result Enrichment Plugins

These run inside `enrich_tool_result()` after the executor returns, still
within the tool span.

| Plugin | What it does | Telemetry opportunity |
|--------|-------------|----------------------|
| **LSP** | Runs diagnostics on modified files | `lsp.diagnostics_count`, `lsp.errors`, `lsp.warnings` |
| **Artifact Tracker** | Tracks created/modified artifacts | `artifact.tracked_count`, `artifact.related_files` |
| **Template** | Extracts embedded templates from results | `template.extracted_count`, `template.syntax` |

**Recommendation:** These are the best candidates for a **second convention**:
result enrichers could return telemetry metadata in their
`ToolResultEnrichmentResult.metadata` dict under a `"_telemetry"` key, and the
registry's `enrich_tool_result()` method could forward those to the active tool
span.  This would be a natural extension of the existing convention.

### System Instruction Enrichment

These run during system instruction assembly.  Useful for understanding token
budget allocation but harder to attribute to specific spans.

| Plugin | What it does | Telemetry opportunity |
|--------|-------------|----------------------|
| **Template** | Injects template catalog | `template.catalog_size` |
| **Memory** | Injects memory system instructions | `memory.instruction_tokens` |

**Recommendation:** Lower priority.  Consider adding a turn-level event
summarizing system instruction enrichment if token budget debugging becomes
important.

---

## Implementation Notes

### Session-Side Fix Needed

The current implementation only extracts `_telemetry` from **tuple** results
`(ok, result_dict)`.  Many executors return **plain dicts** (e.g., memory,
file_edit, todo, web_search).  The ToolExecutor in `ai_tool_runner.py` wraps
these in tuples (`True, result`), so by the time the session sees them they are
tuples — but this should be verified for all code paths.

### Attribute Naming Convention

Follow the pattern established by the memory plugin:

```
jaato.<plugin>.<metric>
```

Examples: `jaato.cli.returncode`, `jaato.file.operation`, `jaato.mcp.server_name`

### What NOT to Telemetry

- **File contents** or command output — too large, privacy-sensitive
- **User prompts** or model responses — covered by LLM span attributes
- **Passwords, tokens, secrets** — never
- **Per-item lists** — use counts, not enumerated values (except small sets
  like maturity/scope enums)

---

## Suggested Implementation Order

1. **CLI + File Edit** — highest call frequency, biggest observability gap
2. **MCP + Subagent** — hardest to debug without telemetry
3. **Interactive Shell** — stateful, long-lived, complements CLI
4. **Filesystem Query + Web Fetch/Search** — I/O-bound, latency-sensitive
5. **Notebook + Todo + Template** — valuable but lower frequency
6. **Enrichment pipeline convention** — extends pattern to non-executor plugins
7. **Remaining Tier 3 plugins** — as needed
