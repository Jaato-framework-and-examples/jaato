# Reliability Policies Configuration Guide

The reliability plugin detects and intervenes when models get stuck in unproductive loops — retrying failed tools, reading without acting, calling the same tool endlessly, or skipping required prerequisites. All detection thresholds and intervention rules are configurable through a single JSON file.

---

## Config File Location

The plugin loads the first file it finds:

1. `.jaato/reliability-policies.json` (workspace — per-project)
2. `~/.jaato/reliability-policies.json` (user home — global default)

Create the workspace version for project-specific tuning. Use the home version for your personal baseline. Workspace always wins.

**Management commands** (from the TUI):

```
reliability policies status    # Show active config and loaded policies
reliability policies edit      # Open config in $EDITOR (creates with defaults if missing)
reliability policies reload    # Hot-reload after manual edits
reliability policies path      # Show which config file is active
```

---

## Full Schema Reference

```json
{
  "pattern_detection": {
    "repetitive_call_threshold": 3,
    "error_retry_threshold": 3,
    "error_retry_overrides": {},
    "introspection_loop_threshold": 2,
    "max_reads_before_action": 5,
    "max_turn_duration_seconds": 120.0,
    "introspection_tool_names": ["list_tools", "get_tool_schemas", "askPermission"],
    "read_only_tools": ["readFile", "Read", "Glob", "Grep", "glob", "grep"],
    "action_tools": ["writeFile", "Write", "Edit", "Bash", "bash", "updateFile", "removeFile"],
    "announce_phrases": ["let me", "proceeding now", "i'll now", "executing"]
  },
  "prerequisite_policies": []
}
```

**Every field is optional.** Missing fields retain their built-in defaults shown above.

---

## Pattern Detection Fields

### Repetition Control

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repetitive_call_threshold` | int | 3 | Consecutive calls to the same tool (regardless of success) before triggering `REPETITIVE_CALLS` |
| `error_retry_threshold` | int | 3 | Consecutive *failures* of the same tool with similar arguments before triggering `ERROR_RETRY_LOOP` |
| `error_retry_overrides` | object | `{}` | Per-tool overrides for `error_retry_threshold`. Maps tool name → threshold. Tools not listed fall back to the global value |

### Introspection & Progress

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `introspection_loop_threshold` | int | 2 | Introspection tool calls without an action before triggering `INTROSPECTION_LOOP` |
| `introspection_tool_names` | string[] | `["list_tools", "get_tool_schemas", "askPermission"]` | Which tools count as "introspection" |
| `max_reads_before_action` | int | 5 | Read-only tool calls without a write/action before triggering `READ_ONLY_LOOP` |
| `read_only_tools` | string[] | *(see above)* | Which tools count as read-only |
| `action_tools` | string[] | *(see above)* | Which tools count as action/mutation |

### Time & Announce

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_turn_duration_seconds` | float | 120.0 | Single turn duration before triggering a stall warning |
| `announce_phrases` | string[] | *(see above)* | Phrases in model output that suggest it's about to act. Detection fires if the model announces but no action tool follows |

---

## Prerequisite Policies

Prerequisite policies enforce ordering: "tool X must be called before tool Y." Each policy is an object in the `prerequisite_policies` array.

### Policy Fields

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `policy_id` | yes | string | — | Unique identifier for this policy |
| `prerequisite_tool` | yes | string | — | The tool that must be called first |
| `gated_tools` | yes | string[] | — | Tools that require the prerequisite |
| `lookback_turns` | no | int | 2 | How many previous turns to check for the prerequisite |
| `severity_thresholds` | no | object | `{"minor": 0, "moderate": 1, "severe": 2}` | Maps severity to minimum *prior* violation count |
| `nudge_templates` | no | object | *(built-in)* | Custom nudge messages per severity |
| `expected_action_template` | no | string | `"Call {prerequisite_tool} before using {tool_name}"` | Template for the expected action field on detected patterns |

### severity_thresholds

Controls how fast violations escalate. Maps severity name (`minor`, `moderate`, `severe`) to the minimum number of *prior* violations needed to reach that level. The detector picks the highest severity whose threshold is met.

```json
"severity_thresholds": {
  "minor": 0,
  "moderate": 1,
  "severe": 2
}
```

This default means: 1st violation → minor (gentle reminder), 2nd → moderate (direct instruction), 3rd+ → severe (interrupt/block).

### nudge_templates

Custom messages injected when a violation is detected. Each entry maps a severity name to a `[nudge_type, message]` pair.

**Nudge types:**
- `"gentle"` — Soft suggestion appended to context
- `"direct"` — Clear instruction injected as system guidance
- `"interrupt"` — Blocks execution and requests user intervention

**Template variables:**
- `{tool_name}` — The gated tool that was called
- `{prerequisite_tool}` — The required tool
- `{count}` — Number of prior violations

---

## Escalation Flow

When a pattern is detected, the severity determines the intervention:

```
MINOR    →  Gentle reminder   →  "Consider trying a different approach."
MODERATE →  Direct instruction →  "NOTICE: Stop retrying and try an alternative."
SEVERE   →  Interrupt (block)  →  "BLOCKED: System pausing for user guidance."
```

Severity escalates automatically as the same pattern repeats within a session.

---

## Usage Examples

### Example 1: Strict Environment — Block Failures Fast

You're running a CI pipeline where any tool failure is likely a real problem, not a transient glitch. Block on the first retry attempt.

```json
{
  "pattern_detection": {
    "error_retry_threshold": 1,
    "repetitive_call_threshold": 2,
    "max_reads_before_action": 3
  }
}
```

**Effect:** Any tool that fails once and is retried immediately triggers `ERROR_RETRY_LOOP`. The model gets two identical calls max before `REPETITIVE_CALLS` fires. Reading 3 files without acting triggers `READ_ONLY_LOOP`.

---

### Example 2: Flaky Network — Per-Tool Tolerance

You're working with remote APIs that have intermittent failures, but local tools should still be strict.

```json
{
  "pattern_detection": {
    "error_retry_threshold": 2,
    "error_retry_overrides": {
      "web_search": 6,
      "http_request": 5,
      "mcp_call": 5,
      "bash": 2,
      "readFile": 1
    }
  }
}
```

**Effect:** Network tools (`web_search`, `http_request`, `mcp_call`) get 5-6 retries before blocking. `bash` blocks after 2 consecutive failures. `readFile` blocks immediately on a retry (if a file doesn't exist, retrying won't help). Everything else uses the global default of 2.

---

### Example 3: Plan-Before-Act Workflow

You want the model to always create a plan before modifying files, and to immediately block (not just warn) on the first violation.

```json
{
  "prerequisite_policies": [
    {
      "policy_id": "plan_before_edit",
      "prerequisite_tool": "createPlan",
      "gated_tools": ["writeFile", "updateFile", "removeFile", "Write", "Edit"],
      "lookback_turns": 10,
      "severity_thresholds": {
        "minor": 0,
        "moderate": 0,
        "severe": 0
      },
      "nudge_templates": {
        "severe": ["interrupt", "BLOCKED: You must create a plan before modifying files. Call createPlan first."]
      }
    }
  ]
}
```

**Effect:** All `severity_thresholds` set to 0 means the very first violation is immediately `SEVERE` → interrupt. The model cannot write any file until it has called `createPlan` within the last 10 turns.

---

### Example 4: Template-Aware Code Generation

You want the model to check available templates before writing new files, but allow updates without checking (since the file already exists).

```json
{
  "prerequisite_policies": [
    {
      "policy_id": "template_check",
      "prerequisite_tool": "listAvailableTemplates",
      "gated_tools": ["writeNewFile", "createFile"],
      "lookback_turns": 3,
      "severity_thresholds": {
        "minor": 0,
        "moderate": 2,
        "severe": 4
      },
      "nudge_templates": {
        "minor": ["gentle", "Consider checking listAvailableTemplates before creating a new file — there might be a relevant template."],
        "moderate": ["direct", "You've created {count} files without checking templates. Call listAvailableTemplates."],
        "severe": ["interrupt", "BLOCKED: Too many files created without checking templates. Call listAvailableTemplates first."]
      }
    }
  ]
}
```

**Effect:** The first 2 violations produce gentle reminders. Violations 3-4 get direct instructions. After 4 violations, the model is blocked. This is intentionally lenient — sometimes you know the template doesn't apply, but after 4 skips it's likely the model forgot about templates entirely.

---

### Example 5: Read-Then-Act for Database Operations

Ensure the model reads the current schema before running any migration or DDL command.

```json
{
  "prerequisite_policies": [
    {
      "policy_id": "schema_check",
      "prerequisite_tool": "describeTable",
      "gated_tools": ["runMigration", "executeDDL", "alterTable"],
      "lookback_turns": 5,
      "severity_thresholds": {
        "minor": 0,
        "moderate": 0,
        "severe": 1
      },
      "nudge_templates": {
        "minor": ["direct", "Call describeTable to check the current schema before running {tool_name}."],
        "severe": ["interrupt", "BLOCKED: You must inspect the schema with describeTable before {tool_name}."]
      }
    }
  ]
}
```

**Effect:** First violation gets a direct instruction (skipping gentle — schema mistakes are costly). Second violation blocks. The model must run `describeTable` within the last 5 turns before any DDL operation.

---

### Example 6: Combined — Full Production Config

A realistic config combining pattern detection tuning with multiple prerequisite policies.

```json
{
  "pattern_detection": {
    "repetitive_call_threshold": 4,
    "error_retry_threshold": 3,
    "error_retry_overrides": {
      "web_search": 5,
      "bash": 2
    },
    "introspection_loop_threshold": 3,
    "max_reads_before_action": 7,
    "max_turn_duration_seconds": 180.0
  },
  "prerequisite_policies": [
    {
      "policy_id": "plan_before_update",
      "prerequisite_tool": "createPlan",
      "gated_tools": ["setStepStatus", "completeStep"],
      "lookback_turns": 8,
      "severity_thresholds": {
        "minor": 0,
        "moderate": 1,
        "severe": 2
      }
    },
    {
      "policy_id": "read_before_edit",
      "prerequisite_tool": "readFile",
      "gated_tools": ["updateFile", "Edit"],
      "lookback_turns": 2,
      "severity_thresholds": {
        "minor": 0,
        "moderate": 0,
        "severe": 1
      },
      "nudge_templates": {
        "minor": ["direct", "Read the file before editing it. Call readFile for {tool_name}'s target."],
        "severe": ["interrupt", "BLOCKED: You must read a file before editing it."]
      }
    }
  ]
}
```

**Effect:**
- Repetitive calls tolerated up to 4 before first nudge
- `bash` failures block after 2 retries; `web_search` after 5; everything else after 3
- Introspection loops tolerated up to 3 calls (useful if the tool set is large)
- Reading up to 7 files before requiring an action (for large refactors)
- Turns can run up to 3 minutes before a stall warning
- Plans required before step updates (standard escalation)
- File reads required before edits (immediate direct instruction, block on second offense)

---

## Severity Threshold Patterns

Quick reference for common `severity_thresholds` configurations:

| Pattern | Thresholds | Behavior |
|---------|-----------|----------|
| **Immediate block** | `{"minor": 0, "moderate": 0, "severe": 0}` | First violation → interrupt |
| **Warn then block** | `{"minor": 0, "moderate": 0, "severe": 1}` | First → direct instruction, second → block |
| **Standard (default)** | `{"minor": 0, "moderate": 1, "severe": 2}` | Gentle → direct → block over 3 violations |
| **Lenient** | `{"minor": 0, "moderate": 3, "severe": 6}` | 3 gentle nudges, then 3 direct, then block |
| **Advisory only** | `{"minor": 0}` | Only gentle reminders, never escalates |

---

## Related Documentation

- [Reliability Plugin Design](reliability-plugin-design.md) — Architecture, failure recording, trust state machine
- [Architecture Overview](architecture.md) — Server-first architecture, event protocol
- [Design Philosophy](design-philosophy.md) — Opinionated design decisions and rationale
