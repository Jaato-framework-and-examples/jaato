# Reliability Command Reference

The `reliability` command manages the reliability plugin — a cross-cutting system that tracks tool failures, detects behavioral patterns in models, and intervenes when the agentic loop stalls. It wraps the existing permission system to dynamically escalate tools that become unreliable, and injects nudges to guide models out of unproductive loops.

---

## Synopsis

```
reliability [subcommand] [args...]
```

When invoked without a subcommand, defaults to `status`.

---

## Subcommands

| Subcommand | Purpose |
|------------|---------|
| [`status`](#status) | Show escalated tools and their trust states |
| [`recovery`](#recovery) | Set automatic or manual recovery mode |
| [`reset`](#reset) | Reset a tool (or all tools) to the trusted state |
| [`history`](#history) | Show recent failure history |
| [`config`](#config) | Show current escalation rule configuration |
| [`settings`](#settings) | View, save, or clear reliability settings |
| [`model`](#model) | Model-specific reliability tracking and switching |
| [`patterns`](#patterns) | Behavioral pattern detection management |
| [`nudge`](#nudge) | Nudge injection intensity control |
| [`policies`](#policies) | File-based prerequisite policy management |
| [`behavior`](#behavior) | Model behavioral profile analysis |

---

## status

Show the current reliability status for all tracked tools.

```
reliability
reliability status
```

**Output:**
- If all tools are trusted: reports the count of tracked tools.
- If any tools are escalated/blocked/recovering: lists each non-trusted tool with its state, recovery progress (if recovering), and escalation reason.

**Example output:**

```
Reliability Status:
------------------------------------------------------------
  readFile|path_prefix=/etc
    State: escalated
    Reason: 3 failures in 1 hour (server_error)
  http_request|domain=api.example.com
    State: recovering (1/3)
    Reason: repeated timeouts
------------------------------------------------------------
2 escalated, 14 trusted
```

---

## recovery

View or set the recovery mode, which controls how escalated tools return to the trusted state.

```
reliability recovery
reliability recovery auto|ask
reliability recovery auto|ask save workspace|user
reliability recovery save workspace|user
```

| Argument | Description |
|----------|-------------|
| *(none)* | Show the current recovery mode |
| `auto` | Automatically recover tools after they accumulate enough consecutive successes |
| `ask` | Prompt the user before recovering an escalated tool |
| `save workspace` | Persist the current recovery mode to `.jaato/reliability.json` |
| `save user` | Persist the current recovery mode to `~/.jaato/reliability.json` |

The mode can be set and saved in one command:

```
reliability recovery auto save workspace
```

**Defaults:** `auto`

---

## reset

Reset a specific tool (or all tools) back to the `TRUSTED` state, clearing failure history and escalation.

```
reliability reset <failure_key>
reliability reset all
```

| Argument | Description |
|----------|-------------|
| `<failure_key>` | The failure key identifying the tool+parameters combination (tab-completable from currently escalated tools) |
| `all` | Reset every tracked tool to trusted |

**Failure key format:** `{tool_name}` or `{tool_name}|{parameter_signature}`

Examples:
- `readFile|path_prefix=/etc`
- `http_request|domain=api.example.com`
- `bash|command=npm`

Tab completion offers only tools currently in the `ESCALATED` or `BLOCKED` state.

---

## history

Show recent failure records.

```
reliability history
reliability history limit <N>
```

| Argument | Description |
|----------|-------------|
| *(none)* | Show the last 10 failures |
| `limit <N>` | Show the last N failures |

Each entry shows timestamp, failure key, severity, and a truncated error message.

**Example output:**

```
Recent failures (last 3):
  [14:32:01] bash|command=npm (server_error): npm ERR! code ENOENT
  [14:31:45] http_request|domain=api.github.com (timeout): deadline exceeded
  [14:30:22] readFile|path_prefix=/tmp (not_found): No such file or directory
```

---

## config

Show the current escalation rule configuration (read-only).

```
reliability config
```

Displays:
- **Count threshold** — failures in the window to trigger escalation
- **Window** — time window in seconds
- **Escalation duration** — how long escalation lasts
- **Recovery successes needed** — consecutive successes to recover
- **Auto recovery** — whether auto-recovery is enabled
- **Max history** — maximum failure records retained

---

## settings

View, save, or clear reliability **runtime settings** across persistence levels.

> **Interacts with `policies`.** Settings and policies use different files but share the nudge domain: policies define *what* nudges say and *when* patterns fire, while settings control *which* nudge types are actually delivered. See [Cross-File Interactions](#cross-file-interactions).

```
reliability settings
reliability settings show
reliability settings save workspace|user
reliability settings clear workspace|session
```

| Subcommand | Description |
|------------|-------------|
| `show` *(default)* | Display all effective settings with their source (session override, workspace, user default, or built-in default) |
| `save workspace` | Save current session settings to `.jaato/reliability.json` |
| `save user` | Save current session settings to `~/.jaato/reliability.json` |
| `clear workspace` | Clear workspace-level settings; inherits from user defaults |
| `clear session` | Clear session-level overrides; inherits from workspace/user |

### Settings Precedence

Settings are resolved in this order (highest to lowest priority):

1. **Session override** — set during the current session via commands
2. **Workspace** — `.jaato/reliability.json` in the project root
3. **User default** — `~/.jaato/reliability.json` in the home directory
4. **Built-in default** — hardcoded in the plugin

### Tracked Settings

| Setting | Values | Default |
|---------|--------|---------|
| `recovery_mode` | `auto`, `ask` | `auto` |
| `nudge_level` | `off`, `gentle`, `direct`, `full` | `direct` |
| `nudge_enabled` | `True`, `False` | `True` |
| `model_switch_strategy` | `disabled`, `suggest`, `auto` | `suggest` |

---

## model

Manage model-specific reliability tracking and model switching behavior.

```
reliability model
reliability model status [<model>]
reliability model compare
reliability model suggest [save workspace|user]
reliability model auto [save workspace|user]
reliability model disabled [save workspace|user]
```

### model status

```
reliability model status
reliability model status <model_name>
```

Show reliability summary for a specific model or the current model. Displays total attempts, failures, success rate, tracked tools, and problematic tools (below 70% success rate). Also shows the current model switching strategy.

### model compare

```
reliability model compare
```

Show a comparison table of all models with tracked reliability data. Columns: model name, success rate, total attempts, tool count. Marks the current model.

### model suggest

```
reliability model suggest
reliability model suggest save workspace|user
```

Set model switching strategy to **SUGGEST**: the plugin notifies the user when a different model has a better track record for a failing tool.

### model auto

```
reliability model auto
reliability model auto save workspace|user
```

Set model switching strategy to **AUTO**: the plugin automatically switches to a better-performing model when failure thresholds are exceeded.

### model disabled

```
reliability model disabled
reliability model disabled save workspace|user
```

Disable model switching entirely. No suggestions or automatic switches.

### Model Switching Strategies

| Strategy | Behavior |
|----------|----------|
| `disabled` | Never suggest or perform model switches |
| `suggest` | Notify user when a switch might improve reliability |
| `auto` | Automatically switch to better-performing model |

**Strategy defaults:** `suggest`

The `save` argument persists the strategy to workspace or user level (same semantics as `reliability settings save`).

---

## patterns

Manage behavioral pattern detection — the system that identifies when models get stuck in unproductive loops.

```
reliability patterns
reliability patterns status
reliability patterns enable
reliability patterns disable
reliability patterns history [<N>]
reliability patterns clear
```

### patterns status

Show whether pattern detection is enabled, the active detection thresholds, total patterns detected, and breakdowns by pattern type and severity.

### patterns enable / disable

Toggle behavioral pattern detection on or off for the current session.

### patterns history

```
reliability patterns history
reliability patterns history <N>
```

Show the last N detected patterns (default: 10). Each entry includes timestamp, pattern type, severity, repetition count, the recent tool sequence, and a suggested corrective action.

### patterns clear

Clear the accumulated pattern detection history.

### Detected Pattern Types

| Pattern | Description |
|---------|-------------|
| `repetitive_calls` | Same tool called N consecutive times regardless of success |
| `error_retry_loop` | Same tool retried after failure with unchanged arguments |
| `introspection_loop` | Stuck calling `list_tools`/`get_tool_schemas` without acting |
| `announce_no_action` | Model announces it will act ("let me...") but only reads |
| `read_only_loop` | Only read-only tool calls, no mutations |
| `planning_loop` | Infinite planning without execution |
| `tool_avoidance` | Model avoids a specific tool repeatedly |
| `prerequisite_violated` | A prerequisite policy was violated |

### Pattern Severity

| Severity | Meaning | Default Intervention |
|----------|---------|---------------------|
| `minor` | 2-3 repetitions, just starting | Gentle reminder |
| `moderate` | 4-5 repetitions, clear stall | Direct instruction |
| `severe` | 6+ repetitions, intervention needed | Interrupt (block) |

Severity escalates automatically as the same pattern repeats within a session.

---

## nudge

Control the intensity of nudge injection — the mechanism that injects guidance into the model's context when patterns are detected.

```
reliability nudge
reliability nudge status
reliability nudge off
reliability nudge gentle
reliability nudge direct
reliability nudge full
reliability nudge history [<N>]
```

### nudge status

Show whether nudge injection is enabled, the current level, cooldown interval, total nudges sent, effectiveness statistics, and breakdown by nudge type.

### nudge off / gentle / direct / full

Set the nudge injection level for the current session.

| Level | Description |
|-------|-------------|
| `off` | No nudges injected |
| `gentle` | Only soft reminders (low urgency) |
| `direct` | Gentle reminders + clear instructions (moderate urgency) |
| `full` | All nudges including interrupts (blocks execution for user input) |

**Default level:** `direct`

### nudge history

```
reliability nudge history
reliability nudge history <N>
```

Show the last N nudges (default: 10). Each entry shows timestamp, nudge type, the triggering pattern, a truncated message, and effectiveness markers.

### Nudge Types

| Type | System Message Prefix | Urgency |
|------|----------------------|---------|
| `gentle` | `[Reminder]` | Low — soft suggestion |
| `direct` | `[NOTICE]` | Moderate — clear instruction |
| `interrupt` | `[SYSTEM INTERRUPT]` | High — blocks execution |

### Nudge Lifecycle

1. Pattern detected by the pattern detector
2. Nudge injected as system message into model context
3. Tracked as `acknowledged` if the model changes behavior
4. Tracked as `effective` if the pattern stops after the nudge

---

## policies

Manage file-based **prerequisite policies** and **pattern detection thresholds** — rules that enforce tool ordering (e.g., "read before edit") and tune loop detection sensitivity.

> **Interacts with `settings` and `nudge`.** Policies define detection thresholds and nudge message templates, but the `nudge_level` setting (managed via `reliability nudge` or `reliability settings`) acts as a global filter that can suppress nudges defined here. See [Cross-File Interactions](#cross-file-interactions).

```
reliability policies
reliability policies status
reliability policies reload
reliability policies edit
reliability policies path
```

### policies status

Show the active config file path, number of loaded policies, any load warnings, and details for each active policy (prerequisite tool, gated tools, lookback window, severity escalation, nudge levels). Also shows pattern detection overrides loaded from the config file.

### policies reload

Re-read and validate the policy config file. Reports the number of policies loaded and any warnings.

### policies edit

Open the policy config file in `$EDITOR` (or `$VISUAL`, falls back to `vi`). If no config file exists, creates one at `.jaato/reliability-policies.json` with annotated defaults. After the editor exits, automatically validates and reloads the file.

### policies path

Show the resolved config file path. If no file exists, shows the search locations and suggests using `policies edit` to create one.

### Config File Locations

Searched in order (first found wins):

1. `.jaato/reliability-policies.json` — workspace-level, per-project
2. `~/.jaato/reliability-policies.json` — user-level, global default

### Config File Schema

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

Every field is optional. Missing fields retain built-in defaults.

### Prerequisite Policy Fields

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `policy_id` | yes | string | — | Unique identifier |
| `prerequisite_tool` | yes | string | — | Tool that must be called first |
| `gated_tools` | yes | string[] | — | Tools that require the prerequisite |
| `lookback_turns` | no | int | `2` | How many previous turns to check |
| `severity_thresholds` | no | object | `{"minor": 0, "moderate": 1, "severe": 2}` | Maps severity to minimum prior violation count |
| `nudge_templates` | no | object | *(built-in)* | Custom nudge messages per severity |
| `expected_action_template` | no | string | `"Call {prerequisite_tool} before using {tool_name}"` | Template for the expected action field |

### severity_thresholds

Maps severity name to the minimum number of *prior* violations needed to reach that level. The detector picks the highest severity whose threshold is met.

| Pattern | Thresholds | Behavior |
|---------|-----------|----------|
| Immediate block | `{"minor": 0, "moderate": 0, "severe": 0}` | First violation interrupts |
| Warn then block | `{"minor": 0, "moderate": 0, "severe": 1}` | First: direct instruction, second: block |
| Standard (default) | `{"minor": 0, "moderate": 1, "severe": 2}` | Gentle, then direct, then block over 3 violations |
| Lenient | `{"minor": 0, "moderate": 3, "severe": 6}` | 3 gentle, 3 direct, then block |
| Advisory only | `{"minor": 0}` | Only gentle reminders, never escalates |

### nudge_templates

Custom messages per severity. Each entry maps a severity name to a `[nudge_type, message]` pair.

Nudge types: `"gentle"`, `"direct"`, `"interrupt"`

Template variables: `{tool_name}`, `{prerequisite_tool}`, `{count}`

See [Reliability Policies Configuration Guide](reliability-policies-config.md) for full examples.

---

## behavior

Analyze model behavioral profiles — aggregated statistics about how each model behaves and responds to nudges.

```
reliability behavior
reliability behavior status [<model>]
reliability behavior compare <model1> [<model2>]
reliability behavior patterns [<model>]
```

### behavior status

```
reliability behavior status
reliability behavior status <model_name>
```

Without a model name: shows an aggregate summary across all tracked models (models tracked, total turns, overall stall rate, total patterns, overall nudge effectiveness, per-model summary).

With a model name: shows detailed profile for that model (total turns, stalled turns, stall rate, total patterns, most common pattern, nudges sent, nudge effectiveness, first/last seen timestamps).

### behavior compare

```
reliability behavior compare <model1> [<model2>]
```

Compare behavioral profiles of two models. If only one model is given, compares against the current model. Shows stall rate, nudge effectiveness, and pattern count for each, with a recommendation of which model appears better behaved.

### behavior patterns

```
reliability behavior patterns
reliability behavior patterns <model_name>
```

Show pattern type breakdown by model. For each model, lists every detected pattern type with its count and average severity.

---

## Concepts

### Configuration Files

The reliability plugin uses **two separate JSON files**, each at workspace and user levels. Understanding which file controls what avoids confusion between the `settings` and `policies` subcommands.

| File | Managed By | Content |
|------|-----------|---------|
| `.jaato/reliability.json` | `reliability settings` | Runtime toggles: `recovery_mode`, `nudge_level`, `nudge_enabled`, `model_switch_strategy`. Also stores persisted tool states and failure history. |
| `.jaato/reliability-policies.json` | `reliability policies` | Pattern detection thresholds (repetitive call limits, retry limits, read-only loop limits) and prerequisite policy rules (tool ordering constraints). |

Both files exist at two levels (workspace wins over user):

```
Workspace:  .jaato/reliability.json             ← project-specific runtime settings
            .jaato/reliability-policies.json     ← project-specific detection rules

User:       ~/.jaato/reliability.json            ← personal default settings
            ~/.jaato/reliability-policies.json   ← personal baseline detection rules
```

**Why two files?** Settings are simple key-value toggles that change often during a session (`reliability nudge full`, `reliability recovery ask`). Policies are structured rules with thresholds and templates that are typically authored once and rarely change at runtime — hence the `edit` subcommand opening `$EDITOR`.

### Cross-File Interactions

The nudge system spans both files. Understanding the interaction prevents surprises where configured nudges are silently suppressed or fire unexpectedly.

```
reliability-policies.json                reliability.json
┌──────────────────────────────┐         ┌──────────────────────┐
│ pattern_detection:           │         │                      │
│   thresholds that DETECT     │──fires──│                      │
│   patterns                   │ pattern │ nudge_level           │
│                              │         │ (off/gentle/direct/   │
│ prerequisite_policies:       │         │  full)                │
│   nudge_templates that       │         │                      │
│   define WHAT to say and     │─────┐   │ Acts as global gate: │
│   at what severity           │     │   │ only nudge types at  │
│                              │     │   │ or below the level   │
└──────────────────────────────┘     │   │ are delivered         │
                                     │   └──────────┬───────────┘
                                     │              │
                                     ▼              ▼
                               NudgeStrategy.should_inject()
                                     │
                          ┌──────────┴──────────┐
                          │ Level allows type?   │
                          │ gentle → gentle only │
                          │ direct → gentle+direct│
                          │ full   → all         │
                          │ off    → none        │
                          └──────────┬──────────┘
                                     │
                              YES ───┼─── NO
                                     │      │
                              delivered   silently
                                          dropped
```

**Concrete scenario:** A prerequisite policy in `reliability-policies.json` defines `"severe": ["interrupt", "BLOCKED: ..."]`. The user then runs `reliability nudge gentle`. Now severe prerequisite violations are detected and the pattern fires, but the interrupt nudge is silently filtered out — only gentle reminders are delivered. The policy's escalation logic still advances the violation count, so when the user later runs `reliability nudge full`, the next violation immediately produces an interrupt (because the count already reached the severe threshold).

**What lives where — full breakdown:**

| Concern | Configured In | Managed By | Notes |
|---------|--------------|-----------|-------|
| When to detect a pattern | `reliability-policies.json` | `policies` | Thresholds like `error_retry_threshold` |
| What message to show | `reliability-policies.json` | `policies` | `nudge_templates` on prerequisite policies |
| Built-in pattern messages | Hardcoded in `nudge.py` | — | Default templates for `repetitive_calls`, `read_only_loop`, etc. |
| Which nudge types to deliver | `reliability.json` | `settings` / `nudge` | `nudge_level` acts as global filter |
| Whether nudges are enabled at all | `reliability.json` | `settings` | `nudge_enabled` toggle |
| Whether detection is active | Session-only | `patterns enable/disable` | Not persisted to either file |
| Recovery behavior | `reliability.json` | `settings` / `recovery` | `recovery_mode` (auto/ask) |
| Model switching strategy | `reliability.json` | `settings` / `model` | `model_switch_strategy` |

### Trust States

The reliability plugin tracks each tool+parameters combination through a state machine:

```
                 ┌────────────────────────────┐
                 │                            │
                 ▼                            │
TRUSTED ──(threshold exceeded)──► ESCALATED ──(reset)──►┘
   ▲                                 │
   │                                 │
   └──(N successes)──► RECOVERING ◄──(auto/ask recovery)
                          │
                          └──(failure)──► ESCALATED

BLOCKED ◄──(critical failure)──── TRUSTED/ESCALATED/RECOVERING
   │
   └──(manual reset only)──► TRUSTED
```

| State | Behavior |
|-------|----------|
| `TRUSTED` | Normal operation; standard permission rules apply |
| `ESCALATED` | Forces explicit user approval regardless of whitelist rules |
| `RECOVERING` | Tracking consecutive successes; still requires approval |
| `BLOCKED` | Permanent block for critical/security failures; manual reset only |

### Failure Keys

Tool failures are tracked by a composite key: `{tool_name}|{parameter_signature}`. The parameter signature is extracted from invocation arguments to distinguish failures by context:

| Tool Category | Parameter Extracted | Example Key |
|---------------|-------------------|-------------|
| File operations (`readFile`, `Write`, `Edit`) | Parent directory path | `readFile\|path_prefix=/etc` |
| HTTP requests (`http_request`, `WebFetch`) | Domain + first path segment | `http_request\|domain=api.github.com\|path_prefix=repos` |
| CLI commands (`bash`, `Bash`) | First command word | `bash\|command=npm` |
| MCP tools | Server name | `mcp_query\|mcp_server=Atlassian` |
| Search tools (`Grep`, `Glob`) | Search path | `Grep\|path_prefix=/src/components` |
| Other tools | Tool name only | `list_tools` |

### Failure Severity

Failures are classified by severity, which affects escalation weight:

| Level | Severities | Weight |
|-------|-----------|--------|
| Low | `transient`, `not_found`, `invalid_input` | 0.5 (model errors) to 1.0 |
| Medium | `permission`, `validation`, `timeout` | 1.0 |
| High | `server_error`, `crash`, `corruption` | 1.0 |
| Critical | `security`, `repeated_auth` | 1.0 (immediate escalation) |

### Escalation Rules

Configurable thresholds that determine when tools transition from `TRUSTED` to `ESCALATED`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Count threshold | 3 | Weighted failures in the time window |
| Window | 3600s (1 hour) | Time window for counting failures |
| Escalation duration | 1800s (30 min) | How long escalation lasts |
| Cooldown | 900s (15 min) | Time without failures before auto-recovery starts |
| Successes to recover | 3 | Consecutive successes needed in recovery state |

Rules can be set per-tool, per-plugin, per-domain, or globally (in decreasing priority).

---

## Permission Integration

The reliability plugin wraps the existing permission plugin via `ReliabilityPermissionWrapper`. When a tool is in `ESCALATED`, `RECOVERING`, or `BLOCKED` state, the wrapper overrides any auto-approval (whitelist) and forces an explicit permission prompt. The prompt includes escalation details:

- Severity label (ESCALATED / BLOCKED / RECOVERING)
- Escalation reason
- Failure history window
- Recovery progress (if recovering)

---

## Tab Completion

All subcommands and arguments support tab completion:

| Level | Context | Completions |
|-------|---------|-------------|
| 1 | `reliability <TAB>` | `status`, `recovery`, `reset`, `history`, `config`, `settings`, `model`, `patterns`, `policies`, `nudge`, `behavior` |
| 2 | `reliability recovery <TAB>` | `auto`, `ask`, `save` |
| 2 | `reliability reset <TAB>` | Currently escalated/blocked failure keys |
| 2 | `reliability settings <TAB>` | `show`, `save`, `clear` |
| 2 | `reliability model <TAB>` | `status`, `compare`, `suggest`, `auto`, `disabled` |
| 2 | `reliability patterns <TAB>` | `status`, `enable`, `disable`, `history`, `clear` |
| 2 | `reliability nudge <TAB>` | `status`, `off`, `gentle`, `direct`, `full`, `history` |
| 2 | `reliability policies <TAB>` | `status`, `reload`, `edit`, `path` |
| 2 | `reliability behavior <TAB>` | `status`, `compare`, `patterns` |
| 3 | `reliability recovery save <TAB>` | `workspace`, `user` |
| 3 | `reliability settings save <TAB>` | `workspace`, `user` |
| 3 | `reliability settings clear <TAB>` | `workspace`, `session` |
| 3 | `reliability model status <TAB>` | Known model names |
| 3 | `reliability model suggest <TAB>` | `save` |
| 3 | `reliability model auto <TAB>` | `save` |
| 3 | `reliability model disabled <TAB>` | `save` |
| 3 | `reliability behavior status <TAB>` | Known model names |
| 3 | `reliability behavior compare <TAB>` | Known model names |
| 3 | `reliability behavior patterns <TAB>` | Known model names |
| 4 | `reliability model suggest save <TAB>` | `workspace`, `user` |

---

## Quick Reference

```
reliability                                    # Show status (default)
reliability status                             # Same as above
reliability recovery auto                      # Enable automatic recovery
reliability recovery ask save workspace        # Set manual recovery, save to project
reliability reset all                          # Reset all tools to trusted
reliability reset bash|command=npm             # Reset specific tool
reliability history                            # Last 10 failures
reliability history limit 25                   # Last 25 failures
reliability config                             # Show escalation thresholds
reliability settings show                      # Show settings with sources
reliability settings save workspace            # Save settings to project
reliability settings clear session             # Clear session overrides
reliability model status                       # Current model reliability
reliability model status gemini-2.5-flash      # Specific model reliability
reliability model compare                      # Compare all models
reliability model suggest save workspace       # Enable suggestions, save
reliability model auto                         # Enable auto model switching
reliability model disabled                     # Disable model switching
reliability patterns status                    # Pattern detection status
reliability patterns enable                    # Enable detection
reliability patterns disable                   # Disable detection
reliability patterns history 20                # Last 20 detected patterns
reliability patterns clear                     # Clear pattern history
reliability nudge status                       # Nudge injection status
reliability nudge off                          # Disable nudges
reliability nudge gentle                       # Soft reminders only
reliability nudge direct                       # Reminders + instructions
reliability nudge full                         # All nudges + interrupts
reliability nudge history                      # Last 10 nudges
reliability policies status                    # Show loaded policies
reliability policies reload                    # Reload from config file
reliability policies edit                      # Open config in $EDITOR
reliability policies path                      # Show config file path
reliability behavior status                    # All-model summary
reliability behavior status gemini-2.5-flash   # Single model profile
reliability behavior compare model1 model2     # Compare two models
reliability behavior patterns                  # Pattern breakdown by model
```

---

## Related Documentation

- [Reliability Plugin Design](reliability-plugin-design.md) — Architecture, failure recording, trust state machine, integration points
- [Reliability Policies Configuration](reliability-policies-config.md) — JSON schema, prerequisite policies, per-tool thresholds, usage examples
- [Permission System](jaato_permission_system.md) — Permission responses, scope semantics, evaluation order
