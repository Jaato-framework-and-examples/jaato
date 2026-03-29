# Permission Evaluators

Permission evaluators are Python scripts that run at permission-check time, letting you add dynamic logic to jaato's declarative permission system. They sit between sanitization checks and the standard blacklist/whitelist pipeline.

## When to use evaluators

Use evaluators when static rules (patterns, globs, blacklists) aren't enough:

- Time-based restrictions (no deployments after hours)
- Argument combination checks (allow `git push` but not `git push --force`)
- External policy service calls (check a compliance API)
- Session history conditions (block destructive commands after errors)
- Environment-aware rules (stricter in production workspaces)

For simple allow/deny rules, stick with the declarative `whitelist`/`blacklist` in `permissions.json`.

## Quick start

### 1. Write an evaluator script

Create `.jaato/policies/cli_guard.py`:

```python
from shared.plugins.permission.evaluator import PolicyDecision

def evaluate(tool_name, args, context):
    command = args.get("command", "")

    # Block force pushes
    if "push" in command and "--force" in command:
        return PolicyDecision.DENY

    # Block rm -rf with absolute paths
    if command.startswith("rm -rf /"):
        return PolicyDecision.DENY

    # Everything else: defer to standard policy
    return PolicyDecision.FALLBACK
```

### 2. Reference it in permission config

In your profile's `plugin_configs`, or in `.jaato/permissions.json`:

```json
{
  "defaultPolicy": "ask",
  "evaluators": {
    "cli_based_tool": "policies/cli_guard.py"
  }
}
```

Or in a profile:

```json
{
  "name": "developer",
  "plugins": ["cli", "file_edit", "permission"],
  "plugin_configs": {
    "permission": {
      "evaluators": {
        "cli_based_tool": "policies/cli_guard.py"
      }
    }
  }
}
```

### 3. Done

The evaluator runs every time `cli_based_tool` is checked for permission. If it returns `DENY`, the tool is blocked. If it returns `FALLBACK`, the standard blacklist/whitelist/default policy applies.

## The evaluate function

Every evaluator script must define an `evaluate` function:

```python
def evaluate(tool_name: str, args: dict, context: EvalContext) -> PolicyDecision:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool being checked (e.g. `cli_based_tool`, `writeNewFile`) |
| `args` | `dict` | Arguments the model passed to the tool |
| `context` | `EvalContext` | Agent and session context (see below) |

### EvalContext fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Same as the `tool_name` parameter |
| `args` | `dict` | Same as the `args` parameter |
| `agent_type` | `str` | `"main"` or `"subagent"` |
| `agent_name` | `str` or `None` | Agent/profile name |
| `session_id` | `str` or `None` | Daemon session manager ID |
| `workspace_path` | `str` or `None` | Workspace directory path |
| `extra` | `dict` | Extensible dict for future fields |

The context carries information the script can't obtain on its own. For anything else (current time, environment variables, file checks), import stdlib modules directly.

## Return values

Evaluators return a `PolicyDecision` indicating what should happen. These mirror the options available at the interactive permission prompt:

### Simple decisions

| Decision | Effect | Equivalent to |
|----------|--------|---------------|
| `PolicyDecision.ALLOW` | Permit this tool call | User pressing `y` |
| `PolicyDecision.DENY` | Block this tool call | User pressing `n` |
| `PolicyDecision.FALLBACK` | Defer to standard policy pipeline | No evaluator |

### Scoped decisions

| Decision | Effect | Equivalent to |
|----------|--------|---------------|
| `PolicyDecision.ALLOW_ONCE` | Permit but don't remember | User pressing `once` |
| `PolicyDecision.ALLOW_TURN` | Permit all tools for the rest of this turn | User pressing `t` |
| `PolicyDecision.ALLOW_UNTIL_IDLE` | Permit until session goes idle | User pressing `i` |
| `PolicyDecision.ALLOW_SESSION` | Add to session whitelist | User pressing `a`/`always` |
| `PolicyDecision.ALLOW_ALL` | Pre-approve all future requests this session | User pressing `all` |
| `PolicyDecision.DENY_SESSION` | Add to session blacklist | User pressing `never` |

### Deny with comment

To deny and pass a message the model can see (like typing `c:message` at the prompt):

```python
from shared.plugins.permission.evaluator import PolicyDecision, EvalResult

def evaluate(tool_name, args, context):
    if risky_condition:
        return EvalResult(
            PolicyDecision.DENY_WITH_COMMENT,
            comment="Use 'git stash' before this operation"
        )
    return PolicyDecision.FALLBACK
```

The comment appears in the tool error and the model can read and act on it.

### Return format flexibility

Evaluators can return any of these formats:

```python
# PolicyDecision enum
return PolicyDecision.ALLOW

# EvalResult (for deny-with-comment or explicit structure)
return EvalResult(PolicyDecision.DENY_WITH_COMMENT, comment="reason here")

# Plain string
return "allow"
return "deny"
return "fallback"

# Tuple (decision, comment) for deny-with-comment shorthand
return (PolicyDecision.DENY_WITH_COMMENT, "Use a safer approach")
return ("deny_with_comment", "Use a safer approach")
```

## Path resolution

Evaluator paths in the config are resolved in this order:

1. **Absolute path** -- used directly
2. **Workspace** -- `{workspace}/.jaato/{path}`
3. **User home** -- `~/.jaato/{path}`

This follows the same precedence as profiles and other jaato configs.

## Tool-specific vs default evaluators

```json
{
  "evaluators": {
    "default": "policies/global.py",
    "cli_based_tool": "policies/cli.py",
    "writeNewFile": "policies/file_write.py"
  }
}
```

- Tool-specific evaluators run first (exact match on tool name)
- If no tool-specific evaluator exists, the `default` evaluator runs
- If neither exists, the evaluator step is skipped entirely

## Evaluation order

Evaluators run after sanitization but before the blacklist/whitelist pipeline:

```
1. Sanitization checks         -> DENY if violations
2. Evaluator (this feature)    -> ALLOW/DENY/FALLBACK
3. Session blacklist            -> DENY if matched
4. Static blacklist             -> DENY if matched
5. Session whitelist            -> ALLOW if matched
6. Static whitelist             -> ALLOW if matched
7. Default policy               -> allow/deny/ask
```

If an evaluator returns `ALLOW` or `DENY`, the remaining steps are skipped. If it returns `FALLBACK`, evaluation continues from step 3.

## Error handling

Evaluators are fail-safe:

- **Script not found** -- logged warning, treated as FALLBACK
- **No `evaluate` function** -- logged warning, treated as FALLBACK
- **Exception during execution** -- logged warning, treated as FALLBACK
- **Invalid return value** -- logged warning, treated as FALLBACK

An evaluator bug will never crash the server or silently block tools.

## Examples

### Time-based restrictions

```python
from datetime import datetime
from shared.plugins.permission.evaluator import PolicyDecision

def evaluate(tool_name, args, context):
    hour = datetime.now().hour
    if hour < 6 or hour > 22:
        return PolicyDecision.DENY
    return PolicyDecision.FALLBACK
```

### Subagent restrictions

```python
from shared.plugins.permission.evaluator import PolicyDecision

def evaluate(tool_name, args, context):
    # Subagents can only read, not write
    if context.agent_type == "subagent":
        return PolicyDecision.DENY
    return PolicyDecision.FALLBACK
```

### Argument inspection

```python
from shared.plugins.permission.evaluator import PolicyDecision, EvalResult

def evaluate(tool_name, args, context):
    path = args.get("path", "")

    # Block writes outside the workspace
    if not path.startswith(context.workspace_path or ""):
        return EvalResult(
            PolicyDecision.DENY_WITH_COMMENT,
            comment=f"Cannot write outside workspace. Use a path under {context.workspace_path}"
        )

    return PolicyDecision.FALLBACK
```

### Auto-approve safe patterns

```python
from shared.plugins.permission.evaluator import PolicyDecision

SAFE_COMMANDS = {"git status", "git diff", "git log", "npm test", "pytest"}

def evaluate(tool_name, args, context):
    command = args.get("command", "").strip()
    if command in SAFE_COMMANDS:
        return PolicyDecision.ALLOW
    return PolicyDecision.FALLBACK
```

### External policy service

```python
import requests
from shared.plugins.permission.evaluator import PolicyDecision

def evaluate(tool_name, args, context):
    try:
        resp = requests.post("https://policy.internal/check", json={
            "tool": tool_name,
            "args": args,
            "agent": context.agent_name,
            "session": context.session_id,
        }, timeout=2)
        if resp.json().get("allowed"):
            return PolicyDecision.ALLOW
        return PolicyDecision.DENY
    except Exception:
        # Policy service down -- fall back to local rules
        return PolicyDecision.FALLBACK
```
