# JAATO Tool Call Permission System

## Executive Summary

JAATO implements a comprehensive permission system that gates tool execution, ensuring models cannot perform sensitive operations without explicit approval. The system supports multiple approval scopes (single use, turn, session), configurable policies (whitelist/blacklist), and various approval channels (console, webhook, file-based). Permissions are evaluated in a deterministic order, with blacklists taking priority over whitelists, and session rules overriding static configuration.

---

## Part 1: Permission Responses

When a tool requires permission, the user can respond with various approval scopes:

### Response Options

| Short | Full | Decision | Scope | Behavior |
|-------|------|----------|-------|----------|
| `y` | `yes` | ALLOW | Single | Execute this tool call once |
| `n` | `no` | DENY | Single | Block this execution |
| `once` | `once` | ALLOW_ONCE | Single | Execute without remembering |
| `a` | `always` | ALLOW_SESSION | Session | Add tool to session whitelist |
| `never` | `never` | DENY_SESSION | Session | Add tool to session blacklist |
| `t` | `turn` | ALLOW_TURN | Turn | Allow all remaining tools this turn |
| `i` | `idle` | ALLOW_UNTIL_IDLE | Idle | Allow until session goes idle |
| `all` | `all` | ALLOW_ALL | Session | Pre-approve all future requests |

### Scope Semantics

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERMISSION SCOPE HIERARCHY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  NARROWEST ◄─────────────────────────────────────────────► WIDEST   │
│                                                                      │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌─────────┐   │
│  │  once  │ < │  yes   │ < │  turn  │ < │  idle  │ < │   all   │   │
│  │        │   │        │   │        │   │        │   │         │   │
│  │ Single │   │ Single │   │ Until  │   │ Until  │   │ Session │   │
│  │  call  │   │ + learn│   │ turn   │   │ idle   │   │  wide   │   │
│  │        │   │        │   │ ends   │   │        │   │         │   │
│  └────────┘   └────────┘   └────────┘   └────────┘   └─────────┘   │
│                                                                      │
│  ┌────────┐                              ┌────────┐                 │
│  │   no   │              VS              │ never  │                 │
│  │        │                              │        │                 │
│  │ Single │                              │ Session│                 │
│  │ denial │                              │ block  │                 │
│  └────────┘                              └────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Turn vs Idle Explained

**Turn Scope (`t`):**
- Approves all tools until the model finishes its current response
- Clears automatically when the turn ends
- Useful for: "I trust this sequence of operations"

**Idle Scope (`i`):**
- Approves all tools until the session becomes idle (no user input, no model activity)
- Persists across multiple consecutive model turns
- Useful for: "Let it run until I interact again"

```
User: "Refactor the authentication module"
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Turn 1: Model reads files, proposes changes                      │
│  ├─ readFile → auto-approved (read-only)                         │
│  ├─ updateFile → user responds 't' (turn approval)               │
│  ├─ updateFile → allowed (turn suspension active)                │
│  └─ updateFile → allowed (turn suspension active)                │
│                                                                   │
│  Turn 1 ends → Turn suspension CLEARS                            │
├──────────────────────────────────────────────────────────────────┤
│  Turn 2: Model continues (still processing)                       │
│  ├─ updateFile → needs NEW permission                            │
│  └─ ...                                                          │
└──────────────────────────────────────────────────────────────────┘

vs.

User: "Refactor the authentication module"
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Turn 1: Model reads files, proposes changes                      │
│  ├─ readFile → auto-approved (read-only)                         │
│  ├─ updateFile → user responds 'i' (idle approval)               │
│  ├─ updateFile → allowed (idle suspension active)                │
│  └─ updateFile → allowed (idle suspension active)                │
│                                                                   │
│  Turn 1 ends → Idle suspension PERSISTS                          │
├──────────────────────────────────────────────────────────────────┤
│  Turn 2: Model continues (still processing)                       │
│  ├─ updateFile → allowed (idle suspension still active)          │
│  └─ ...                                                          │
│                                                                   │
│  Session goes IDLE → Idle suspension CLEARS                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Policy Evaluation Order

Permission decisions follow a strict, deterministic evaluation order:

### Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    POLICY EVALUATION ORDER                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Tool Call: updateFile(path="src/main.py", content="...")           │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  1. SUSPENSION CHECK                                     │        │
│  │     ├─ _idle_suspended? → ALLOW                         │        │
│  │     ├─ _turn_suspended? → ALLOW                         │        │
│  │     └─ _allow_all?      → ALLOW                         │        │
│  └────────┬────────────────────────────────────────────────┘        │
│           │ (if not suspended)                                       │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  2. SANITIZATION (if enabled)                            │        │
│  │     ├─ Shell injection detected?      → DENY            │        │
│  │     ├─ Dangerous command patterns?    → DENY            │        │
│  │     └─ Path outside allowed scope?    → DENY            │        │
│  └────────┬────────────────────────────────────────────────┘        │
│           │ (if no violations)                                       │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  3. SESSION BLACKLIST                                    │        │
│  │     └─ Tool in session_blacklist?     → DENY            │        │
│  └────────┬────────────────────────────────────────────────┘        │
│           │ (if not blacklisted)                                     │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  4. STATIC BLACKLIST (config file)                       │        │
│  │     └─ Tool matches blacklist pattern? → DENY           │        │
│  └────────┬────────────────────────────────────────────────┘        │
│           │ (if not blacklisted)                                     │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  5. SESSION WHITELIST                                    │        │
│  │     └─ Tool in session_whitelist?     → ALLOW           │        │
│  └────────┬────────────────────────────────────────────────┘        │
│           │ (if not whitelisted)                                     │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  6. STATIC WHITELIST (config file)                       │        │
│  │     └─ Tool matches whitelist pattern? → ALLOW          │        │
│  └────────┬────────────────────────────────────────────────┘        │
│           │ (if not whitelisted)                                     │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  7. DEFAULT POLICY                                       │        │
│  │     ├─ Session default override?  → Use session default │        │
│  │     └─ Static default             → Use config default  │        │
│  │                                                          │        │
│  │     Values: "allow" | "deny" | "ask"                    │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principle: Blacklist Always Wins

```
┌──────────────────────────────────────────────────────────────────┐
│  BLACKLIST PRIORITY RULE                                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Scenario 1: Conflicting Rules                                    │
│  ├─ Session blacklist: updateFile                                │
│  ├─ Session whitelist: updateFile                                │
│  └─ Result: DENIED (blacklist checked first)                     │
│                                                                   │
│  Scenario 2: Pattern vs Exact                                     │
│  ├─ Session blacklist: "create*" (pattern)                       │
│  ├─ Session whitelist: "createPlan" (exact)                      │
│  └─ Result: createPlan is ALLOWED (exact beats pattern)          │
│             createFile is DENIED (pattern applies)               │
│                                                                   │
│  Scenario 3: Session vs Static                                    │
│  ├─ Static whitelist: run                                        │
│  ├─ Session blacklist: run                                       │
│  └─ Result: DENIED (session blacklist beats static whitelist)    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Auto-Approved Tools

Certain tools bypass the permission system entirely because they are inherently safe.

### What Makes a Tool Auto-Approved?

1. **Read-only operations** - Cannot modify system state
2. **User-initiated actions** - Already explicitly requested by user
3. **Deterministic calculations** - Pure functions with no side effects
4. **Response handlers** - Processing user's response to a question

### Auto-Approved Tools by Plugin

| Plugin | Tools | Reason |
|--------|-------|--------|
| **introspection** | `list_tools`, `get_tool_schemas` | Discovery only |
| **file_edit** | `readFile`, `undoFileChange` | Read-only or reversible |
| **web_search** | `web_search` | Read-only, no system modification |
| **web_fetch** | `webFetch` | Read-only HTTP GET |
| **calculator** | `calculate` | Pure computation |
| **environment** | `get_environment` | Read-only system info |
| **clarification** | `answerClarification` | User response handler |
| **memory** | `addMemory`, `getMemory` | User-controlled storage |
| **permission** | `permissions` | Meta-command for permission mgmt |
| **anthropic_auth** | Auth commands | User-initiated OAuth |
| **github_auth** | Auth commands | User-initiated OAuth |

### How Auto-Approval Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTO-APPROVAL FLOW                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Plugin Registration                                              │
│     ┌────────────────────────────────────────────────────┐          │
│     │  class FileEditPlugin:                              │          │
│     │      def get_auto_approved_tools(self):             │          │
│     │          return ["readFile", "undoFileChange"]      │          │
│     └────────────────────────────────────────────────────┘          │
│                              │                                       │
│                              ▼                                       │
│  2. Registry Collection                                              │
│     ┌────────────────────────────────────────────────────┐          │
│     │  registry.get_auto_approved_tools()                 │          │
│     │  → ["readFile", "undoFileChange", "web_search", ...] │         │
│     └────────────────────────────────────────────────────┘          │
│                              │                                       │
│                              ▼                                       │
│  3. Permission Plugin Whitelisting                                   │
│     ┌────────────────────────────────────────────────────┐          │
│     │  permission_plugin.add_whitelist_tools(auto_approved) │        │
│     │  # Adds to policy.whitelist_tools set               │          │
│     └────────────────────────────────────────────────────┘          │
│                              │                                       │
│                              ▼                                       │
│  4. Tool Execution                                                   │
│     ┌────────────────────────────────────────────────────┐          │
│     │  Tool: readFile(path="config.json")                 │          │
│     │  Policy check: matches whitelist                    │          │
│     │  Result: ALLOW (no prompt shown)                    │          │
│     └────────────────────────────────────────────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Whitelist and Blacklist Management

### Configuration File Structure

Permissions can be configured via `permissions.json`:

```json
{
  "defaultPolicy": "ask",
  "blacklist": {
    "tools": ["dangerous_tool", "rm_rf"],
    "patterns": ["sudo*", "kill*"],
    "arguments": {
      "run": {
        "command": ["rm -rf", "sudo", "chmod 777"]
      }
    }
  },
  "whitelist": {
    "tools": ["git_status", "npm_install"],
    "patterns": ["read*", "list*"]
  }
}
```

### Pattern Matching (Glob-Style)

```
┌──────────────────────────────────────────────────────────────────┐
│  PATTERN EXAMPLES                                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  "read*"     → readFile, readDirectory, readConfig               │
│  "*File"     → readFile, writeFile, updateFile                   │
│  "git_*"     → git_status, git_commit, git_push                  │
│  "run"       → run (exact match only)                            │
│  "*"         → matches everything (use with caution)             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### User Commands

| Command | Effect | Persistence |
|---------|--------|-------------|
| `permissions show` | Display current effective policy | Read-only |
| `permissions allow <pattern>` | Add to session whitelist | Session only |
| `permissions deny <pattern>` | Add to session blacklist | Session only |
| `permissions default <policy>` | Override default (allow/deny/ask) | Session only |
| `permissions check <tool>` | Test what decision a tool would get | Read-only |
| `permissions clear` | Reset all session modifications | Clears state |
| `permissions suspend` | Enable allow-all mode | Session only |
| `permissions resume` | Disable allow-all mode | Session only |
| `permissions status` | Show suspension state | Read-only |

### Session vs Static Rules

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  STATIC (permissions.json)          SESSION (runtime)            │
│  ─────────────────────────          ────────────────             │
│  • Loaded at startup                • Modified during session    │
│  • Persists across sessions         • Lost when session ends     │
│  • Defines base policy              • Overrides static rules     │
│  • Managed by editing file          • Managed via commands       │
│                                                                   │
│  Evaluation: Static ──► Session rules applied on top             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Permission Channels

The permission system is channel-agnostic. Different channels handle the user interaction:

### Channel Types

| Channel | Use Case | Communication |
|---------|----------|---------------|
| **ConsoleChannel** | Interactive terminal | stdin/stdout |
| **QueueChannel** | TUI applications | Queue-based I/O |
| **WebhookChannel** | External approval systems | HTTP POST/response |
| **FileChannel** | Separate approval processes | File-based |
| **ParentBridgedChannel** | Subagent mode | Parent agent forwarding |

### Console Channel Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONSOLE CHANNEL INTERACTION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  🔒 Permission Required: updateFile                          │    │
│  │                                                              │    │
│  │  File: src/auth/login.py                                    │    │
│  │                                                              │    │
│  │  ┌────────────────────────────────────────────────────────┐ │    │
│  │  │  @@ -15,7 +15,8 @@                                      │ │    │
│  │  │   def authenticate(username, password):                 │ │    │
│  │  │  -    return check_credentials(username, password)      │ │    │
│  │  │  +    result = check_credentials(username, password)    │ │    │
│  │  │  +    log_attempt(username, result)                     │ │    │
│  │  │  +    return result                                     │ │    │
│  │  └────────────────────────────────────────────────────────┘ │    │
│  │                                                              │    │
│  │  [y]es  [n]o  [a]lways  [never]  [t]urn  [i]dle  [all]      │    │
│  │  > _                                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Webhook Channel Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WEBHOOK CHANNEL FLOW                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  JAATO                              External Approval System         │
│    │                                         │                       │
│    │  POST /approve                          │                       │
│    │  {                                      │                       │
│    │    "request_id": "abc123",              │                       │
│    │    "tool_name": "run",                  │                       │
│    │    "tool_args": {"command": "npm i"},   │                       │
│    │    "context": {...}                     │                       │
│    │  }                                      │                       │
│    │ ─────────────────────────────────────► │                       │
│    │                                         │                       │
│    │         (External review process)       │                       │
│    │                                         │                       │
│    │  Response:                              │                       │
│    │  {                                      │                       │
│    │    "request_id": "abc123",              │                       │
│    │    "decision": "allow",                 │                       │
│    │    "reason": "Approved by admin",       │                       │
│    │    "approver": "alice@example.com"      │  (optional, #859)     │
│    │  }                                      │                       │
│    │ ◄───────────────────────────────────── │                       │
│    │                                         │                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Subagent Channel (ParentBridgedChannel)

When subagents need permissions, they route through their parent:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUBAGENT PERMISSION ROUTING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────┐  │
│  │   SUBAGENT   │          │    PARENT    │          │   USER   │  │
│  │              │          │              │          │          │  │
│  │  updateFile  │          │              │          │          │  │
│  │      │       │          │              │          │          │  │
│  │      ▼       │          │              │          │          │  │
│  │  Permission  │ ──XML──► │  Receives    │          │          │  │
│  │  request via │ message  │  request     │ ──────►  │  Prompt  │  │
│  │  thread-local│          │              │          │          │  │
│  │  channel     │          │              │ ◄──────  │  Response│  │
│  │              │ ◄─────── │  Forwards    │          │          │  │
│  │      ▼       │ response │  response    │          │          │  │
│  │  Continue    │          │              │          │          │  │
│  │  execution   │          │              │          │          │  │
│  └──────────────┘          └──────────────┘          └──────────┘  │
│                                                                      │
│  Thread isolation ensures subagent doesn't affect parent's channel  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Permission Display Formatting

Plugins can provide rich permission displays tailored to their tools.

### PermissionDisplayInfo Structure

```python
@dataclass
class PermissionDisplayInfo:
    summary: str                          # "Update file: src/main.py"
    details: str                          # Full diff content
    format_hint: str = "text"             # "diff", "json", "text", "code"
    language: Optional[str] = None        # For syntax highlighting
    truncated: bool = False               # Content was truncated
    original_lines: Optional[int] = None  # Pre-truncation line count
    warnings: Optional[str] = None        # Security warnings
    warning_level: Optional[str] = None   # "info", "warning", "error"
```

### Plugin Integration

```python
class FileEditPlugin:
    def format_permission_request(
        self,
        tool_name: str,
        args: Dict[str, Any],
        channel_type: str
    ) -> Optional[PermissionDisplayInfo]:

        if tool_name == "updateFile":
            # Generate unified diff
            diff = generate_diff(args['path'], args['content'])

            return PermissionDisplayInfo(
                summary=f"Update file: {args['path']}",
                details=diff,
                format_hint="diff",
                language="python" if args['path'].endswith('.py') else None,
                truncated=len(diff.splitlines()) > 100,
                original_lines=len(diff.splitlines())
            )

        return None  # Use default formatting
```

### Display Examples by Tool Type

```
┌──────────────────────────────────────────────────────────────────┐
│  FILE EDIT (format_hint="diff")                                   │
├──────────────────────────────────────────────────────────────────┤
│  @@ -10,5 +10,7 @@                                                │
│   class User:                                                     │
│       def __init__(self, name):                                   │
│  -        self.name = name                                        │
│  +        self.name = name                                        │
│  +        self.created_at = datetime.now()                        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  SHELL COMMAND (format_hint="code")                               │
├──────────────────────────────────────────────────────────────────┤
│  Command: npm install express@4.18.2                              │
│                                                                   │
│  Working directory: /home/user/project                           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  WEB FETCH (format_hint="text")                                   │
├──────────────────────────────────────────────────────────────────┤
│  URL: https://api.github.com/repos/user/repo                     │
│  Method: GET                                                      │
│  Headers: Authorization: Bearer ***                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Tool Execution Integration

The permission system integrates with `ToolExecutor` to gate all tool calls.

### Execution Flow with Permissions

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION WITH PERMISSIONS                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model: "Call updateFile(path='x.py', content='...')"               │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  ToolExecutor.execute(name, args, callback, call_id)         │    │
│  └────────┬────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  if self._permission_plugin:                                 │    │
│  │      allowed, perm_info = permission_plugin.check_permission(│    │
│  │          tool_name=name,                                     │    │
│  │          args=args,                                          │    │
│  │          context=self._permission_context,                   │    │
│  │          call_id=call_id                                     │    │
│  │      )                                                       │    │
│  └────────┬────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ├─────── if not allowed ──────►  Return error with        │
│           │                                 _permission metadata     │
│           │                                                          │
│           ▼ (if allowed)                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Execute tool                                                │    │
│  │  result = executor(args)                                     │    │
│  └────────┬────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Inject permission metadata into result                      │    │
│  │  result['_permission'] = {                                   │    │
│  │      'decision': 'allowed',                                  │    │
│  │      'reason': 'User approved',                              │    │
│  │      'method': 'user_approved'                               │    │
│  │  }                                                           │    │
│  └────────┬────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ▼                                                          │
│  Return (success=True, result)                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Permission Metadata in Results

Every tool result includes `_permission` metadata:

```python
# Allowed execution
{
    "output": "File updated successfully",
    "_permission": {
        "decision": "allowed",
        "reason": "User approved",
        "method": "user_approved"
    }
}

# Denied execution
{
    "error": "Permission denied",
    "_permission": {
        "decision": "denied",
        "reason": "Tool is blacklisted",
        "method": "blacklist"
    }
}
```

### Method Values

| Method | Meaning |
|--------|---------|
| `user_approved` | User explicitly approved in channel |
| `whitelist` | Matched whitelist rule |
| `blacklist` | Matched blacklist rule (denied) |
| `default` | Default policy applied |
| `auto_approved` | Tool is auto-approved |
| `suspended` | Turn/idle/all suspension active |
| `timeout` | Channel timeout (configurable default) |
| `sanitization` | Sanitization check failed (denied) |

---

## Part 8: Thread Safety and Parallel Execution

When multiple tools execute in parallel, the permission system ensures safe, serialized prompts.

### Channel Lock

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLEL PERMISSION HANDLING                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model: "Read config, update main.py, and create backup"            │
│           │                                                          │
│           ├──► Thread 1: readFile (auto-approved, no lock needed)   │
│           │                                                          │
│           ├──► Thread 2: updateFile                                 │
│           │         │                                                │
│           │         ▼                                                │
│           │    ┌───────────────────────────────────────┐            │
│           │    │  Acquire _channel_lock               │            │
│           │    │  (blocks if another thread has it)   │            │
│           │    └───────────────────────────────────────┘            │
│           │         │                                                │
│           │         ▼                                                │
│           │    ┌───────────────────────────────────────┐            │
│           │    │  Re-check suspensions                 │            │
│           │    │  (another thread may have set 'all') │            │
│           │    └───────────────────────────────────────┘            │
│           │         │                                                │
│           │         ▼                                                │
│           │    ┌───────────────────────────────────────┐            │
│           │    │  Show permission prompt               │            │
│           │    │  User sees ONE prompt at a time      │            │
│           │    └───────────────────────────────────────┘            │
│           │         │                                                │
│           │         ▼                                                │
│           │    Release _channel_lock                                │
│           │                                                          │
│           └──► Thread 3: writeNewFile                               │
│                     │                                                │
│                     ▼                                                │
│                Waits for lock (Thread 2 has it)                     │
│                     │                                                │
│                     ▼                                                │
│                Acquires lock after Thread 2 releases                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Call ID Matching

When multiple permissions are pending, `call_id` helps match responses:

```python
# Executor passes call_id
executor.execute("updateFile", args, callback, call_id="tool-call-123")

# Permission check includes call_id
permission_plugin.check_permission(..., call_id="tool-call-123")

# Event includes call_id for UI matching
PermissionInputModeEvent(call_id="tool-call-123", ...)
```

---

## Part 9: Server Events

The permission system emits events for client UI integration.

### Event Types

```python
# When permission prompt should be shown
@dataclass
class PermissionRequestedEvent(Event):
    agent_id: str                          # Which agent is requesting
    request_id: str                        # Unique request identifier
    tool_name: str                         # Tool being called
    tool_args: Dict[str, Any]              # Raw arguments
    response_options: List[Dict]           # Valid response options
    prompt_lines: Optional[List[str]]      # Pre-formatted prompt
    format_hint: Optional[str]             # "diff", "text", "code"
    warnings: Optional[str]                # Security warnings
    warning_level: Optional[str]           # "info", "warning", "error"

# When client should enter input mode
@dataclass
class PermissionInputModeEvent(Event):
    agent_id: str
    request_id: str
    call_id: Optional[str]                 # For parallel tool matching

# When permission has been resolved
@dataclass
class PermissionResolvedEvent(Event):
    agent_id: str
    request_id: str
    granted: bool                          # Was it approved?
    method: str                            # How was it decided?
    comment: str                           # Advisory comment, if any
    user_id: Optional[str]                 # WHO answered: the daemon-authenticated
                                           # user of the responding client (#859)
    approver: Optional[str]                # WHO an external approval system
                                           # (webhook / file response) named
```

### Who Decided (#859)

`method` says *how* a decision was reached; `user_id` and `approver` say
*who* reached it, and they are kept apart because their provenance differs:

| Field | Set when | Source | Verified? |
|-------|----------|--------|-----------|
| `user_id` | a client answered the prompt over an authenticated transport | the daemon's `get_client_user(client_id)` for the client that sent `PermissionResponseRequest`, stamped on the `PromptResponse` the runner receives | yes — the transport's identity, never the request body |
| `approver` | an external system answered through the webhook or file channel | the `approver` key of that response JSON | no — recorded as claimed |
| neither | a policy rule, evaluator or suspension decided; or the transport has no user (local IPC) | — | — |

The same attribution lands on the ledger's `permission-check` record and on
the plugin's own execution log, and the session record header carries
`created_by` (record version 2.9) so a session is attributable after the
fact without telemetry.  The ledger's `response` records carry the same
`user_id` the telemetry `user.id` attribute does.

### Event Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERMISSION EVENT SEQUENCE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Server                                              Client          │
│    │                                                    │            │
│    │  PermissionRequestedEvent                          │            │
│    │  {tool: "updateFile", args: {...},                │            │
│    │   prompt_lines: [...], format_hint: "diff"}       │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                                    │            │
│    │                                          Render permission      │
│    │                                          panel with diff        │
│    │                                                    │            │
│    │  PermissionInputModeEvent                          │            │
│    │  {request_id: "...", call_id: "..."}              │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                                    │            │
│    │                                          Focus input field      │
│    │                                          Show response options  │
│    │                                                    │            │
│    │  PermissionResponseRequest                         │            │
│    │  {request_id: "...", response: "y"}               │            │
│    │ ◄───────────────────────────────────────────────── │            │
│    │                                                    │            │
│    │  PermissionResolvedEvent                           │            │
│    │  {granted: true, method: "user_approved",         │            │
│    │   user_id: "sso|alice", approver: null}           │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                                    │            │
│    │                                          Clear permission       │
│    │                                          panel, resume normal   │
│    │                                                    │            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 10: Suspend and Resume

The permission system supports temporary suspension of all permission checks.

### Suspension States

| State | Set By | Cleared By | Behavior |
|-------|--------|------------|----------|
| `_turn_suspended` | User responds `t` | Turn ends | Allow all for current turn |
| `_idle_suspended` | User responds `i` | Session goes idle | Allow all until idle |
| `_allow_all` | User responds `all` or command | `permissions resume` | Allow all permanently |

### Suspension Priority

```python
# Check order in check_permission():
if self._idle_suspended:     # 1st: Idle suspension
    return True, {"method": "suspended", "reason": "Idle suspension active"}
if self._turn_suspended:     # 2nd: Turn suspension
    return True, {"method": "suspended", "reason": "Turn suspension active"}
if self._allow_all:          # 3rd: All suspension
    return True, {"method": "suspended", "reason": "All permissions suspended"}
# Otherwise: Use policy evaluation
```

### Lifecycle Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUSPENSION LIFECYCLE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User responds 'i' (idle)                                           │
│    │                                                                 │
│    ▼                                                                 │
│  _idle_suspended = True                                             │
│    │                                                                 │
│    ├──► Turn 1: All tools allowed                                   │
│    ├──► Turn 2: All tools allowed                                   │
│    ├──► Turn 3: All tools allowed                                   │
│    │                                                                 │
│    ▼                                                                 │
│  Session goes IDLE (no activity)                                    │
│    │                                                                 │
│    ▼                                                                 │
│  Server calls: permission_plugin.clear_idle_suspension()            │
│    │                                                                 │
│    ▼                                                                 │
│  _idle_suspended = False                                            │
│    │                                                                 │
│    ▼                                                                 │
│  Next tool call: Permission check required again                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 11: Sanitization (Security Layer)

An optional security layer that runs before whitelist/blacklist evaluation.

### Sanitization Checks

| Check | Purpose | Examples Blocked |
|-------|---------|------------------|
| **Shell Injection** | Prevent command injection | `; rm -rf /`, `$(malicious)`, `` `cmd` `` |
| **Dangerous Commands** | Block destructive operations | `sudo`, `rm`, `chmod 777`, `kill -9` |
| **Path Scope** | Restrict file access | `../../etc/passwd`, `/root/.ssh` |

### Configuration

```python
# Enable sanitization
policy.set_sanitization(config={
    "shell_injection": True,
    "dangerous_commands": ["sudo", "rm", "chmod", "kill"],
    "allowed_paths": ["/home/user/project"]
}, cwd="/home/user/project")

# Or use strict sandbox mode
policy.enable_strict_sandbox(cwd="/home/user/project")
```

### Evaluation Position

```
Sanitization runs FIRST, before any whitelist/blacklist:

  Tool Call
      │
      ▼
  Sanitization ──► DENY (if violations found)
      │
      ▼ (if clean)
  Blacklist Check
      │
      ▼
  Whitelist Check
      │
      ▼
  Default Policy
```

---

## Part 12: Visual Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERMISSION SYSTEM OVERVIEW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         ┌─────────────────┐                         │
│                         │   TOOL CALL     │                         │
│                         │  updateFile()   │                         │
│                         └────────┬────────┘                         │
│                                  │                                   │
│              ┌───────────────────┴───────────────────┐              │
│              ▼                                       ▼              │
│  ┌─────────────────────────┐          ┌─────────────────────────┐  │
│  │    AUTO-APPROVED?       │          │      SUSPENDED?         │  │
│  │                         │          │                         │  │
│  │  readFile       ✓       │          │  _turn_suspended   ✓    │  │
│  │  web_search     ✓       │          │  _idle_suspended   ✓    │  │
│  │  calculate      ✓       │          │  _allow_all        ✓    │  │
│  │  updateFile     ✗       │          │                         │  │
│  └─────────────────────────┘          └─────────────────────────┘  │
│              │                                       │              │
│              │ No                                    │ No           │
│              └───────────────────┬───────────────────┘              │
│                                  ▼                                   │
│                    ┌─────────────────────────┐                      │
│                    │    POLICY EVALUATION    │                      │
│                    │                         │                      │
│                    │  1. Sanitization        │                      │
│                    │  2. Session Blacklist   │                      │
│                    │  3. Static Blacklist    │                      │
│                    │  4. Session Whitelist   │                      │
│                    │  5. Static Whitelist    │                      │
│                    │  6. Default Policy      │                      │
│                    └────────────┬────────────┘                      │
│                                 │                                    │
│           ┌─────────────────────┼─────────────────────┐             │
│           ▼                     ▼                     ▼             │
│     ┌──────────┐         ┌──────────┐         ┌──────────┐         │
│     │  ALLOW   │         │   DENY   │         │   ASK    │         │
│     │          │         │          │         │          │         │
│     │ Execute  │         │  Return  │         │  Channel │         │
│     │ tool     │         │  error   │         │  prompt  │         │
│     └──────────┘         └──────────┘         └────┬─────┘         │
│                                                     │                │
│                                    ┌────────────────┴────────────┐  │
│                                    ▼                             ▼  │
│                              ┌──────────┐                 ┌────────┐│
│                              │   USER   │                 │TIMEOUT ││
│                              │ RESPONSE │                 │        ││
│                              └────┬─────┘                 └───┬────┘│
│                                   │                           │     │
│     ┌─────────────────────────────┼───────────────────────────┤     │
│     ▼           ▼           ▼     ▼           ▼               ▼     │
│  ┌─────┐    ┌─────┐    ┌─────┐ ┌─────┐    ┌─────┐        ┌─────┐   │
│  │  y  │    │  n  │    │  a  │ │never│    │  t  │        │deny │   │
│  │     │    │     │    │     │ │     │    │     │        │ or  │   │
│  │Allow│    │Deny │    │Add  │ │Add  │    │Turn │        │allow│   │
│  │once │    │once │    │to WL│ │to BL│    │susp.│        │     │   │
│  └─────┘    └─────┘    └─────┘ └─────┘    └─────┘        └─────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 13: Color Coding Suggestion for Infographic

- **Green:** Allow decisions (whitelist, auto-approved, user approved)
- **Red:** Deny decisions (blacklist, user denied, sanitization violation)
- **Yellow:** Ask/pending states (waiting for user input)
- **Blue:** Suspension states (turn, idle, all)
- **Gray:** Policy evaluation flow
- **Orange:** Channel communication
- **Purple:** Server events
