# Profiles & RBAC Comparison: Jaato vs Google ADK vs LangChain

Comparison of how each framework handles agent profiles (role definitions),
tool access control, permission enforcement, and security boundaries.

---

## Executive Summary

| Capability | Jaato | Google ADK | LangChain / LangGraph |
|---|---|---|---|
| **Declarative agent profiles** | JSON profiles with plugin lists, model/provider overrides, GC config, env vars | Agent class with `tools` list; no external profile files | No built-in profiles; tools passed per-agent in code |
| **Permission policy engine** | 3-layer: sanitization → blacklist/whitelist → default policy | Callback-based (`before_tool_callback`) + Plugin guardrails | `HumanInTheLoopMiddleware` with per-tool approve/edit/reject policies |
| **Interactive approval** | Multi-mode: y/n/always/never/turn/idle + channels (console, webhook, queue) | Must be hand-coded in callbacks | `interrupt_before` / `interrupt_after` + `interrupt()` function + HITL middleware |
| **Per-agent tool scoping** | Plugin list per profile → registry only exposes listed tools | Tools list per Agent constructor | Tools list per `ToolNode`; middleware can filter dynamically by user role |
| **Blacklist / whitelist** | Static + session-level, pattern-based (globs), argument-level | Not built-in; implementable in callbacks | Deep Agents: `-S` shell allow-list (specific cmds / `recommended` / `all`) + 13 blocked injection patterns |
| **Session isolation** | ContextVar + threading.local; CI-enforced plugin safety | Separate Session objects; no thread isolation guarantees | Separate state per graph node; no isolation enforcement |
| **Sandboxing** | Path scoping, shell metachar blocking, sanitization config; **AppArmor profiles** (premium) for kernel-level MAC | GKE Code Executor (container/microVM), VPC-SC | LangSmith Sandboxes (microVM), deprecated Pyodide |
| **Auth delegation** | OAuth plugins per provider (Anthropic PKCE, GitHub device code, Google) | `ToolContext.request_credential()` + OAuth flows | Not built-in; manual token management |

---

## 1. Agent Profiles / Role Definitions

### Jaato

Declarative JSON files in `.jaato/profiles/`:

```json
{
  "name": "github-resolver",
  "description": "Fixes GitHub issues autonomously",
  "model": "claude-sonnet-4-20250514",
  "provider": "anthropic",
  "plugins": ["cli", "file_edit", "lsp(preload)", "permission"],
  "plugin_configs": {
    "permission": {
      "policy": {
        "defaultPolicy": "allow",
        "blacklist": { "arguments": { "cli_based_tool": { "command": ["git push -f", "rm -rf"] } } }
      }
    }
  },
  "max_turns": 10,
  "auto_approved": false,
  "gc": { "type": "budget", "threshold_percent": 80.0 },
  "env": { "GITHUB_TOKEN": "${VAULT_SECRET_ID}" }
}
```

**Strengths:**
- Profiles are **declarative and portable** — checked into the repo, no code changes needed
- Per-profile permission policy overrides (a "researcher" profile can be permissive on web tools but locked down on file writes)
- Plugin `(preload)` syntax controls deferred vs eager tool loading
- `env` with `${VAR}` expansion and secret URI support
- Discovery precedence: workspace → user → premium (entry points)
- Profile-level GC strategy (budget, summarize, hybrid)

### Google ADK

Profiles are defined **in code** via the `Agent` class:

```python
agent = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    tools=[web_search, code_execution],
    instruction="You are a research analyst...",
    before_tool_callback=my_guardrail,
)
```

**Strengths:**
- Simple, Pythonic API
- `before_tool_callback` and `after_tool_callback` for per-agent guardrails
- Plugin system for cross-cutting guardrails (registered on Runner, applies globally)
- Agent teams with hierarchical delegation (single-parent rule, strict tree)
- **Dynamic tool scoping** via `BaseToolset.get_tools(readonly_context)` — toolset decides at runtime which tools to expose based on user permissions, session state, etc.
- **`MCPToolset` with `tool_filter`** — effectively a whitelist for MCP tools
- **`LongRunningFunctionTool`** — built-in human-in-the-loop: runner pauses, client decides to approve/reject, sends `FunctionResponse` to resume

**Gaps vs Jaato:**
- No external profile files — tool/permission config lives in application code
- No declarative blacklist/whitelist — must implement in callbacks
- No declarative per-agent permission policies
- No profile discovery or layered precedence
- **Shared session state in multi-agent** — agents in `ParallelAgent` share `session.state` with no data isolation (must use unique keys to avoid races); jaato provides per-session isolation via ContextVar + threading.local

### LangChain / LangGraph

Tools passed directly in code, with newer middleware-based dynamic filtering:

```python
# LangGraph — static tool binding
builder.add_node("tools", ToolNode(tools=[search, calculator]))

# LangGraph — middleware-based dynamic filtering (newer API)
# Middleware reads user role from state and filters tools at runtime
agent = create_agent(model, tools, middleware=[role_filter_middleware])
```

**Strengths:**
- Extremely flexible graph-based composition (LangGraph)
- `interrupt_before` / `interrupt_after` + programmatic `interrupt()` function
- `HumanInTheLoopMiddleware` — declarative per-tool approval policy (approve/edit/reject)
- Middleware can dynamically scope tools by user role, feature flags, or Store config
- Deep Agents for containerized execution (Modal, Daytona, Runloop)
- Deep Agents CLI shell allow-list: specific commands, `recommended` (42 safe defaults), or `all`

**Gaps vs Jaato:**
- No declarative profile files — everything is imperative code
- No argument-level policy matching (tool-level only)
- HITL requires external checkpointer (PostgreSQL etc.) for state persistence
- No multi-channel approval (console only, or custom code)
- No turn/idle approval scoping — decisions are per-invocation

---

## 2. Permission & Access Control

### Jaato: 3-Layer Policy Engine

```
Sanitization checks → Session blacklist → Static blacklist
    → Session whitelist → Static whitelist → Default policy
```

**Key design:**
- Blacklist **always** beats whitelist (defense in depth)
- Argument-level matching (block `rm -rf /` but allow `rm temp.txt`)
- Glob patterns for commands (`git push -f *`)
- Multiple approval channels: console, webhook, file, queue, parent-bridged
- 7 approval modes: yes, no, always, never, turn, idle, once
- Session-level dynamic rules (user runs `permissions allow git*` mid-session)

**Config (`permissions.json`):**
```json
{
  "defaultPolicy": "ask",
  "blacklist": { "patterns": ["rm -rf /*", "sudo *"] },
  "whitelist": { "patterns": ["git status", "npm test"] },
  "sanitization": { "block_shell_metacharacters": true, "path_scope": { "allowed_roots": ["."] } }
}
```

### Google ADK: Callback + Plugin + LongRunningFunctionTool

```python
# Callback-based guardrail (block or allow)
def my_guardrail(callback_context, tool, args, tool_context):
    if tool.name == "execute_sql" and "DROP" in args.get("query", ""):
        return {"error": "DROP statements are not allowed"}
    return None  # allow execution

# Dynamic tool scoping (runtime filtering)
class RoleAwareToolset(BaseToolset):
    async def get_tools(self, readonly_context):
        role = readonly_context.state.get("user_role", "viewer")
        if role == "admin":
            return [read_tool, write_tool, delete_tool]
        return [read_tool]  # viewers get read-only

# Human-in-the-loop (built-in pause/resume)
approval_tool = LongRunningFunctionTool(func=transfer_money)
# Runner pauses → client shows approval UI → sends FunctionResponse to resume
```

**Key design:**
- `before_tool_callback` per-agent or globally via Plugin
- Full access to `ToolContext` (state, auth, artifacts)
- Plugin system for reusable cross-agent guardrails (registered on Runner)
- `BaseToolset.get_tools()` for dynamic runtime tool filtering
- `MCPToolset(tool_filter=...)` for MCP tool whitelisting
- `LongRunningFunctionTool` for human-in-the-loop approval
- Authentication via `tool_context.request_credential()`
- Policy data stored in `session.state` and enforced in callbacks

**Strengths:** Very flexible — any Python logic as a guardrail. Dynamic toolsets enable runtime RBAC. `LongRunningFunctionTool` provides real HITL.
**Weaknesses:** No declarative policy language. Every rule is imperative code. No multi-mode approval (turn/idle/always/never). No argument-level pattern matching. Shared session state means no data isolation between parallel agents.

### LangChain / LangGraph: HITL + Middleware + Shell Allow-Lists

Three layers of increasing sophistication:

```python
# 1. Static breakpoints (compile-time)
graph.compile(interrupt_before=["tools"])

# 2. Programmatic interrupt (runtime, Jan 2025+)
def my_node(state):
    if state["risk_level"] > threshold:
        answer = interrupt({"question": "Approve this action?", "tool": tool_name})

# 3. HumanInTheLoopMiddleware (declarative, newest)
HumanInTheLoopMiddleware(interrupt_on={
    "write_file": True,                              # approve/edit/reject
    "execute_sql": {"allowed_decisions": ["approve", "reject"]},
    "read_data": False,                              # auto-approve
})
```

**Deep Agents CLI shell allow-list:**
```bash
deepagents run -S "pytest,git,make"     # specific commands only
deepagents run -S recommended           # 42 curated safe defaults
deepagents run -S all                   # permit anything
# 13 injection patterns always blocked regardless of allow-list
```

**Key design:**
- `interrupt_before`/`interrupt_after` for static breakpoints
- `interrupt()` function for dynamic, conditional pauses inside node logic
- `HumanInTheLoopMiddleware` maps tool names → approval policies (closest to jaato's permission plugin)
- Human can **modify agent state** before resuming (unique to LangGraph)
- Requires **external checkpointer** (PostgreSQL, SQLite) for state persistence across interrupts
- Deep Agents: 3-tier shell allow-list + 13 always-blocked injection patterns + path traversal validation

**Strengths:** Most flexible interruption model (static + dynamic + declarative); state editing on resume; strong sandboxing via LangSmith.
**Weaknesses:** No argument-level pattern matching. No multi-channel approval (webhook, file, queue). No turn/idle scoping. Requires external persistence for HITL. No declarative policy files.

---

## 3. Session Isolation & Security Boundaries

### Jaato

| Mechanism | Purpose |
|---|---|
| `ContextVar` per-session | Prevents cross-session data leakage in async code |
| `threading.local()` per-plugin | Prevents cross-thread state sharing |
| CI test enforcement | `test_plugin_session_safety.py` scans for `self._*` in `set_session()` |
| Per-thread permission channels | Subagent approvals don't leak to parent |
| Profile-scoped `env` vars | Only applied to the subagent's thread |
| Path scoping / sanitization | Configurable allowed/denied filesystem paths |
| **AppArmor profiles** (premium) | Kernel-level Mandatory Access Control — confines tool processes to declared file, network, and capability rules |

### Google ADK

| Mechanism | Purpose |
|---|---|
| Per-agent `tools` list | Each agent can only use explicitly declared tools |
| Single-parent tree rule | `ValueError` if agent assigned to two parents |
| OAuth-scoped identity | Tools use controlling user's OAuth token (natural external boundary) |
| GKE Code Executor | Container-level isolation for code execution |
| VPC-SC perimeters | Network-level data exfiltration prevention |
| Model Armor | Centralized content safety with RBAC |

**Notable gap:** Agents in `ParallelAgent` share `session.state` — no data isolation at the state level. Each agent must write to unique keys to avoid race conditions. Google ADK relies more on **infrastructure-level isolation** (GKE, VPC-SC) than application-level session isolation.

### LangChain / LangGraph

| Mechanism | Purpose |
|---|---|
| Graph state isolation | Each node operates on its own state slice |
| LangSmith Sandboxes | MicroVM isolation for code execution |
| Binary authorization | Control which binaries can run in sandbox |
| Deprecated: Pyodide WASM | Application-level isolation (not recommended) |

LangChain's strongest isolation story is **LangSmith Sandboxes** — true microVM isolation with binary authorization and network restrictions. However, this is a paid service, not a framework feature.

**Comparison note:** Jaato's premium AppArmor support provides **kernel-level MAC** (Mandatory Access Control) — a different isolation approach that operates at the OS level rather than requiring container/microVM infrastructure. AppArmor profiles confine tool processes to declared filesystem paths, network access, and Linux capabilities, and are enforced by the kernel itself (not bypassable by the agent). This gives jaato a strong isolation primitive that doesn't require external infrastructure, though it complements rather than replaces container-level isolation for full defense-in-depth.

---

## 4. Unique Differentiators

### Jaato Only
- **Declarative profile files** — non-developers can define agent roles via JSON
- **7-mode approval system** — turn/idle/always/never/once granularity
- **Multiple approval channels** — webhook, file, queue (not just console)
- **Argument-level blacklist/whitelist** — block specific command arguments, not just tool names
- **Plugin (preload) syntax** — fine-grained control over tool loading strategy
- **CI-enforced session safety** — automated tests catch cross-session leakage
- **Profile-level GC strategy** — each agent role can have different context management
- **AppArmor confinement** (premium) — kernel-level Mandatory Access Control that confines tool processes to declared file paths, network access, and Linux capabilities; enforced by the kernel, not bypassable by the agent

### Google ADK Only
- **`ToolContext.request_credential()`** — first-class OAuth flow integrated into tool execution
- **`BaseToolset.get_tools(context)`** — dynamic runtime tool filtering based on session state / user role
- **`LongRunningFunctionTool`** — built-in HITL with runner pause/resume via `FunctionResponse`
- **`MCPToolset(tool_filter=...)`** — declarative MCP tool whitelisting
- **Plugin system for global guardrails** — register once on Runner, applies everywhere
- **Model Armor integration** — enterprise content safety with centralized RBAC
- **GKE Code Executor** — native Kubernetes integration with RBAC YAML configs
- **Agent teams** — built-in hierarchical agent delegation with single-parent tree rule

### LangChain / LangGraph Only
- **Graph-based execution** — most flexible composition model for complex workflows
- **State modification on interrupt** — human can edit agent state before resuming
- **`HumanInTheLoopMiddleware`** — declarative per-tool approval mapping (approve/edit/reject)
- **Middleware-based dynamic tool scoping** — filter tools by user role/feature flags at runtime
- **Deep Agents shell allow-list** — 3-tier system (specific/recommended/all) + 13 injection pattern blocks
- **Deep Agents** — first-party integrations with Modal, Daytona, Runloop for containerized agents
- **LangSmith Sandboxes** — production-grade microVM isolation with binary authorization
- **Massive ecosystem** — most integrations, community tools, and third-party extensions

---

## 5. Gap Analysis for Jaato

### What Jaato Has That Others Don't
1. **Declarative, portable profiles** — biggest differentiator for enterprise/team use
2. **Granular approval modes** — no other framework offers turn/idle/always/never
3. **Argument-level policy matching** — ADK and LangChain filter at tool level; Deep Agents blocks injection patterns but doesn't support custom argument globs
4. **CI-enforced isolation** — automated safety testing for plugin developers
5. **Multi-channel approval** — webhook/file channels enable CI/CD and external approval workflows

### Potential Improvements Inspired by Others
1. **Global guardrail plugins** (from ADK) — jaato's permission plugin is per-profile; a Runner-level plugin that applies cross-cutting policies to all sessions could simplify enterprise governance
2. **`request_credential()` in tool context** (from ADK) — jaato has auth plugins per-provider, but a standardized `tool_context.request_credential()` API would unify the pattern
3. **Graph-based interruption** (from LangGraph) — jaato's approval is synchronous; async graph-style "pause and resume with modified state" could enable more complex approval workflows
4. **Container/microVM sandbox integration** (from both) — jaato already has kernel-level confinement via AppArmor (premium), which is stronger than application-level sandboxing; adding container or microVM isolation (GKE, Docker) would complement AppArmor for full defense-in-depth in code execution scenarios

---

## Sources

- [Google ADK Safety & Security](https://google.github.io/adk-docs/safety/)
- [Google ADK Callbacks](https://google.github.io/adk-docs/callbacks/)
- [Google ADK Callback Patterns](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/)
- [Google ADK Plugins](https://google.github.io/adk-docs/plugins/)
- [Google ADK Authentication](https://google.github.io/adk-docs/tools-custom/authentication/)
- [Google ADK Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/)
- [Google ADK Custom Tools](https://google.github.io/adk-docs/tools-custom/)
- [Google ADK MCP Tools](https://google.github.io/adk-docs/tools-custom/mcp-tools/)
- [Google ADK Tool Limitations](https://google.github.io/adk-docs/tools/limitations/)
- [Google ADK GKE Code Executor](https://google.github.io/adk-docs/integrations/gke-code-executor/)
- [LangChain Security Best Practices](https://python.langchain.com/docs/security/)
- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangGraph interrupt() Function](https://blog.langchain.com/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt/)
- [LangChain Agent Authorization](https://blog.langchain.com/agent-authorization-explainer/)
- [Deep Agents CLI](https://docs.langchain.com/oss/python/deepagents/cli/overview)
- [LangSmith Sandboxes](https://docs.smith.langchain.com/evaluation/how_to_guides/sandboxes)
