# Resilience Analysis — Layer-by-Layer Failure Modes

**Date:** 2026-04-09
**Scope:** A detailed audit of jaato's resilience to failures across all
layers, identifying both well-handled cases and gaps. Captures the
analysis from the 2026-04-09 conversation so it can guide future
hardening work without re-deriving the picture each time.

## Layer overview

| Layer | Purpose | Grade |
|---|---|---|
| Transport (WS/IPC) | Client ↔ server connections | C |
| Session lifecycle | Multi-client session management | B |
| Provider (LLM calls) | Model API interaction | B+ |
| Tool execution | ToolExecutor + plugins | B− |
| Context/budget | History management & GC | B |
| Storage | Session/history/workspace persistence | C− |
| Plugin lifecycle | Discovery, init, shutdown | D |
| MCP / external services | External tool servers | D |
| Service connector | HTTP API calls | C+ |
| Subagents | Multi-agent coordination | C |
| Telemetry (OTel) | Observability | C |
| AppArmor | Sandboxing | B |

Grades reflect 2026-04-09 state. They're rough — the goal is relative
comparison, not absolute scoring.

---

## 1. Transport (WS / IPC)

### Strengths

- WS heartbeat: `ping_interval=30`, `ping_timeout=10`
  (`server/websocket.py:520-521`)
- SDK `IPCRecoveryClient` (`jaato-sdk/jaato_sdk/client/recovery.py`)
  has a real state machine + exponential backoff with jitter
  (max 10 attempts, base 1s, cap 60s, ±30% jitter)
- Auto session re-attach after reconnect via `reattach_session=True`
- WS disconnect cleanup propagates to `attached_clients` (commit
  `7c8c4aaf`)

### Gaps

- **No server-side message replay.** Events emitted while a client is
  disconnected are lost. Re-attached clients only see the current
  state snapshot, not the events they missed.
- **IPC has no heartbeat.** Connection state is unknown until the
  next read/write fails.
- **Ephemeral `client_id`.** Each reconnect gets a fresh ID
  (`client_1`, `client_2`...). The server can't correlate
  reconnections to identity without external auth.
- **No SDK outbound buffering.** Sends during `RECONNECTING` raise
  `ReconnectingError` instead of being queued.

---

## 2. Session lifecycle

### Strengths

- Persistence on multiple triggers (`server/session_manager.py`):
  - On turn completion (line 720–722) — captures interrupted state
  - Manual checkpoint (`save_session`, line 1943–1956)
  - On unload (line 1816–1817) — flushes dirty state
  - Bulk save on shutdown (line 2615–2627)
- Saved state covers: history, turn accounting, user inputs,
  interrupted turn state (pending tool calls), workspace file hashes,
  plugin state, budget snapshot
- Re-attaching client triggers load from disk + reinit of `JaatoServer`
- WS workspace reaper (24h default)

### Gaps

- **No idle timeout for sessions.** A session with at least one
  attached client stays in memory forever, even if completely idle.
- **No orphan cleanup.** A client crash without clean detach leaves
  the session attached to a phantom client.
- **No token expiry detection.** If a user's auth token expires
  mid-session, calls fail; user must manually re-auth and create
  a new session.
- **No cross-client notification of state changes.** When one client
  mutates session state, other attached clients see stale data
  until the next event arrives.

---

## 3. Provider (LLM calls)

### Strengths

`with_retry()` in `shared/retry_utils.py` is mature:

- Three-way error classification: `transient` / `rate_limit` / `infra`
  (line 135–183)
- Provider-specific exception detection: Google, GitHub Models,
  Anthropic, Antigravity
- Honors `Retry-After` header when present (line 237–259)
- Exponential backoff `base × 2^(attempt-1)`, capped at `max_delay`,
  ±50% jitter (line 262–291)
- Cancel-token-aware: `interruptible_sleep()` polls the token every
  100ms and can abort mid-backoff (line 294–327)
- Configurable via env: `AI_RETRY_ATTEMPTS`, `AI_RETRY_BASE_DELAY`,
  `AI_RETRY_MAX_DELAY`
- `RequestPacer` for proactive rate limiting (`AI_REQUEST_INTERVAL`)
- Streaming exceptions bubble up as exceptions for `with_retry`
  to handle, instead of being silently swallowed

### Gaps

- **No per-provider circuit breaker.** A globally-down provider
  exhausts the retry budget on every single call. Sessions don't
  share failure state across calls.
- **Streaming partial responses are not preserved across retries.**
  A retry restarts from scratch.
- **`Retry-After` only honored for HTTP-style transports.**

---

## 4. Tool execution

### Strengths

- All tool exceptions are caught and returned as
  `(False, {error, traceback})` — never crash the turn
  (`shared/ai_tool_runner.py:719-770`)
- `CancelledException` is special-cased, no retry, clean exit
- **Auto-background** for long-running tools that opt in via
  `BackgroundCapable`: tools exceeding the threshold return a
  continuation handle and the agent gets the result later via
  callback (line 390–487)
- **Reliability plugin** tracks `(tool_name, args_signature)` failure
  patterns and can escalate to "require explicit approval" after
  threshold (line 692–747)
- Permission denials are **soft fails** — the model gets the error
  in the tool result and can react

### Gaps

- **No hard timeout on individual tools.** Auto-background mitigates
  this only for plugins that opt in. File plugins, web plugins, and
  most others have no upper bound.
- **No retry on flaky tools.** Tools that fail intermittently surface
  the failure on every call.
- **Reliability tracking is per-session.** Cross-session learning
  (e.g. "this tool always fails with these args on this model")
  is not propagated.

---

## 5. Context / budget management

### Strengths

- **Two-stage recovery** on context-limit errors
  (`shared/jaato_session.py:4836-4922`):
  1. Try GC first via `_try_gc_for_context_recovery()`
  2. Fall back to `_truncate_results_to_fit()` if GC didn't help
- Token count extracted from error message via regex
  (handles "X exceeds limit Y")
- Truncation strategy: largest results first, line-based then
  char-based, preserves first 20 lines / 2000 chars, attaches notice
- Proactive `_cap_tool_results` catches oversized results before
  they enter history (commit `9e5af82b` and follow-ups)

### Gaps

- **GC plugin exceptions are not caught.** If `gc_plugin.collect()`
  raises, the entire context recovery fails. Fallback to truncation
  only triggers when GC returns `success=False`, not on exception.
- **Token counting failures are silent.** When `count_tokens()` returns
  0 or errors, the budget falls back to estimates without surfacing
  the divergence.
- **Reactive truncation can fail to free enough.** If truncation
  can't fit the context, the original error re-raises and the turn
  dies.

---

## 6. Storage / persistence ⚠️

The **weakest layer**. Multiple silent failures and non-atomic writes.

### Critical issues

- **Session save is non-atomic** (`shared/plugins/session/file_session.py:137-160`).
  Direct write — no `tmp + fsync + rename`. A power loss during save
  can leave a corrupted JSON.
- **Backup metadata writes silently fail**
  (`shared/plugins/file_edit/backup.py:165`):
  `except IOError: pass`. The backup file exists but the metadata
  pointing to it is lost → orphaned backup.
- **Waypoint save silently fails**
  (`shared/plugins/waypoint/manager.py:141`): same `except IOError: pass`.
  Waypoint exists in memory but never persisted to disk.
- **Token ledger has no fsync.** Append-mode JSONL means partial
  lines on crash.

### What's reasonable

- File backup creation itself is atomic (`Path.write_bytes()`)
- Waypoint **load** handles corruption gracefully (starts fresh)
- Token ledger errors are caught and returned as `None`

---

## 7. Plugin lifecycle ⚠️

The **second-weakest layer**.

### Critical issues

- **`plugin.initialize()` is not wrapped in try/except**
  (`shared/plugins/registry.py:725-779`). If a plugin's init raises,
  the entire `expose_tool()` fails and may take down session creation.
- **`plugin.shutdown()` is not wrapped.** First plugin to fail stops
  all subsequent shutdowns. Worse: `_exposed.discard()` never runs,
  so the registry is left inconsistent.
- **Re-initialization (config change) is not wrapped.** If shutdown
  fails, init still runs; if init fails, the plugin is in an
  unknown state.
- **Missing `PLUGIN_KIND` is silently skipped** with NO warning
  (`registry.py:507-541`). A plugin author who forgets the marker
  gets zero feedback — the plugin never loads.

### What's reasonable

- Workspace path broadcasting catches and logs per-plugin errors
- Parallel initialization has a 30s timeout and catches exceptions
- Entry-point load failures are logged at trace level

---

## 8. MCP / external tool servers ⚠️

### Critical issues

- **No try/except in `connect()` flow**
  (`shared/mcp_context_manager.py:122-166`). If `stdio_client` fails,
  the partial context leaks.
- **No reconnection logic.** A dead MCP server stays dead.
- **`get_connection()` raises `KeyError`** if the server is
  disconnected mid-call → propagates up and crashes tool execution.
- **No health check** before tool calls.

### What's reasonable

- Cleanup is defensive: try/except per context (line 264–269)

---

## 9. Service connector

### Strengths

- Structured error responses with hints (SSL → "retry with
  insecure=true", proxy → "retry with no_proxy=true")
- Auth resolution returns structured errors (commit `2bc34206`:
  `auth_context` field with `credentials_resolved` flag)
- SSL/proxy/auth error detection in
  `shared/plugins/service_connector/plugin.py`

### Gaps

- **No retry/backoff** at the `call_service` layer. Transient network
  errors fail immediately.
- **No connection pooling.**

---

## 10. Subagents

### Strengths

- Crashes propagate to parent via the parent's message queue with
  structured error
- Activity phase tracking for observability:
  `IDLE`, `WAITING_FOR_LLM`, `STREAMING`, `EXECUTING_TOOL`

### Gaps

- **No hard timeout on subagent model calls.** Documentation tells
  the user `WAITING_FOR_LLM` is "not stuck" — but there's no
  framework-level escape if the model truly hangs.
- **Thread pool max_workers=4, unbounded queue.** Spawning a 5th
  subagent blocks indefinitely.
- **No graceful pool exhaustion error.**

---

## 11. Telemetry (OTel)

### Gaps

- `_ensure_imports()` not wrapped — `ImportError` crashes plugin init
- File exporter has no disk-full detection
- Long-lived spans stored in dict; if `span.end()` raises, they leak
- OTLP endpoint missing → silent skip with no warning

---

## 12. AppArmor

### Strengths

- `is_available()` is exhaustive (Linux check, tools check, kernel
  module check, sudo check, cache writability) and degrades to
  no-ops if anything fails
- Profile provisioning catches write/parser/timeout errors and
  cleans up the file
- Profile teardown catches both unload and delete errors
- Thread-level confinement exit failure stays in the safe state

### Gaps

- **Thread confinement entry failure is logged at debug level.**
  A tool may run unconfined with no warning.
- **Profile file write success not validated** before invoking
  `apparmor_parser`.

---

## Cross-cutting concerns

### Cancellation propagation

Reasonably consistent across layers. Cancel tokens flow through
`with_retry`, tool execution, and provider streaming. **Gap:** the
`fork_ask` gate raises `TimeoutError` after 120s but doesn't actively
cancel the hung provider call.

### Logging vs. raising

Inconsistent style across the codebase:

- Some failure paths log and continue (good for resilience)
- Others log and raise (good for observability)
- **Storage and plugin lifecycle lean too far toward "log silently
  and continue"**, hiding bugs
- **MCP and OTel lean too far toward "let it raise"**, taking down
  whole sessions

### No cascading failure protection

When one layer fails, there's no attempt to isolate the blast radius:

- A failed plugin shutdown leaves the registry inconsistent
- A hung MCP server takes down a tool call
- A corrupted session save can break attachment

---

## Prioritized remediation roadmap

### Tier 1 — data loss prevention

1. **Atomic session save**: write to `<file>.tmp`, fsync, rename.
2. **Surface storage errors** in `backup.py:165` and `manager.py:141`.
3. **fsync the token ledger** after each append.

### Tier 2 — plugin lifecycle hardening

4. **Wrap `plugin.initialize()` in try/except** in `registry.py`.
   A failing plugin should be skipped with a logged error, not
   crash the session.
5. **Wrap `plugin.shutdown()` in try/except**. Continue shutdowns
   even if one fails. Update `_exposed` regardless.
6. **Warn loudly on missing `PLUGIN_KIND`**. The current silent
   skip is the #1 source of "why isn't my plugin loading"
   debugging time.

### Tier 3 — external service resilience

7. **MCP reconnection.** Detect dead servers, retry with backoff,
   mark as failed after N attempts.
8. **Wrap MCP `connect()` in try/except** with cleanup of partial
   contexts.
9. **Add retry/backoff to `call_service`** for transient network
   errors (using the existing `with_retry` infrastructure).

### Tier 4 — observability of degraded modes

10. **AppArmor entry failure should warn**, not debug-log. Operating
    unconfined silently is a security gap.
11. **Token expiry detection** at the session level — periodic
    refresh check or 401 propagation.
12. **GC plugin exception handling** in context-limit recovery —
    fall back to truncation-only on exception, don't crash the turn.

### Tier 5 — multi-tenant safety

13. **Session idle timeout** for orphan cleanup.
14. **Subagent pool queue limit + submission timeout.** Fail fast
    instead of blocking forever.
15. **Per-provider circuit breaker** that short-circuits requests
    when a provider is globally down.

---

## What's already strong

Worth highlighting because the analysis is heavy on gaps:

- **Provider retry logic** is mature and well thought through —
  three-way classification, retry-after honoring, jitter, cancel
  integration
- **Context-limit recovery** is a clever two-stage cascade
  (GC → truncation)
- **SDK recovery client** has a real state machine and exponential
  backoff
- **Session persistence triggers** are comprehensive (turn complete,
  attach, detach, shutdown, manual)
- **AppArmor degraded-mode handling** is exemplary — exhaustive
  availability check + graceful no-op fallback
- **Tool execution never crashes the turn** — every exception
  becomes a structured error
