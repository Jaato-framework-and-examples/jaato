# Session-Owned History Impact Analysis

**Date:** 2026-02-22
**Status:** Design — no implementation started
**Cross-references:**
- [Cache Plugin Design](../jaato-cache-plugin-design.md) — Variants A/B, GC-cache coordination
- [Cache Plugin Sequencing](cache-plugin-sequencing-impact-analysis.md) — A→B transition breakdown
- [Pending Implementation Plan](cache-plugin-pending-implementation-plan.md) — Tier 2/3 items blocked on this

## Executive Summary

This document proposes **inverting message history ownership** from providers to the session. Today, each `ModelProviderPlugin` maintains its own `self._history: List[Message]` and the session reads it back via `provider.get_history()`. The proposal moves the canonical history to `JaatoSession._messages`, making providers **stateless with respect to conversation history** — they receive messages as parameters and return responses without maintaining accumulated state.

The migration proceeds in **5 phases**, each independently deployable:

| Phase | Description | Scope | Unblocks |
|-------|-------------|-------|----------|
| 1 | `SessionHistory` wrapper — session holds canonical copy, still syncs to provider | Foundation | Phase 2 |
| 2 | Stateless `complete()` on Anthropic provider | ~200 LOC | Cache plugin Variant B |
| 3 | Migrate remaining 6 providers to `complete()` | ~600 LOC | Phase 4 |
| 4 | Remove legacy protocol methods | ~400 LOC removed | Phase 5 |
| 5 | Simplify session internals | ~300 LOC simplified | Full architecture |

**Total estimated scope:** ~1500 LOC touched across all providers and session.

---

## Motivation

### 1. Dual Ownership Creates Friction

The session needs history for GC decisions, budget tracking, and context management, but the provider owns the authoritative copy. The session must call `provider.get_history()` to read it and `provider.create_session(history=...)` to write it back after GC. This round-trip is error-prone and couples the session's context management to the provider's session lifecycle.

```
Current: session asks provider for history → GC processes → session tells provider to replace
Target:  session owns history → GC slices directly → no provider round-trip
```

### 2. Expensive GC Reset

After GC collects content, the session calls `reset_session(new_history)` which calls `_create_provider_session(history)` which calls `provider.create_session(system_instruction, tools, history=new_history)`. This recreates the entire provider session state — system instruction, tool schemas, and history — even though only the history changed. For providers like GoogleGenAI that create SDK `Chat` objects, this is a non-trivial reconstruction.

### 3. Subagent Spawning Overhead

Each subagent gets its own `JaatoSession` with its own provider instance (via `runtime.create_provider()`). Under session-owned history, the provider wouldn't need to maintain history state at all — the session would manage its own `_messages` and pass them to the provider's stateless `complete()` method. This makes subagent spawning lighter.

### 4. Cache Plugin Integration

The cache plugin (Variant A) requires the provider to expose `set_cache_plugin()` and delegate to the plugin inside `_build_api_kwargs()`, because the provider owns the data that drives cache breakpoint decisions. Under session-owned history (Variant B), the session holds the plugin directly and the provider receives pre-annotated data or calls the plugin using method parameters. See the [Cache Plugin Design doc](../jaato-cache-plugin-design.md#variant-b-session-orchestrates-future) for the detailed A→B transition.

### 5. Serialization Indirection

Session persistence currently round-trips through the provider: `provider.serialize_history(history) → JSON → provider.deserialize_history(data)`. But the `Message` type is already provider-agnostic — the provider's serializer just calls `json.dumps()` on generic `Message` objects. With session-owned history, serialization can be a session-level concern using the existing `SessionPlugin` infrastructure, without involving the provider.

---

## Current Architecture: Provider-Owned History

### Data Flow

```
JaatoSession                            ModelProviderPlugin
────────────                            ───────────────────
configure()
  ├─ provider = runtime.create_provider(model)
  ├─ provider.create_session(system, tools)     ──→ self._system_instruction = system
  │                                                  self._tools = tools
  │                                                  self._history = []
  │
send_message(prompt)
  ├─ _maybe_collect_before_send()               ← GC check
  ├─ _run_chat_loop(prompt)
  │    └─ provider.send_message(text)           ──→ self._history.append(user_msg)
  │                                                  api_call(self._history)
  │                                                  self._history.append(model_msg)
  │         ◄── ProviderResponse                ←── return response
  │
  │    # If function calls:
  │    └─ provider.send_tool_results(results)   ──→ self._history.append(tool_msg)
  │                                                  api_call(self._history)
  │                                                  self._history.append(model_msg)
  │         ◄── ProviderResponse                ←── return response
  │
  ├─ _maybe_collect_after_turn()
  │    ├─ history = provider.get_history()       ──→ return list(self._history)
  │    ├─ gc_plugin.collect(history, budget)
  │    ├─ reset_session(new_history)
  │    │    └─ provider.create_session(          ──→ self._history = new_history
  │    │         system, tools, history)              (full state reconstruction)
  │    └─ cache_plugin.on_gc_result(result)
  │
  └─ return response
```

### Provider Protocol Surface (History-Related)

From `shared/plugins/model_provider/base.py`:

| Method | Line | Statefulness | Role |
|--------|------|:---:|------|
| `create_session(system, tools, history)` | 216 | **Write** | Initialize/reset provider state |
| `get_history()` | 231 | **Read** | Return accumulated conversation |
| `send_message(message)` | 255 | **Read+Write** | Append user msg, call API, append response |
| `send_message_with_parts(parts)` | 278 | **Read+Write** | Multimodal variant |
| `send_tool_results(results)` | 297 | **Read+Write** | Append tool results, call API, append response |
| `send_message_streaming(...)` | 401 | **Read+Write** | Streaming variant with callbacks |
| `send_tool_results_streaming(...)` | 458 | **Read+Write** | Streaming tool result variant |
| `serialize_history(history)` | 348 | **Pure** | Convert `List[Message]` → str |
| `deserialize_history(data)` | 362 | **Pure** | Convert str → `List[Message]` |

All 7 methods marked **Read+Write** both read the accumulated history (to build the API request) and mutate it (to append the new user/tool/model messages). This is the coupling that the migration eliminates.

### Per-Provider History Patterns

| Provider | LOC | History Storage | Session Reconstruction | Notes |
|----------|:---:|-----------------|----------------------|-------|
| **Anthropic** | 1545 | `self._history: List[Message]` | `create_session()` replaces `_history`, `_system_instruction`, `_tools` | Base class for ZhipuAI, Ollama |
| **GoogleGenAI** | 1543 | SDK `Chat` object (internal history) + `self._history: List[Message]` (shadow copy) | Recreates `Chat` via `client.chats.create()` | **Highest complexity** — SDK manages its own history |
| **GitHub Models** | 2377 | `self._history: List[Message]` | Standard replacement | Largest provider by LOC |
| **Antigravity** | 1205 | `self._history: List[Message]` | Standard replacement | HTTP-based, similar pattern to Anthropic |
| **Claude CLI** | 1309 | `self._history: List[Message]` | Standard replacement | **Special**: CLI runs its own agentic loop in `delegated` mode |
| **ZhipuAI** | 510 | Inherits from Anthropic | Inherited | Overrides converters only |
| **Ollama** | 323 | Inherits from Anthropic | Inherited | Overrides connection only |

---

## Target Architecture: Session-Owned History

### Data Flow

```
JaatoSession                            ModelProviderPlugin
────────────                            ───────────────────
configure()
  ├─ provider = runtime.create_provider(model)
  ├─ provider.initialize(config)
  ├─ provider.connect(model)
  │                                     (provider has NO history state)
  │
send_message(prompt)
  ├─ _maybe_collect_before_send()               ← GC slices self._messages directly
  ├─ self._messages.append(user_msg)
  ├─ _run_chat_loop()
  │    └─ provider.complete(                    ──→ convert messages to API format
  │         self._messages,                          api_call(converted_messages)
  │         system, tools, ...)                      return ProviderResponse
  │         ◄── ProviderResponse                ←── (no state mutation)
  │    └─ self._messages.append(model_msg)
  │
  │    # If function calls:
  │    └─ self._messages.append(tool_result_msg)
  │    └─ provider.complete(                    ──→ convert, call, return
  │         self._messages, system, tools, ...)
  │         ◄── ProviderResponse
  │    └─ self._messages.append(model_msg)
  │
  ├─ _maybe_collect_after_turn()
  │    ├─ gc_plugin.collect(self._messages, budget)
  │    ├─ self._messages = new_messages          ← direct replacement, no provider call
  │    └─ cache_plugin.on_gc_result(result)
  │
  └─ return response
```

### Stateless `complete()` Method

The core protocol addition:

```python
def complete(
    self,
    messages: List[Message],
    system_instruction: str,
    tools: List[ToolSchema],
    *,
    response_schema: Optional[Dict[str, Any]] = None,
    cancel_token: Optional[CancelToken] = None,
    on_chunk: Optional[StreamingCallback] = None,
    on_usage_update: Optional[UsageUpdateCallback] = None,
    on_function_call: Optional[FunctionCallDetectedCallback] = None,
    on_thinking: Optional[ThinkingCallback] = None,
) -> ProviderResponse:
    """Stateless completion: convert messages to provider format, call API, return response.

    Unlike send_message(), this method does NOT modify any internal state.
    The caller (session) is responsible for maintaining the message list.

    When on_chunk is provided, the response is streamed token-by-token.
    When on_chunk is None, the response is returned in batch mode.

    Args:
        messages: Full conversation history in provider-agnostic Message format.
        system_instruction: System prompt text.
        tools: Available tool schemas.
        response_schema: Optional JSON Schema for structured output.
        cancel_token: Optional cancellation signal.
        on_chunk: If provided, enables streaming mode.
        on_usage_update: Real-time token usage callback (streaming).
        on_function_call: Callback when function call detected mid-stream.
        on_thinking: Callback for extended thinking content.

    Returns:
        ProviderResponse with text, function calls, and usage.
    """
```

Key design decisions:

1. **Unified streaming/batch**: A single `complete()` method handles both modes via the presence/absence of `on_chunk`. This halves the protocol surface compared to the current architecture which has separate `send_message()` / `send_message_streaming()` and `send_tool_results()` / `send_tool_results_streaming()` methods.

2. **No separate tool results method**: Tool results are appended to `messages` by the session before calling `complete()` again. The provider doesn't need to distinguish between "user message" and "tool results" — it converts all messages to API format uniformly.

3. **System instruction and tools per-call**: These are passed explicitly rather than set once via `create_session()`. This allows dynamic updates (e.g., deferred tool loading, system instruction enrichment) without session reconstruction.

### What Gets Removed from Protocol (Phase 4)

| Method | Replacement |
|--------|------------|
| `create_session(system, tools, history)` | `complete()` receives these as parameters |
| `get_history()` | Session owns `self._messages` |
| `send_message(message)` | `complete(messages, ...)` |
| `send_message_with_parts(parts)` | Session constructs `Message` with parts, passes to `complete()` |
| `send_tool_results(results)` | Session appends tool result `Message`, calls `complete()` |
| `send_message_streaming(...)` | `complete(..., on_chunk=callback)` |
| `send_tool_results_streaming(...)` | `complete(..., on_chunk=callback)` |

**Retained:** `serialize_history()` and `deserialize_history()` may remain as utility methods (pure functions), or move to a shared serialization module.

---

## 5-Phase Migration Plan

### Phase 1: `SessionHistory` Wrapper

**Goal:** Session holds the canonical history copy, synced bidirectionally with the provider. No behavioral change — this is a thin wrapper that makes the ownership explicit.

**Changes:**

```python
# shared/session_history.py (new file, ~60 LOC)
class SessionHistory:
    """Canonical conversation history owned by the session.

    Wraps List[Message] with mutation tracking. During the migration,
    changes are synced to the provider via create_session(). After Phase 4,
    the provider sync is removed.
    """

    def __init__(self) -> None:
        self._messages: List[Message] = []
        self._dirty: bool = False

    def append(self, msg: Message) -> None:
        self._messages.append(msg)
        self._dirty = True

    def replace(self, messages: List[Message]) -> None:
        self._messages = list(messages)
        self._dirty = True

    @property
    def messages(self) -> List[Message]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
```

```python
# JaatoSession changes (~30 LOC)
class JaatoSession:
    def __init__(self, ...):
        self._history = SessionHistory()  # New canonical copy

    def get_history(self) -> List[Message]:
        # Phase 1: return session's copy (identical to provider's)
        return self._history.messages

    def reset_session(self, history=None):
        if history:
            self._history.replace(history)
        else:
            self._history.replace([])
        # Still sync to provider during Phase 1
        self._create_provider_session(history)
```

**Scope:** ~100 LOC new, ~30 LOC modified in session.

**Risk:** Low — behavioral no-op. The session just caches what it was already reading via `provider.get_history()`.

**Validation:** Existing tests pass unchanged. Add a test asserting `session.get_history() == provider.get_history()` after every operation.

### Phase 2: Stateless `complete()` on Anthropic Provider

**Goal:** Add the stateless `complete()` method to `AnthropicProvider` alongside existing methods. Session gains a code path that calls `complete()` instead of `send_message()` when the provider supports it.

**Why Anthropic first:** It's the base class for ZhipuAI and Ollama (3 providers for the price of 1), and it unblocks cache plugin Variant B.

**Changes to `AnthropicProvider`:**

```python
def complete(
    self,
    messages: List[Message],
    system_instruction: str,
    tools: List[ToolSchema],
    *,
    response_schema=None,
    cancel_token=None,
    on_chunk=None,
    on_usage_update=None,
    on_function_call=None,
    on_thinking=None,
) -> ProviderResponse:
    """Stateless completion without history mutation."""
    # Convert to Anthropic format (reuse existing converters)
    api_messages = messages_to_anthropic(messages)
    api_tools = tools_to_anthropic(tools)
    api_system = [{"type": "text", "text": system_instruction}]

    # Cache plugin annotation (reuses existing delegation)
    if self._cache_plugin:
        cache_result = self._cache_plugin.prepare_request(
            system=api_system, tools=api_tools,
            messages=api_messages, budget=None
        )
        api_system = cache_result["system"]
        api_tools = cache_result["tools"]
        api_messages = cache_result["messages"]

    # Build kwargs (reuse _build_system_blocks, _build_tool_list patterns)
    kwargs = {
        "model": self._model_name,
        "system": api_system,
        "messages": api_messages,
        "max_tokens": self._max_tokens,
    }
    if api_tools:
        kwargs["tools"] = api_tools

    # Streaming or batch
    if on_chunk:
        return self._stream_complete(kwargs, on_chunk, cancel_token,
                                      on_usage_update, on_function_call, on_thinking)
    else:
        return self._batch_complete(kwargs)
```

**Changes to `JaatoSession`:**

```python
def _run_chat_loop(self, message, on_output, on_usage_update=None):
    # Check if provider supports stateless complete()
    if hasattr(self._provider, 'complete'):
        return self._run_chat_loop_stateless(message, on_output, on_usage_update)
    else:
        return self._run_chat_loop_legacy(message, on_output, on_usage_update)
```

The `_run_chat_loop_stateless()` method manages `self._history` directly:
1. Appends user message to `self._history`
2. Calls `provider.complete(self._history.messages, system, tools, ...)`
3. Appends model response to `self._history`
4. If function calls: executes tools, appends results to `self._history`, loops

**Scope:** ~200 LOC in Anthropic provider, ~150 LOC in session.

**Risk:** Medium — dual code paths. Mitigated by feature flag (`JAATO_SESSION_OWNED_HISTORY=true`, default `false`). Existing tests pass on legacy path; new tests cover stateless path.

**Validation:**
- All existing Anthropic tests pass (legacy path)
- New tests verify `complete()` returns same response as `send_message()` for same input
- Integration test: full turn with tools via `complete()`

**Unblocks:** Cache plugin Variant B (A→B transition, ~60 lines mechanical).

### Phase 3: Migrate Remaining Providers

**Goal:** Add `complete()` to all remaining providers. Session switches to stateless path by default.

| Provider | Complexity | Notes |
|----------|:---:|-------|
| **ZhipuAI** | Low | Inherits from Anthropic — `complete()` inherited for free |
| **Ollama** | Low | Inherits from Anthropic — `complete()` inherited for free |
| **GitHub Models** | Medium | Similar structure to Anthropic, different SDK (`azure-ai-inference`) |
| **Antigravity** | Medium | HTTP-based, needs to construct request from parameters |
| **Claude CLI** | Special | In `delegated` mode, CLI manages its own agentic loop. `complete()` may wrap `claude --print` for single-turn completion. In `passthrough` mode, standard implementation. |
| **GoogleGenAI** | High | SDK `Chat` object manages history internally. `complete()` must bypass the Chat abstraction and call `client.models.generate_content()` directly with the full message list. |

**GoogleGenAI Deep Dive:**

The GoogleGenAI SDK provides a `Chat` object that manages history internally. Currently, `GoogleGenAIProvider` uses `self._chat.send_message()` which appends to the Chat's internal history. For `complete()`, the provider must bypass the Chat and call the lower-level `generate_content()` API:

```python
# GoogleGenAIProvider.complete()
def complete(self, messages, system_instruction, tools, **kwargs):
    # Convert to SDK format
    sdk_messages = history_to_sdk(messages)
    sdk_tools = tools_to_sdk(tools)

    # Bypass Chat — call generate_content directly
    response = self._client.models.generate_content(
        model=self._model_name,
        contents=sdk_messages,
        config=GenerateContentConfig(
            system_instruction=system_instruction,
            tools=sdk_tools,
            # ... other config
        )
    )
    return self._convert_response(response)
```

This is the highest-risk provider because it requires bypassing the SDK's intended usage pattern (Chat-based) in favor of the stateless API.

**Scope:** ~600 LOC across 4 providers (ZhipuAI/Ollama are free; GoogleGenAI ~200, GitHub Models ~150, Antigravity ~150, Claude CLI ~100).

**Risk:** Medium for GoogleGenAI (SDK bypass), low for others.

**Validation:** Each provider gets `complete()` tests mirroring existing `send_message()` tests.

### Phase 4: Remove Legacy Protocol Methods

**Goal:** Delete the stateful methods from `ModelProviderPlugin` and all implementations.

**Methods removed from protocol:**

```python
# These are removed from ModelProviderPlugin (base.py)
def create_session(self, system_instruction, tools, history) -> None
def get_history(self) -> List[Message]
def send_message(self, message) -> ProviderResponse
def send_message_with_parts(self, parts) -> ProviderResponse
def send_tool_results(self, results) -> ProviderResponse
def send_message_streaming(self, message, on_chunk, ...) -> ProviderResponse
def send_tool_results_streaming(self, results, on_chunk, ...) -> ProviderResponse
```

**Retained:**

```python
# These stay (not history-related)
def initialize(self, config) -> None
def connect(self, model) -> None
def verify_auth(self, ...) -> bool
def shutdown(self) -> None
def complete(self, messages, system, tools, ...) -> ProviderResponse  # NEW
def count_tokens(self, content) -> int
def get_context_limit(self) -> int
def get_token_usage(self) -> TokenUsage
def serialize_history(self, history) -> str      # May stay as utility
def deserialize_history(self, data) -> List[Message]  # May stay as utility
# ... capabilities, agent context, thinking, error classification
```

**Per-provider removal estimates:**

| Provider | Lines Removed | Notes |
|----------|:---:|-------|
| Anthropic | ~300 | `create_session`, `send_message`, `send_message_streaming`, `send_tool_results`, `send_tool_results_streaming`, history management |
| GoogleGenAI | ~400 | Same + Chat object management |
| GitHub Models | ~350 | Same methods, larger due to detailed error handling |
| Antigravity | ~200 | Same pattern |
| Claude CLI | ~150 | Same, but some CLI-specific session management |
| ZhipuAI | ~50 | Mostly inherited, override removals |
| Ollama | ~30 | Mostly inherited |

**Total removed:** ~1480 lines across all providers.

**Session changes:**

```python
# Removed
def _create_provider_session(self, history) -> None  # ~30 lines
# reset_session no longer calls provider

# Modified
def reset_session(self, history=None):
    if history:
        self._history.replace(history)
    else:
        self._history.replace([])
    self._turn_accounting = []
    # No provider.create_session() call
```

**Scope:** ~1500 LOC removed, ~50 LOC modified.

**Risk:** High — breaking change to provider protocol. All providers and tests must be updated simultaneously.

**Validation:** Full test suite must pass. No provider should reference `self._history` after this phase.

### Phase 5: Simplify Session Internals

**Goal:** Clean up session code that worked around provider-owned history.

**Simplifications:**

1. **GC flow simplification:**

```python
# Before (current)
def _maybe_collect_after_turn(self):
    history = self.get_history()                    # Read from provider
    new_history, result = gc_plugin.collect(history, ...)
    self.reset_session(new_history)                 # Write back to provider
    self._apply_gc_removal_list(result)

# After (Phase 5)
def _maybe_collect_after_turn(self):
    new_messages, result = gc_plugin.collect(self._history.messages, ...)
    self._history.replace(new_messages)             # Direct mutation
    self._apply_gc_removal_list(result)
```

2. **Budget sync simplification:** `_populate_instruction_budget()` reads from `self._history` directly instead of `provider.get_history()`.

3. **Serialization simplification:** `SessionPlugin` serializes `session._history.messages` directly, without going through `provider.serialize_history()`.

4. **Cache plugin wiring (Variant B):**

```python
# Before (Variant A)
def _wire_cache_plugin(self):
    cache_plugin = load_cache_plugin_for_provider(provider_name, config)
    if hasattr(self._provider, 'set_cache_plugin'):
        self._provider.set_cache_plugin(cache_plugin)  # Provider holds reference
    self._cache_plugin = cache_plugin

# After (Variant B)
def _wire_cache_plugin(self):
    cache_plugin = load_cache_plugin_for_provider(provider_name, config)
    self._cache_plugin = cache_plugin  # Session holds reference only
    # No provider involvement — provider calls plugin via complete() parameters
```

**Scope:** ~300 LOC simplified/removed.

**Risk:** Low — cleanup of already-migrated code.

---

## Impact on Dependent Systems

### Cache Plugin (A→B Transition)

The cache plugin `CachePlugin` protocol requires **no changes**. The `prepare_request(messages, tools, system, budget)` signature takes explicit parameters, not provider state. Only the **orchestration** changes:

| Aspect | Variant A (current) | Variant B (after Phase 2+) |
|--------|--------------------|-----------------------------|
| Who holds plugin | Provider + Session | Session only |
| Who calls `prepare_request()` | Provider inside `_build_api_kwargs()` | Provider inside `complete()` (data from parameters) |
| `set_cache_plugin()` | On provider protocol | Removed |
| Budget source | `cache_plugin.set_budget()` from session | Same (unchanged) |
| GC notification | `cache_plugin.on_gc_result()` from session | Same (unchanged) |

**Transition effort:** ~60 lines, mechanical. See [Sequencing Impact Analysis](cache-plugin-sequencing-impact-analysis.md#2-what-the-ab-transition-requires).

### GC System

GC plugins already receive `List[Message]` and return `(List[Message], GCResult)`. They don't interact with providers. The only change is **how the session applies the result**:

| Current | After Phase 5 |
|---------|---------------|
| `self.reset_session(new_history)` → `provider.create_session(history=...)` | `self._history.replace(new_messages)` |

The GC plugin protocol, all 4 implementations (`gc_budget`, `gc_truncate`, `gc_summarize`, `gc_hybrid`), and `GCResult`/`GCRemovalItem` types are **completely unaffected**.

### Subagent Architecture

Subagents currently share the parent's `JaatoRuntime` but get their own `JaatoSession` + own provider instance. Under session-owned history:

- Subagent gets its own `SessionHistory` (isolated state) — no change
- Subagent's provider instance doesn't need to hold history — lighter footprint
- `create_session()` calls are eliminated from subagent initialization

### Session Persistence (`SessionPlugin`)

Current flow:
```
Save:    session → provider.get_history() → provider.serialize_history(history) → disk
Restore: disk → provider.deserialize_history(data) → provider.create_session(history=...)
```

After migration:
```
Save:    session._history.messages → serialize(messages) → disk
Restore: disk → deserialize(data) → session._history.replace(messages)
```

The `serialize_history()` / `deserialize_history()` methods on the provider protocol could be retained as utility functions, moved to a shared module, or the existing `shared/plugins/session/serializer.py` could be extended to handle `Message` serialization directly (it may already do so).

### Telemetry

Minor updates to span attributes. Currently, spans wrap `provider.send_message()` calls. After migration, spans wrap `provider.complete()` calls. The `llm_telemetry` span in `_run_chat_loop()` would be placed around `provider.complete()` instead of `provider.send_message_streaming()` / `provider.send_message()`.

---

## Risk Analysis

### High Risk: GoogleGenAI SDK Chat Bypass

The GoogleGenAI SDK's `Chat` object manages history internally and provides convenience methods. Bypassing it to call `generate_content()` directly means:
- Manual content construction (SDK `Content` objects)
- Loss of SDK-level history validation
- Potential incompatibility with future SDK versions

**Mitigation:** The `history_to_sdk()` converter already exists. The `generate_content()` API is stable and well-documented. Test coverage should include round-trip verification (same input via Chat vs direct call produces same output).

### Medium Risk: Claude CLI Delegated Mode

In `delegated` mode, the Claude CLI runs its own agentic loop — jaato sends a prompt and the CLI decides which tools to call, executes them, and returns the final result. The `complete()` abstraction assumes the caller (session) controls the tool execution loop, which conflicts with delegated mode.

**Options:**
1. `complete()` in delegated mode wraps the entire CLI invocation as a single API call (no tool loop control)
2. Delegated mode is exempt from the migration and keeps `send_message()` as a special case
3. `complete()` receives a `mode` parameter that controls whether to return after the first model response or run the full CLI loop

**Recommendation:** Option 1 (wrap as single call). The `passthrough` mode can use standard `complete()` implementation.

### Medium Risk: Dual Code Paths (Phase 2-3)

During the transition, the session has both `_run_chat_loop_legacy()` and `_run_chat_loop_stateless()`. Both must be maintained and tested until Phase 4 removes the legacy path.

**Mitigation:** Feature flag (`JAATO_SESSION_OWNED_HISTORY`). Legacy is default; stateless is opt-in. Phase 4 flips the default and removes legacy.

### Low Risk: Message Format Consistency

The `Message` type is already provider-agnostic and used consistently across the codebase. Providers convert to/from API-specific formats in their converters. This doesn't change under session-owned history — the converters are still called inside `complete()`.

---

## Dependencies and Sequencing

```
Phase 1 (SessionHistory wrapper)           ← No dependencies
    │
Phase 2 (Anthropic complete())             ← Depends on Phase 1
    │
    ├─── Cache Plugin A→B transition       ← Unblocked by Phase 2
    │
Phase 3 (All providers complete())         ← Depends on Phase 2
    │
Phase 4 (Remove legacy methods)            ← Depends on Phase 3
    │
Phase 5 (Simplify session internals)       ← Depends on Phase 4
```

**Phase 1** can be implemented immediately with zero behavioral change.

**Phase 2** is the critical milestone — it unblocks the cache plugin A→B transition and proves the `complete()` pattern works for the most complex provider (Anthropic is the base class with cache plugin integration, extended thinking, and streaming).

**Phases 3-5** can be executed incrementally, one provider at a time.

---

## Appendix: Provider Method Inventory

Methods on each provider that would be affected by the migration (methods that read or write `self._history`):

### AnthropicProvider (1545 LOC)

| Method | Reads History | Writes History | LOC |
|--------|:---:|:---:|:---:|
| `create_session()` | | Yes | ~20 |
| `get_history()` | Yes | | ~5 |
| `send_message()` | Yes | Yes | ~60 |
| `send_message_streaming()` | Yes | Yes | ~80 |
| `send_tool_results()` | Yes | Yes | ~40 |
| `send_tool_results_streaming()` | Yes | Yes | ~50 |
| `_build_api_kwargs()` | Yes | | ~40 |
| `_compute_history_cache_breakpoint()` | Yes | | ~20 |
| **Total affected** | | | **~315** |

### GoogleGenAIProvider (1543 LOC)

| Method | Reads History | Writes History | LOC |
|--------|:---:|:---:|:---:|
| `create_session()` | | Yes | ~30 |
| `get_history()` | Yes | | ~5 |
| `send_message()` | Yes | Yes | ~80 |
| `send_message_streaming()` | Yes | Yes | ~100 |
| `send_tool_results()` | Yes | Yes | ~60 |
| `send_tool_results_streaming()` | Yes | Yes | ~70 |
| `_build_messages()` | Yes | | ~40 |
| `_chat` management | Yes | Yes | ~30 |
| **Total affected** | | | **~415** |

### GitHubModelsProvider (2377 LOC)

| Method | Reads History | Writes History | LOC |
|--------|:---:|:---:|:---:|
| `create_session()` | | Yes | ~20 |
| `get_history()` | Yes | | ~5 |
| `send_message()` | Yes | Yes | ~70 |
| `send_message_streaming()` | Yes | Yes | ~90 |
| `send_tool_results()` | Yes | Yes | ~50 |
| `send_tool_results_streaming()` | Yes | Yes | ~60 |
| **Total affected** | | | **~295** |

### Summary

| Provider | Affected LOC | Phase |
|----------|:---:|:---:|
| Anthropic | ~315 | 2 |
| ZhipuAI | ~50 (overrides) | 3 (inherits from Phase 2) |
| Ollama | ~30 (overrides) | 3 (inherits from Phase 2) |
| GoogleGenAI | ~415 | 3 |
| GitHub Models | ~295 | 3 |
| Antigravity | ~200 | 3 |
| Claude CLI | ~150 | 3 |
| **Total** | **~1455** | |
