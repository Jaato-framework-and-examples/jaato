# Impact Analysis: Session-Owned Message History

**Date:** 2025-02-15
**Status:** Analysis / Pre-implementation
**Pattern source:** pi-mono comparison — "Context as Plain Serializable Data"

## Executive Summary

This document reassesses the proposal to invert message history ownership from the
provider to the session. After deep analysis of the actual codebase, the conclusion
is nuanced: **the principle is sound and the benefits are real, but the implementation
is materially harder than the original comparison suggested** due to provider-specific
behaviors that are tightly coupled to history mutation. A phased approach is
recommended.

---

## 1. Current Architecture (As-Built)

### Ownership Model

```
JaatoSession  ──get_history()──>  Provider._history (or SDK chat object)
              <──List[Message]──
```

The session **delegates** all history storage to the provider. It never holds its
own `messages` list. Every read goes through `provider.get_history()`, every write
goes through `provider.send_message()` / `provider.send_tool_results()` (which
internally append), and every replacement goes through `provider.create_session(history=...)`.

### Provider Variance

| Provider | Storage Mechanism | Append Timing | Special Behaviors |
|----------|-------------------|---------------|-------------------|
| **Google GenAI** | SDK `chat` object (`self._chat`) | Inside SDK, opaque | History lives in SDK; `get_history()` converts on read |
| **Anthropic** | `self._history: List[Message]` | Before API call | `validate_tool_use_pairing()`, `_compute_history_cache_breakpoint()` |
| **Ollama** | Inherited from Anthropic | Before API call | Same as Anthropic (subclass) |
| **GitHub Models** | `self._history: List[Message]` | Before API call | Dual code paths (Copilot API vs Azure SDK) |
| **Antigravity** | `self._history: List[Message]` | Before API call | Token account refresh tied to message flow |
| **Claude CLI** | `self._history: List[Message]` | **After** API call | CLI session ID (`--resume`) must be preserved across turns |

### Key Observation

Five of six providers already use a plain `List[Message]` that they manage
themselves. Google GenAI is the outlier, using the SDK's stateful chat object.
This means the "provider owns history" pattern is already somewhat accidental
for most providers — they maintain the list because the protocol requires it,
not because they need statefulness.

---

## 2. What "Session Owns History" Means Concretely

### Target State

```python
class JaatoSession:
    _messages: List[Message]  # Session owns the canonical message list

    def send_message(self, message: str, ...) -> str:
        # 1. Append user message to self._messages
        # 2. Pass snapshot to provider: provider.complete(self._messages, ...)
        # 3. Receive ProviderResponse
        # 4. Append model response to self._messages
        # 5. If function calls: execute tools, append results, loop

    def get_history(self) -> List[Message]:
        return list(self._messages)  # Local read, no delegation

    def reset_session(self, history: Optional[List[Message]] = None) -> None:
        self._messages = list(history) if history else []
        # Provider state: nothing to recreate
```

### New Provider Contract

```python
class ModelProviderPlugin(Protocol):
    def complete(
        self,
        messages: List[Message],
        system_instruction: Optional[str],
        tools: Optional[List[ToolSchema]],
        response_schema: Optional[Dict] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk: Optional[StreamingCallback] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> ProviderResponse:
        """Stateless completion: takes full context, returns response."""
```

The provider receives an immutable snapshot of messages and returns a response.
No `create_session()`, no `get_history()`, no internal `_history` list.

---

## 3. Benefit Analysis (Reassessed)

### 3.1 GC Becomes a Local List Operation

**Current:** GC reads history from provider, computes a new list, then calls
`reset_session(new_history)` which destroys and recreates the provider session
(including the Google GenAI chat object).

**After:** GC slices `session._messages` in place. No provider reconstruction.
No risk of SDK state corruption.

**Impact: HIGH.** This is the clearest win. The current GC flow
(`_maybe_collect_before_send` / `_maybe_collect_after_turn`) goes through
`reset_session` -> `_create_provider_session` -> `provider.create_session(history)`.
This is expensive for Google GenAI (creates a new chat object with full history
conversion) and fragile (any SDK state besides history is lost).

### 3.2 Context Limit Recovery Simplifies

**Current:** `_send_tool_results_and_continue` catches context limit errors and
must reach into the provider's internals:

```python
# jaato_session.py:3872 — reaches into provider._history directly
history = getattr(self._provider, '_history', None)
```

This `getattr` on a private field is a code smell that screams "wrong layer owns
this data." The session needs to remove tool results from history for retry, but
can't do it cleanly because it doesn't own the list.

**After:** Session removes from its own `_messages` list. No private field access.
No `getattr`.

**Impact: HIGH.** Eliminates a fragile coupling point.

### 3.3 Session Persistence Becomes Trivial

**Current:** Persistence goes through `provider.serialize_history()` /
`provider.deserialize_history()`, but the actual serializer
(`shared/plugins/session/serializer.py`) already works with `List[Message]` —
provider-agnostic types. The provider methods are pass-throughs.

**After:** Session serializes `self._messages` directly. No provider round-trip.

**Impact: MEDIUM.** The current path already works; this just removes an
unnecessary indirection. But it does open the door to session persistence
that survives provider swaps.

### 3.4 Cross-Provider Handoff

**Current:** Switching providers mid-conversation requires: read history from
old provider -> destroy old session -> create new provider -> create session
with history. If the two providers serialize `Message` differently, data can
be lost.

**After:** Session keeps `_messages` intact. Swap the provider. Next `complete()`
call just passes the same list to the new provider.

**Impact: MEDIUM.** Not a frequently requested feature, but architecturally
significant. Enables dynamic model routing (e.g., use a cheaper model for
simple turns, switch to a stronger model for complex reasoning).

### 3.5 History Indexing, Caching, Slicing

**Current:** `get_turn_boundaries()`, `revert_to_turn()`, `_update_conversation_budget()`
all call `self.get_history()` which calls `provider.get_history()` which copies
the list. Multiple calls per turn = multiple copies.

**After:** Direct access to `self._messages`. Zero copies for read operations.

**Impact: LOW-MEDIUM.** Performance is not a bottleneck here (lists are small
relative to API latency), but the code clarity improvement is meaningful.

### 3.6 Provider Simplification

**Current:** Every provider implements 5 history-related methods: `create_session`,
`get_history`, `send_message`, `send_tool_results`, `send_message_streaming`,
`send_tool_results_streaming`. Each must correctly append user messages, validate
pairing, add responses, handle rollback on error.

**After:** Provider implements one method: `complete()` (plus a streaming variant).
No state management, no rollback logic, no pairing validation.

**Impact: HIGH.** This is where the largest code reduction happens. The
Anthropic provider alone has ~6 places where it calls `validate_tool_use_pairing`
and `_compute_history_cache_breakpoint` in identical patterns.

---

## 4. Obstacle Analysis (What Makes This Hard)

### 4.1 Google GenAI SDK Chat Object

The Google GenAI provider uses the SDK's `chats.create()` which returns a stateful
chat object that owns history internally. There is no "pass messages, get response"
stateless API in the current SDK.

**Mitigation options:**
- Use `models.generate_content()` instead of `chats.create()` — this is the
  stateless API. We'd pass the full message list each time.
- Keep a thin adapter that recreates the chat object from session messages on
  each call (equivalent to current `create_session` but happening transparently).

**Difficulty: MEDIUM.** The stateless `generate_content()` API exists and supports
all the same features. The chat object is a convenience wrapper, not a necessity.
The main work is converting `List[Message]` to the SDK's content format each call,
which `history_to_sdk()` already does.

### 4.2 Anthropic Prompt Caching

The Anthropic provider computes `_compute_history_cache_breakpoint()` which
examines the history to find the optimal place to insert a `cache_control`
marker. This is a provider-specific optimization that needs access to the
message list structure.

**Mitigation:** This logic doesn't need to *own* history — it just needs to
*read* it. The `complete()` method receives the full message list as input.
The provider can compute the cache breakpoint from that input before converting
to API format.

```python
def complete(self, messages: List[Message], ...) -> ProviderResponse:
    breakpoint = self._compute_cache_breakpoint(messages)
    api_messages = messages_to_anthropic(messages, cache_breakpoint_index=breakpoint)
    ...
```

**Difficulty: LOW.** The function already operates on `List[Message]` input.
Moving it from "operate on `self._history`" to "operate on `messages` parameter"
is mechanical.

### 4.3 Anthropic Tool-Use Pairing Validation

`validate_tool_use_pairing()` repairs history when a cancellation leaves
dangling tool_use blocks without matching tool_result blocks. Currently called
inside every provider `send_*` method.

**Mitigation:** Move validation to the session layer. The session already has
`ensure_tool_call_integrity()` for GC — this is the same concept. Validation
happens on `self._messages` before passing to provider.

**Difficulty: LOW.** The validation function already takes `List[Message]`
as input and returns `List[Message]`. It's provider-agnostic in implementation
despite living in the Anthropic package.

### 4.4 Provider Error Rollback

Current providers append the user message *before* the API call, then pop it
on failure:

```python
self._history.append(user_msg)
try:
    response = api_call(...)
    self._history.append(response_msg)
except:
    self._history.pop()  # rollback
```

**Mitigation:** With session-owned history, the session controls append timing.
Append only on success:

```python
response = provider.complete(self._messages + [user_msg], ...)
# Only append if successful:
self._messages.append(user_msg)
self._messages.append(response_to_message(response))
```

**Difficulty: LOW.** Simpler than current rollback logic. The temporary
`messages + [user_msg]` is a cheap list concatenation for the sizes involved.

### 4.5 Claude CLI Session ID Preservation

The Claude CLI provider preserves `self._cli_session_id` across turns for the
`--resume` flag. This is provider state that isn't in the message list.

**Mitigation:** Provider can hold non-history state. The `complete()` method
is stateless with respect to *messages*, but the provider object can still
maintain connection state (session IDs, client handles, etc.).

**Difficulty: NONE.** This is orthogonal to history ownership.

### 4.6 Mid-Turn Interrupt and Cancellation

`_notify_model_of_cancellation()` (line 4685) constructs a history with an
injected system note and calls `_create_provider_session(new_history)`. With
session-owned history, this becomes `self._messages.append(cancel_note)`.

**Difficulty: NONE.** Simpler.

### 4.7 `send_message_with_parts` and Multimodal

The session has a `send_message_with_parts()` path for multimodal input. Under
the new model, this translates to building a `Message` with multimodal `Part`
objects and including it in the snapshot passed to `complete()`.

**Difficulty: LOW.** Same pattern as text messages.

---

## 5. Migration Risk Assessment

### What Breaks If Done Wrong

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Token counting diverges from actual API payload | HIGH | MEDIUM | Provider returns `TokenUsage`; session updates budget from that |
| Google GenAI conversion loses fidelity | HIGH | LOW | `history_to_sdk()` already round-trips; add integration tests |
| Prompt caching stops working (Anthropic) | MEDIUM | LOW | Keep `_compute_cache_breakpoint` in provider, test cache hits |
| Tool pairing breaks across providers | HIGH | LOW | Move `validate_tool_use_pairing` to shared, test all providers |
| Session persistence format changes | LOW | LOW | Serializer already uses `List[Message]` — no format change |
| Subagent history isolation regresses | MEDIUM | LOW | Subagents already get separate sessions; history follows |

### Test Coverage Requirements

- **GC round-trip test:** GC -> slice messages -> next turn succeeds (all providers)
- **Context limit recovery:** Truncate tool results -> retry succeeds
- **Prompt caching:** Verify Anthropic cache_read_tokens > 0 after migration
- **Cross-provider swap:** Send N turns to provider A, swap to B, continue
- **Cancellation + resume:** Cancel mid-stream, next turn sees correct history
- **Session persistence:** Save -> load -> continue (verify message_ids survive)

---

## 6. Proposed Implementation Phases

### Phase 1: Introduce `SessionHistory` (Low Risk, High Clarity)

Create a thin wrapper that the session owns:

```python
@dataclass
class SessionHistory:
    messages: List[Message] = field(default_factory=list)

    def append(self, msg: Message) -> None: ...
    def slice(self, start: int, end: int) -> List[Message]: ...
    def snapshot(self) -> List[Message]: return list(self.messages)
    def replace(self, new: List[Message]) -> None: self.messages = list(new)
```

Session uses `self._history = SessionHistory()`. For now, it still syncs
with the provider via `create_session()` after every mutation. This is
a **refactoring step** — behavior is identical, but history has a clear owner.

### Phase 2: Add Stateless `complete()` to Provider Protocol

Add `complete(messages, system, tools, ...) -> ProviderResponse` alongside
existing methods. Implement for Anthropic first (simplest — already uses
`List[Message]` internally).

Session switches to calling `complete()` where available, falling back to
old methods where not. This is a **parallel path** — both APIs coexist.

### Phase 3: Migrate Remaining Providers

- **GitHub Models, Antigravity:** Straightforward — same pattern as Anthropic.
- **Ollama:** Inherits from Anthropic, gets it for free.
- **Google GenAI:** Switch from `chats.create()` to `models.generate_content()`.
- **Claude CLI:** Keep session ID state; make `complete()` pass messages as
  conversation context to CLI.

### Phase 4: Remove Legacy Provider Methods

Remove `create_session()`, `get_history()`, `send_message()`,
`send_tool_results()` from the protocol. Clean up all sync points in session.

### Phase 5: Simplify Session Internals

- `reset_session()` becomes `self._history.replace(new_messages)` — no provider call
- `_remove_tool_results_from_history()` becomes `self._history.messages.pop()` — no getattr hack
- `_notify_model_of_cancellation()` becomes `self._history.append(note)` — no provider session recreation
- GC operates directly on `self._history.messages`

---

## 7. Lines-of-Code Impact Estimate

| Component | Current LOC | Estimated After | Change |
|-----------|-------------|-----------------|--------|
| `jaato_session.py` (history-related) | ~300 | ~150 | -50% |
| Provider protocol (`base.py`) | 100 | 60 | -40% |
| Anthropic provider | ~600 (send_* methods) | ~250 (complete) | -58% |
| GitHub Models provider | ~400 (send_* methods) | ~180 (complete) | -55% |
| Antigravity provider | ~350 (send_* methods) | ~160 (complete) | -54% |
| Google GenAI provider | ~300 (send_* methods) | ~200 (complete) | -33% |
| Claude CLI provider | ~250 (send_* methods) | ~180 (complete) | -28% |
| GC plugins | ~50 (reset_session calls) | ~20 (direct mutation) | -60% |
| **Total estimated** | **~2350** | **~1200** | **~-49%** |

---

## 8. Verdict

### Original Claim Reassessed

> "This is the single highest-impact architectural change we could make."

**Confirmed, with caveats.** The original comparison correctly identified the
core issue: the session doesn't own its most important data structure. But
the implementation is not a simple "move the list" — it requires:

1. A new provider contract (`complete()` replacing 6 methods)
2. Careful handling of prompt caching (Anthropic)
3. Google GenAI SDK API switch (`chats` -> `generate_content`)
4. Moving validation logic (`validate_tool_use_pairing`) to shared code
5. Comprehensive test coverage across all providers

The phased approach (5 phases) reduces risk to near-zero at each step while
delivering incremental value. Phase 1 alone (SessionHistory wrapper) provides
immediate code clarity benefits with zero behavioral change.

### Recommendation

**Proceed with Phase 1 immediately.** It's low-risk, clarifies ownership, and
creates the foundation for subsequent phases. Phases 2-3 can follow
incrementally per provider as time allows. The Anthropic provider is the
best candidate for Phase 2 pilot due to its existing `List[Message]` storage.
