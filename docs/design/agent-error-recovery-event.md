# Agent Error Recovery Event (`AgentErrorEvent` + `on_agent_error`)

**Status:** Proposed (design + implementation plan)
**Origin:** kb-orchestrator ask #5 — make terminal agent errors *reactor-managed*
instead of *cascade-fatal*.

## Problem

When an agent's model thread hits a terminal error, the framework
**unconditionally terminates the session** with no recovery surface:

```python
# server/core.py:4426-4441  (model-thread finally block)
if terminal_error is not None:
    server.emit(SessionTerminatedEvent(
        session_id=..., agent_id=..., reason="error",
        error_summary=str(terminal_error),
        error_type=type(terminal_error).__name__,
    ))
    clear_logging_context()
    return            # <-- session dead, no first refusal
```

The only signal a reactor receives is `SessionTerminatedEvent(reason="error")`
— a generic *teardown* event also used for `natural` / `stopped` /
`cascade_cancelled`. By the time it arrives the session is already gone, so a
cascade driver can only do one thing: write `cascade_aborted.json` and die.
This is the single pipeline path that is **not** reactor-driven.

**Evidence (kb-orchestrator, one session):** `host_validator` on a gpt5mini
cascade was killed 3× — `NudgeExhausted`, and 2× OpenAI 500-class `APIError`
(`req_b2e1179920…`, `req_5aab6d83…`) — each aborting 15–30 min of work. A
reactor able to recover the stage would have saved every one.

## Principle: two layers, reactor sees only the residual

Recovery is **not** retry-the-same-request. It is stage-level: re-spawn,
reroute to another model/provider, escalate, or abort-this-stage. It applies
whether or not the error was ever transient.

| Layer | Owner | Behavior | Visible to reactor? |
|-------|-------|----------|---------------------|
| 1 — automatic | framework (`with_retry`, nudge) | per-request retry of errors it knows how to retry | **No** — succeeds silently |
| 2 — recovery | framework emits event → reactor acts | fires only when Layer 1 is **exhausted** or never applied | **Yes** — `AgentErrorEvent` |
| 3 — policy | kb reactor (`cascade_on_error.py`) | decide re-spawn / reroute / escalate / abort + caps | — |

**The terminal chokepoint already _is_ the Layer-1-exhaustion boundary.**
`core.py:4426` is, by construction, where an exception *escaped `with_retry`*
(its own comment says so); `NudgeExhausted` (`core.py:4619`) is the nudge
mechanism *exhausting*; bootstrap failure (`runner_spawn.py:770`) has no
auto-retry to wait on. So emitting at these three sites satisfies "fire only
after the framework's automatic management is out of moves" for free. The
reactor is never bothered with what the framework resolved itself.

**Classification is an optional hint, never a gate.** The event fires for
*every* terminal agent error. A coarse `classification` field may ride along to
*inform* kb policy, but it must never condition whether the event fires or
whether recovery is offered.

## The contract

### `AgentErrorEvent` (SDK — `jaato-sdk/jaato_sdk/events.py`)

First-class event, single source of truth (server imports + emits it).
Mirrors `SessionTerminatedEvent`'s error fields and adds recovery-relevant ones:

| Field | Type | Meaning |
|-------|------|---------|
| `session_id` | str | failed session |
| `agent_id` | str | failed agent / cascade stage |
| `error_type` | str | exception class name (`APIError`, `RunnerCallError`, `NudgeExhausted`, …) |
| `error_summary` | str | human-readable cause |
| `request_id` | Optional[str] | provider request id (OpenAI `req_…`, etc.) for observability |
| `attempt` | int | **reactor-level** re-spawn count (see below) — kb-owned, echoed |
| `classification` | Optional[str] | coarse hint: `transient_provider` / `fatal_contract` / `unknown`. **Non-gating.** |
| `framework_retries_exhausted` | Optional[int] | informational: auto-retries Layer 1 already burned |
| `occurred_at` | float | emit timestamp |

There is no `recoverable` flag — *every* `AgentErrorEvent` is a recovery point
by definition.

### `on_agent_error` hook (`shared/plugins/subagent/ui_hooks.py`)

Symmetric with `on_agent_completed`; fire-and-forget (returns `None`):

```python
def on_agent_error(
    self,
    agent_id: str,
    error_type: str,
    error_summary: str,
    *,
    session_id: str,
    request_id: Optional[str] = None,
    attempt: int = 0,
    classification: Optional[str] = None,
    framework_retries_exhausted: Optional[int] = None,
    occurred_at: Optional[float] = None,
) -> None: ...
```

The daemon-side `ServerAgentHooks` implementation (`core.py:2806+`) emits
`AgentErrorEvent` in its body — exactly as `on_agent_completed` emits
`AgentCompletedEvent` (`core.py:2920`).

### Two distinct "attempt" counters — do not conflate

- `with_retry`'s internal per-request attempts → **framework's business, never
  surfaced.** The reactor neither sees nor counts them.
- The event's `attempt` → the **reactor-level re-spawn count** (how many times
  the reactor has recovered this stage). The kb passes it at spawn time via
  `agent_params`; the framework stashes it on the session (the same way
  `cascade_driver_id` is threaded) and **echoes** it back on the event. The
  framework owns no retry-count state and makes no retry-count decision — this
  is the loop-guard counter, and the cap lives kb-side.

## Framework changes (touch points)

1. **`jaato-sdk/jaato_sdk/events.py`** — add `AgentErrorEvent` (mirror
   `SessionTerminatedEvent` ~L406-452 + the fields above).
2. **`shared/plugins/subagent/ui_hooks.py`** — add `on_agent_error` to the
   `AgentUIHooks` protocol.
3. **`server/core.py` `ServerAgentHooks`** (~L2806-3055) — implement
   `on_agent_error` to `server.emit(AgentErrorEvent(...))`.
4. **Emit at the three exhaustion sites, BEFORE teardown:**
   - `core.py:4426` (model-thread terminal) — call `hooks.on_agent_error(...)`
     immediately before the existing `SessionTerminatedEvent` emit.
   - `core.py:4619` (`NudgeExhausted`).
   - `runner_spawn.py:770` (bootstrap failure).
   Ordering at each site: **`AgentErrorEvent` first** (recovery point), then the
   existing `SessionTerminatedEvent(reason=error)` (teardown signal). Teardown
   still proceeds — the framework does not block.
5. **`request_id` extraction** — a small provider-aware helper at the emit site
   (`getattr(exc, "request_id", None)` covers OpenAI; extend per provider).
6. **`attempt` plumbing** — accept `attempt` via spawn `agent_params`, stash on
   the session, echo onto the event. (Mirror the `cascade_driver_id` threading
   in `session_manager.create_session`.)

`SessionTerminatedEvent(reason="error")` is still emitted → full back-compat:
a cascade with no `on_agent_error` policy behaves exactly as today (abort).

## KB consumer contract (`cascade_on_error.py`)

Subscribes to `AgentErrorEvent` via the reactor surface and applies **policy**
(the framework holds none):

```
on AgentErrorEvent(stage, error_type, classification, attempt, …):
    if attempt >= CAP:                      -> abort        # loop guard (kb cap)
    elif classification == "fatal_contract" -> abort        # no point recovering
    elif error_type == "NudgeExhausted"     -> retry once, then escalate
    elif classification == "transient_provider" and attempt < N -> retry (backoff)
    else                                    -> escalate / reroute
```

- **retry / reroute** = re-spawn the stage with the same (retry) or adjusted
  (reroute: different model/provider) inputs, passing `attempt` (as a **string**
  — the envelope's `agent_params` is `Dict[str, str]`) in `agent_params`.
- **abort** = today's behavior (write `cascade_aborted.json`).
- Caps and which-errors-to-retry live **here**, per-cascade — never in the
  framework (no-hardcoded-fallback).

**`attempt` echo is at the `create_session` boundary only.** `agent_params`
rides the `SessionInitEnvelope` → `JaatoSession._agent_params` → read by the
error-emit to echo. `create_session` builds that envelope for **both**
cold-spawn and the warm-slot `session.bootstrap` dispatch, so the warm /
`slot.settled` handoff does **not** bypass it. The framework does **not** reach
above `create_session` into a kb-side `persist_pending_spawn → slot.settled`
carrier — the kb must thread `attempt` through its own carrier to the *final*
`create_session(agent_params={"attempt": ...})` call. Framework guarantee: once
it lands at `create_session`, it is stashed and echoed.

**Dedupe (open-q #3) resolves via the existing `is_handled` guard** — no new
infra. The kb's `cascade_on_error.py` (on `AgentErrorEvent`, fires first) calls
`mark_handled(session_id)` on its decision; the existing
`cascade_on_session_error.py` (on `SessionTerminatedEvent(reason=error)`, fires
second) already does `is_handled() → skip`, so it no-ops on recovery and remains
the back-compat abort path when no `AgentErrorEvent` fired. This is why the emit
order (AgentErrorEvent first) matters.

## Non-goals

- **In-place, state-preserving retry of the same session** (suspend the model
  thread and await a blocking directive over IPC). Heavy lifecycle surgery
  against an out-of-process reactor; stage-level re-spawn delivers recovery
  without it. Revisit only if preserving mid-stage session state across the
  error becomes a hard requirement.
- **Transient-classification-as-gate** / framework-side retry policy. The
  framework auto-handles Layer 1 and surfaces the rest; it does not decide
  reactor-level retriability.
- **Provider error-mapping fixes** (e.g. a bare `openai.APIError` 5xx not being
  mapped to a transient type so Layer 1 never retries it). That is a separate
  provider-classification bug; improving it only changes how much Layer 1
  absorbs before Layer 2 fires. Tracked independently of this contract.

## Test plan

- `AgentErrorEvent` round-trips through the SDK event (de)serialization.
- `on_agent_error` fires at each of the three sites, before
  `SessionTerminatedEvent`, with the right `error_type` / `request_id`.
- `attempt` echoes the value passed via spawn `agent_params`.
- No `on_agent_error` handler → unchanged abort behavior (back-compat).
- (kb-side) `cascade_on_error.py` re-spawns on a synthetic `AgentErrorEvent` and
  caps at the configured attempt count.

## Open questions / risks

1. **`request_id` survival** — confirm the provider exception still carries
   `request_id` when it reaches `core.py:4426` (wrapped through `with_retry` /
   RPC). If stripped, plumbing it is the fiddliest part.
2. **Re-spawn timing** — failed-session teardown vs immediate re-spawn (warm-slot
   reuse vs cold). Validate one run.
3. ~~**Event ordering** — dedupe `AgentErrorEvent` vs `SessionTerminatedEvent`.~~
   **RESOLVED** — reuses the existing `is_handled` guard
   (`cascade_on_error.py` `mark_handled` → `cascade_on_session_error.py`
   `is_handled()→skip`); no new infra. See the KB consumer section.

## Decision log

- **Non-blocking (emit + reactor re-spawns) over suspend-and-await-directive** —
  fits the existing "reactor drives via `create_session`" pattern; avoids
  blocking the daemon on an out-of-process decision.
- **Dedicated `AgentErrorEvent` over enriching `SessionTerminatedEvent`** —
  first-class recovery contract + a clean `on_agent_error` subscription surface;
  keeps recovery-only fields (`request_id`, `attempt`, `classification`) off the
  generic lifecycle event.
- **Classification is a hint, not a gate; event fires only after Layer-1
  exhaustion** — the reactor is bothered only with what the framework could not
  resolve itself.
