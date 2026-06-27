# Org-wide assessment: simplifying SDK clients with the new convenience facade

**Date:** 2026-06-27 · **Facade:** jaato-sdk `IPCClient.session` / `Session.ask|complete|stream` / `IPCRecoveryClient.session` / module `ask` (PRs #400 + #402, merged to `jaato` main). **Scope:** all 15 repos under github.com/Jaato-framework-and-examples.

## TL;DR

- **5 repos have real wins** (SIMPLIFY/PARTIAL), ~**300+ LOC** of event-loop boilerplate removable.
- **2 migrations are BLOCKED on facade gaps** (config_root/apparmor passthrough) — fix the facade first.
- **6 repos are correctly KEEP / not-applicable** — do not send agents to "simplify" these (wasted effort): telegram (WebSocket, not IPC), premium (no consumer clients), tui-driven-tests walker, LoRA-training (no SDK), symphony (no SDK), and the TUI.
- **The single highest-leverage change is the scaffold generator + docs** in `jaato` itself — it teaches the verbose pattern to every new user.

---

## Facade gaps surfaced (fix these FIRST — they unblock Tier 2 and affect everyone)

> **G0 (host tools) — RESOLVED 2026-06-27 (#403).** `IPCClient.session(..., client_tools=[...])` now registers host tools after connect / before create_session, so host-tool clients use the facade. (Surfaced during the doc rewrite; Daniel's suggestion.) The remaining gaps below are still open.

| # | Gap | Who it blocks | Fix |
|---|---|---|---|
| G1 | `session()`/`open_session` does **not** forward `config_root=` or `apparmor=` to the `IPCClient` ctor (only client_type/auto_start/env_file/workspace_path/on_status_change) | handoff_test, kb-enablement `sdk_harness` — both rely on `config_root` (decoupled config) + `apparmor=True` (confinement). **Cannot migrate without this.** | Add `config_root`/`apparmor` passthrough (conditional, like `on_status_change`). Small. |
| G2 | No `send_message(parallel_tools=False)` equivalent on `ask`/`complete` | reliability-exercise drivers (used everywhere there) | Add a `parallel_tools` kwarg to `ask`/`complete`/`stream`, or document the omission. |
| G3 | No **connect-only / one-client-many-sessions** entry; `session()` couples connect+create_session+disconnect per block | enphase (connect-once, N sessions/interval), kb-enablement `connect_client`, tui walker | Optional: a `Client.connected(...)` ctx-mgr that yields a connected client whose `.session()` per-turn reuses it. Defer unless demand. |
| G4 | Facade is IPC-only; no WebSocket convenience layer | jaato-client-telegram (WSTransport) | Separate effort (WS facade) — out of scope for this push. |
| G5 | Confirm a **denied/escalated tool** turn (turn still completes with an in-turn tool error) does **not** raise `AgentError` | reliability-exercise, any permission-gated client | Verify/spec: `ask` raises only on `SESSION_TERMINATED(reason="error")`, not on tool-level failures. (Believed correct — confirm.) |

**Cross-cutting migration caveats** (tell every fixer agent): `ask()` defaults to **model-only** output → pass `sources=None` to preserve clients that collected all output; errors now **RAISE** (`AgentError`) → wrap in `try/except` to keep per-item resilience loops; keep `client_type=ClientType.API` and a real `env_file`; requires a jaato-sdk version that includes the facade (bump the dep).

---

## Tier 1 — clean wins (do now; no facade change needed)

### enphase-energy-monitoring — `src/jaato_advisor.py`
- `JaatoEnergyAdvisor.run_specialist` (~327-361): textbook Event + 3×`subscribe` + `send_message` + `wait_for(done)` + collect + `end_session`. **SIMPLIFY** the inner body → `async with IPCClient.session(profile=agent, agent=agent, client_type=API, env_file=…, workspace_path=…) as s: text = await s.ask(prompt, sources=None)`. **M, ~30-40 LOC.**
- Caveats: use `sources=None` (current code collects all output); wrap in `try/except AgentError` to keep per-specialist fault isolation (it currently captures ErrorEvent and continues); this shifts connect-once → per-session connect (warm reconnects cheap; flag if interval cadence is tight — or wait for G3).

### reliability-exercise — 6 driver files
- `run_t2.py` (43-74): pure send-and-wait, no event inspection. **SIMPLIFY**, S, ~30 LOC → `session()` + loop of `ask()`.
- `run_cascade.py` (46-100): multi-turn + a "should-never-fire" PERMISSION_REQUESTED logger. **SIMPLIFY** (lean PARTIAL), M, ~45 LOC → `session(on_permission=log_cb)` + `ask()` loop; keep the post-turn `sleep(1.0)`.
- `run_hibernate.py` (35-104): **PARTIAL** — setup loop → `ask()`, but keep the TOOL_CALL_END observer window (needs `s` to expose `subscribe`, else stay low-level).
- `diag2.py`, `run_t3.py`, `diag.py`: **KEEP** — entire purpose is inspecting `TOOL_CALL_END` fields / `subscribe_all` event discovery / deny→allow flip detection; `ask()` discards all of it.
- Caveats: all use `send_message(parallel_tools=False)` (G2); confirm denied/escalated turns don't raise (G5).

### jaato-cascade-based-prototype (`jaato-based-kb-enablement-2.0`)
Already has a hand-rolled mini-facade (`orchestrator/sdk_harness.py`) — the facade overlaps it almost exactly.
- **3 raw pytest files** that bypass the harness and inline the full dance **incl. a `drain_events()`/`client._events_active` race hack** the facade obviates: `tests/test_discovery_determinism.py` (111-197), `tests/test_context_determinism.py` (65-137), `tests/test_cross_stage_cascade.py` (89-145, already `@skip` → consider deleting). **SIMPLIFY**, M, ~110 LOC total → per-run `session()` + `complete()`.
- `sdk_harness.py`: `run_session_on_client` (166-264) **PARTIAL** → reimplement body on `s.complete()` (keep the function as the seam; preserve `disabled_tools`, the early `on_agent_completed` callback, and error_type/summary in the raised error); `run_one_session` (407-439) **SIMPLIFY**. Every delegated consumer (smoke_context, test_context_5x, inspect_session, standalone_memory_curator, cascade_develop's entry spawn) improves for free. **BLOCKED by G1** (harness passes `config_root`/`apparmor`).
- `connect_client`, `watch_cascade_events` (cascade_events observer), `cascade_develop.py` connection/`cascade.cancel`: **KEEP** — low-level by design.

### jaato-via-hypothesis-integration — `annotation_agent.py`
- `JaatoAgent` (93-169): **PARTIAL** — lifecycle (95-104) → `IPCRecoveryClient.session(agent="annotation-agent", …)`; plain-turn text (send + AgentOutput accumulate + break on TurnCompleted) → `await s.ask(prompt)`. **Keep** the mid-turn `PermissionInputModeEvent`/`ClarificationInputModeEvent` loop (posts partial output, awaits out-of-band human reply, then `respond_to_permission`/`respond_to_clarification`) — no facade equivalent. **S, ~15-20 LOC.**
- Caveat: preserve session-reused-across-annotations (don't wrap each annotation in its own `session()`).

---

## Tier 2 — blocked on facade gap G1 (do after fixing the facade)

### handoff_test — `src/sdk_harness.py` + `src/orchestrator.py`
- Server-**reactor** cascade with a pure **observer** client (no `cascade_driver_id`, no client-side stepping). The bulk (~300 LOC of `STPEventLoop`/`EscalationEventLoop`, `attach_session` on GateAnnounced, token/timeline/payload harvesting) is **KEEP** — `ask`/`complete`/`stream` can't surface it; `complete()` returns the *first* AGENT_COMPLETED, but this cascade needs the **auditor's** terminal one.
- **Realistic win: ~30-40 LOC** of lifecycle bookends (`connect_client`/`teardown_client` + the two flow wrappers) → `IPCClient.session(...)` — **but BLOCKED by G1** (config_root + apparmor are load-bearing here) and must keep explicit post-create `permissions`/`disable_tool` calls (no facade equivalent).

---

## Tier 3 — `jaato` framework: scaffold + docs (highest leverage; teaches everyone)

- **`jaato-server/shared/scaffold/_client_templates.py`** — the generator behind `jaato-scaffold new`. `CLIENT_TEMPLATE` **SIMPLIFY** → `session()`+`ask()` (its docstring documents the very hang the facade kills). `FIRE`/`CASCADE`/`HOST_TOOLS` **PARTIAL**; `OBSERVER` **KEEP**. **L** (update golden tests `test_client_template_completion_wait.py` in lockstep). Note: keep a commented low-level variant since scaffold output is teaching material. *(Whether to convert scaffolds to the facade is itself a decision — they're deliberately explicit.)*
- **Docs to make facade-first:** `docs/jaato-ipc-ws-transport-clients.md`, `docs/client-sdk-reference.md` (add `.session/.ask/.complete/.stream` rows), `CLAUDE.md` "Tool Execution Flow". **S each.**
- **Provider smoke harnesses** (`shared/plugins/model_provider/<P>/smoke/smoke_chat.py` ×~12): **PARTIAL/optional** — `smoke_chat`→`ask`, `smoke_signal_completion`→`complete`, but they're deliberate low-level conformance probes (explicit ErrorEvent assertions); converting changes the test surface. **Optional.**

---

## KEEP / not-applicable (do NOT dispatch fix agents here)

| Repo | Why |
|---|---|
| **jaato-client-telegram** | WebSocket client (`WSTransport` + `jaato_sdk.events` wire types only) — **zero `IPCClient`**; facade is IPC-only (G4). Long-lived per-chat session pool + reconnect + host-tool dispatch + binary staging + rich multi-EventType rendering — all genuinely low-level. 0 LOC. |
| **jaato-premium** | **Zero IPCClient consumers.** All `create_session`/`subscribe`/`send_message` are server-side (reactors/gates/gossip/runtime). The "send+wait" code (`gc_benchmark`, `modlog_training_pipeline`, `cli_mcp_harness`) is the **wrong client** — direct Vertex `shared.jaato_client.JaatoClient`, not the daemon SDK. |
| **jaato-tui-driven-tests** | `harness/walker.py` — 1 long-lived `IPCRecoveryClient` → N sessions, rebuild-on-CLOSED recovery, completion on `AgentCompletedEvent(agent_id=="documenter")` + retry/error routing. Justified advanced low-level consumer. |
| **kb-stage-agent-LoRA-training** | No `jaato_sdk` import — drives a vLLM `/chat/completions` endpoint via raw `urllib` (STaR loop). N/A by design. |
| **jaato-based-symphony-spec-implementation** | No SDK client code found. |
| **jaato (TUI)** `jaato-tui/rich_client.py` | Interactive terminal client — needs streaming/permission/plan/tool event API. KEEP by design. |

---

## Recommended sequencing

1. **Fix facade gaps G1 (config_root/apparmor passthrough) + G2 (parallel_tools) + confirm G5** — small, unblocks Tier 2 and removes migration caveats. *(I can do this.)*
2. **Tier 1** — dispatch per-repo fix agents: enphase, reliability (run_t2/run_cascade/run_hibernate), kb-enablement (3 pytest files + sdk_harness body), hypothesis. ~300 LOC, mostly S/M.
3. **Tier 3 docs** — facade-first edits in `jaato` so new users start with the facade.
4. **Tier 2** — handoff_test + kb-enablement harness lifecycle (after G1).
5. Skip the KEEP/N-A repos.

Each Tier-1/2 entry above is self-contained (file:line + verdict + replacement + caveats) and can be handed to a fix agent as-is.
