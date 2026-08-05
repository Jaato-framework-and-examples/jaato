# Example: reliability reactor (Phase 1)

A **reference** tenant-authored reactor — the event-driven successor to the
in-process `reliability` plugin. See
[`../../reliability-event-driven-migration.md`](../../reliability-event-driven-migration.md)
for the full design and the §8 decisions this implements.

> **This is reference code, not a wired plugin.** It targets premium reactor
> infrastructure (`jaato_premium.reactors`) and reuses the public
> `shared.plugins.reliability` types, so it is **not executed or registered in
> this repo** (no premium reactor engine in jaato-server; CI does not import
> it). Copy/adapt it into a tenant package — or a premium example package —
> and register it via a `jaato.premium_reactors` entry point.

## What it does (Phase 1)

| Subscribes (bus events) | → | Does |
|---|---|---|
| `tool.call_started` | | feed the `PatternDetector` (pre-execution patterns) |
| `tool.call_completed` | | update the per-`FailureKey` trust ledger using `success` **and** the framework-computed `is_error_result` (PR #319); on a `TRUSTED→ESCALATED` transition → emit `reliability.escalated` + nudge |
| `agent.output` | | feed the detector's model-text stream (announce-no-action, etc.) |
| `turn.completed` | | end-of-turn pattern eval + advance the turn window |
| `plan.step_updated` | | (extension point — prerequisite-policy context) |

**Steer** is a single non-blocking `ctx.inject_prompt` nudge (both interactive
and headless sessions — safe by construction, the §7c invariant). **Observe**
is `ctx.emit_event("reliability.escalated" | "reliability.pattern_detected", …)`
(the event types added in PR #318).

## What it reuses (logic stays public)

- `shared.plugins.reliability.types`: `FailureKey`, `EscalationRule`,
  `TrustState`, `BehavioralPattern`.
- `shared.plugins.reliability.patterns.PatternDetector` — instantiated
  per-session; the migration moves the *wiring* to a reactor, not the logic.
- SDK substrate: the `reliability.*` event types (#318) and the
  `is_error_result` field on `tool.call_completed` (#319).

## What it deliberately leaves out

- **Enforcement** (forced-approval). Phase 1 is detect + nudge only. Phase 2 is
  **presentation-aware**: interactive = synchronous forced-permission-prompt;
  headless = async (deny+nudge → `post_webhook` notify → **T3** async-approval
  `gate` suspend/resume). See migration doc §5–§6, §9.
- The plugin's `RECOVERING` window, per-model profiles, and the full
  prerequisite-policy engine — the trust ledger here is a compact
  consecutive-failure counter. A production reactor would port more of
  `ToolReliabilityState`.
- **`FailureKey` params on completion**: `tool.call_completed` carries
  `tool_name`+`success`+`is_error_result` but **not** `tool_args`. This example
  keys escalation on the tool name alone (`args={}`); a production reactor
  correlates by `call_id` with the `tool.call_started` it saw to rebuild the
  parameter signature.

## Files (mirror `jaato_premium/drift_monitor/`)

| File | Role |
|---|---|
| `reactor_logic.py` | `handle_event(params, event, ctx)` + import-cached `_session_states` |
| `state.py` | `ReliabilityReactorState` — per-session trust ledger + `PatternDetector` |
| `registration.py` | `get_reactor_definition()` → `PremiumReactor` (rules + action shim) |
| `__init__.py` | re-exports |

## Adopt

1. Copy this package into your tenant/premium package (e.g.
   `mypkg/reliability/`).
2. Fix the two `<your_pkg>` placeholders in `registration.py` (the shim import).
3. Register the entry point:
   ```toml
   [project.entry-points."jaato.premium_reactors"]
   reliability = "mypkg.reliability.registration:get_reactor_definition"
   ```
4. Tune `EscalationRule` (default `count_threshold=3`) and the nudge cooldown to
   taste; the installer writes the rule + shim to `~/.jaato/` at daemon start.
