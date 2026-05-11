# Backlog: clarification + references plugins lack RunnerRPCChannel equivalents

> **Status**: Pre-existing gap surfaced by §7c Step 7 disposition audit
> (commit `285449d0`) + confirmed by Step 7.4 investigation
> (`a292999`-era).  Not in scope for the §7c+§7d Phase 3 critical path.
> Mirrors the permission plugin's [`runner_rpc_channel.py`](../jaato-server/shared/plugins/permission/runner_rpc_channel.py)
> pattern which ships today.

## Problem

Both `shared/plugins/clarification/` and `shared/plugins/references/`
declare `PLUGIN_TIER = "runner"` and run runner-side post-§7c
seat-flip.  Their `_get_channel()` methods return the in-process
`self._channel` directly — no detection of runner-side execution
context, no relay through the daemon's RPC.

The permission plugin already ships the right pattern
([`runner_rpc_channel.py`](../jaato-server/shared/plugins/permission/runner_rpc_channel.py)):
when the plugin runs runner-side (`registry.runner_rpc_client` is
set, which §7c Step 7.2 wires at commit `cb656034`), the plugin
routes ASKs through `client.prompt_operator` — a daemon-side RPC
handler that emits a `PermissionRequestedEvent` and awaits the
matching response.

Without equivalent channels for clarification + references:

- Runner-fired `request_clarification` ASKs land in a runner-side
  in-process queue that no daemon-side listener reads.  The
  daemon's `respond_to_clarification` pushes to a daemon-side
  queue that no runner-side listener reads.  ASKs orphan.
- Runner-fired `selectReferences` ASKs same shape.

The daemon's `respond_to_clarification` / `respond_to_clarification_batch`
/ `respond_to_reference_selection` methods at `core.py:3842-3905`
still push to `_channel_input_queue` — that worked pre-seat-flip
when the daemon-side plugin instance was the live one.  Post-§7c
the runner-side instance is live; the queue push is orphaned for
those flows.

## Audit context

Surfaced during the §7c Step 7 disposition audit + Step 7.4
investigation:

- `permission/plugin.py:135` `_get_runner_rpc_channel()` exists and
  is wired via the §7c Step 7.2 commit.
- `clarification/plugin.py:56` `_get_channel()` returns the in-process
  channel directly — no runner-side detection.
- `references/plugin.py` same pattern.

This is a **latent regression**: the existing test suite passes
because no automated test exercises a runner-fired clarification
/ references ASK end-to-end (the existing tests stub the
channels or operate on daemon-side plugin instances).  Real-user
behavior would be: model invokes `request_clarification` → ASK
fires runner-side → queue write the daemon never reads → tool
call hangs until timeout.

## What still works vs what's broken

| Surface | Status |
|---|---|
| Daemon-side plugin instance + channel (pre-seat-flip path) | Works (unused post-§7c) |
| Daemon-side `respond_to_clarification` queue push | Reaches the orphaned daemon-side channel; runner-side instance doesn't see it |
| Permission plugin runner-side ASK (post-Steps 7.1+7.2+7.3) | ✅ Works end-to-end |
| Clarification runner-side ASK | ❌ orphans |
| References (`selectReferences`) runner-side ASK | ❌ orphans |
| Existing automated tests | ✅ pass (no test exercises the regression path) |

## Likely fix shape

Mirror the permission plugin's `runner_rpc_channel.py` pattern for
each of clarification + references:

### Per-plugin files (new)

1. `shared/plugins/clarification/types.py` — `PromptClarificationPayload` +
   `PromptClarificationResponse` dataclasses (analogues of
   `permission/types.py:PromptPayload` / `PromptResponse`).
2. `shared/plugins/clarification/runner_rpc_channel.py` — implements
   `ClarificationChannel` protocol; routes through
   `rpc_client.prompt_clarification(payload)`.
3. `shared/plugins/clarification/plugin.py` `_get_channel()` extended
   to detect `registry.runner_rpc_client` (mirror of permission's
   `_get_runner_rpc_channel`).
4. `server/runner_rpc_handlers/clarification.py` — daemon-side handler
   class (mirror of `prompt_operator.py`); emits
   `ClarificationRequestedEvent`; awaits matching response via
   `resolve_response(request_id, response)`.

Same set of 4 files for `references/`.

### Wiring (daemon-side)

In `JaatoServer.set_runner_rpc` (the §7c Step 7.1 site), after
registering `PromptOperatorHandler`, also instantiate +
register:

- `PromptClarificationHandler(emit_event=self.emit)` →
  `register_handler("client.prompt_clarification", ...)`
- `PromptReferencesHandler(emit_event=self.emit)` →
  `register_handler("client.prompt_references", ...)`

Stash both on `self._prompt_clarification_handler` and
`self._prompt_references_handler` so `respond_to_clarification`
and `respond_to_reference_selection` can dual-path-route through
them (Path 1 / Path 2 pattern from Step 7.3).

### Wiring (runner-side wrapper)

Extend `server/runner/rpc_client.py:RunnerRPCClient` with
`prompt_clarification()` and `prompt_references()` methods — thin
wrappers over `RunnerRPC.outgoing_call("client.prompt_clarification", ...)`
etc.  Same shape as the existing `prompt_operator()` method
already in the file.

### Tests

Per-plugin:

- Unit tests for the new RunnerRPCChannel (mirror of
  `permission/tests/test_plugin_runner_rpc_wiring.py`)
- E2E pin for the full daemon-half round-trip (mirror of
  `test_respond_to_permission_routing_step7_3.py`'s E2E test)

Plus AST-level pins that `set_runner_rpc` registers both new
handlers and that `respond_to_clarification` /
`respond_to_reference_selection` route dual-path.

## Effort estimate

Roughly **2 sub-commits × ~12 tests + ~300 LOC code** per plugin,
following the Step 7.1-7.3 pattern exactly.  Total ~4-6 hours of
implementation for both clarification + references, plus the
daemon-side ``respond_to_*`` rerouting.

## Out of scope for this backlog entry

- Generalizing the permission/clarification/references pattern into
  a shared `ASKPlugin` abstract base — premature; only 3 instances
  exist, and the SDK contract is intentionally per-plugin.
- Migrating the daemon-side `_channel_input_queue` away entirely —
  the queue still serves daemon-fired ASKs (pre-init permission
  prompts, certain auth flows).  Out of §7-series scope.

## Files to touch (when scheduled)

- `shared/plugins/clarification/types.py` (new)
- `shared/plugins/clarification/runner_rpc_channel.py` (new)
- `shared/plugins/clarification/plugin.py` (extend `_get_channel`)
- `shared/plugins/references/types.py` (new)
- `shared/plugins/references/runner_rpc_channel.py` (new)
- `shared/plugins/references/plugin.py` (extend `_get_channel`)
- `server/runner_rpc_handlers/clarification.py` (new)
- `server/runner_rpc_handlers/references.py` (new)
- `server/runner/rpc_client.py` (add `prompt_clarification` + `prompt_references` methods)
- `server/core.py` (extend `set_runner_rpc` registration + rewire
  `respond_to_clarification` and `respond_to_reference_selection`)
- Test files: 4-6 new test files mirroring Steps 7.1-7.3
