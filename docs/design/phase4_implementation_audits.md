# Phase 4 implementation audits

Accumulating ledger for in-flight audit corrections discovered during
Phase 4 implementation.  Modeled on Phase 3's
`per_session_confined_runner_phase3_3c_rpc_surface.md` accumulation
pattern (24 audits grew that doc to 2351 lines).

Each entry records: WHEN the audit fired, WHAT the original plan said,
WHAT the audit found, and HOW the scope was corrected.  Per Phase 3
audit-discipline #2 ("audit-doc commits before implementation commits
at architectural decision points"), these audits land as their own
commits BEFORE the corresponding code commits, so reviewers can pre-
correct scope.

---

## Audit 1 — §4.4 description-callback regression: session plugin tier-flip

**Plan reference:** `per_session_confined_runner_phase4_plan.md` §3.4.

**Backlog reference:** `project_backlog_description_callback_gap.md`.

**Original framing** (Phase 4 plan §3.4):
> "Mechanical extension of Path F's pattern" — single commit, sub-§7b.2
> size.  `description_updated` event flows daemon-side pre-§7c but
> post-§7c the runner-side session has no install hook for the
> description callback.

**Implicit premise** (per the backlog doc):
> "the runner has its own `session_plugin` instance constructed from
> the bootstrap envelope's plugin list"

### Audit finding

The backlog doc + Phase 4 plan §3.4 both predate Path D's
`_configure_runtime_plugins` (commit `00fa6d86`), which discovers
runner-side plugins with `tier_filter="runner"`.

Today's state (verified via grep at audit time):

1. `shared/plugins/session/__init__.py:35` declares
   `PLUGIN_TIER = "daemon"`.
2. Path D's runner discover (`server/runner/session.py:_configure_runtime_plugins`)
   filters via `tier_filter="runner"` — daemon-tier plugins are
   excluded from the runner registry.
3. `grep "session_describe\|set_description\b"
   jaato-server/server/runner/` returns ZERO non-test hits.

Consequence: **`session_describe` is unreachable** from the model
post-§7c.  The model never receives the tool schema; even if it did,
the executor isn't loaded.  This is BROADER than the backlog's
"daemon-side callback wired on wrong instance" framing — the tool
itself doesn't exist runner-side.

Daemon-side state at audit time:
- `JaatoServer._setup_session_plugin` (core.py:2440) constructs a
  daemon-side session plugin instance + wires
  `set_description_callback`.
- core.py:2492 registers the plugin with `enrichment_only=True` — its
  tool schemas are NOT exposed daemon-side either.
- The callback (line 2487) IS wired but only fires when the daemon-
  side instance's `set_description` is invoked — which it never is
  (no daemon-side model loop post-§7c).
- `SessionManager._session_plugin` (session_manager.py:157) is a
  SEPARATE independent instance used for save/load/delete — unrelated
  to tool exposure; constructed directly in `__init__`, doesn't go
  through PluginRegistry discovery.  Unaffected by tier-flip.

### Corrected scope

§4.4 closes the gap in 4 sub-actions (still single commit, but the
"mechanical Path F extension" framing was too narrow):

A. **Tier-flip** `shared/plugins/session/__init__.py`:
   `PLUGIN_TIER = "daemon"` → `"runner"`.
   - Path D's runner discover now loads the plugin.
   - Daemon-side discover (no filter) still loads it too (it's a
     superset).  The daemon-side `_setup_session_plugin` continues to
     run; we'll prune the dead-callback wiring in sub-action D.
   - `test_plugin_tier_partition.py` doesn't pin individual plugins
     to specific tiers (verified: it pins partition disjointness +
     "has-tier" property, not per-plugin tier).  No test updates
     needed for the flip itself.

B. **NotificationFrame extension** (the original plan's mechanical
   change, now with a valid premise):
   - `server/runner/rpc.py` — add
     `_NOTIF_DESCRIPTION_UPDATED = "description_updated"` constant.
   - Extend `_install_session_notification_callbacks` to install a
     description-callback shim on the runner-side session plugin
     (lookup via `session._runtime.registry.get_plugin("session")`).
   - Mirror restore in `_restore_session_notification_callbacks`.

C. **Daemon-side demuxer extension**:
   - Extend `_build_send_message_notification_handler` with a
     `description_updated` branch that emits
     `SessionDescriptionUpdatedEvent`.

D. **Dead-code removal** (daemon-side callback wiring):
   - Delete the `if hasattr(session_plugin, 'set_description_callback')`
     block at core.py:2480-2488.  Post-flip, the daemon-side instance
     still exists (registry has no tier filter daemon-side) but its
     `set_description` is never invoked — model loop is runner-side.
     The callback wiring is dead-weight.
   - Keep the rest of `_setup_session_plugin` (session_id propagation
     + registry registration + auto-approved tools).  These still
     matter for daemon-side enrichment + auto-approval policy.

### Risk: two-instance disk-write contention?

The `session` plugin's tools write session state to disk (e.g.
`set_description` updates `{workspace}/.jaato/sessions/{id}.json`).
Post-flip there are TWO instances:

- Daemon-side: still loaded via no-filter discover, but
  `set_description` never invoked (no daemon-side model loop) — no
  writes.
- Runner-side: newly loaded; `set_description` invoked by model →
  writes to disk.

`SessionManager._session_plugin` (third instance) also writes —
specifically `save()`/`delete()` via daemon's session-lifecycle paths.

The two writers (runner's plugin via `set_description`,
SessionManager via `save`) target the same files.  Pre-§7c the
same shape existed (daemon's JaatoServer.session_plugin +
SessionManager._session_plugin both wrote); Path D-era introduced
the SECOND daemon-side writer (the now-defunct daemon-side
session_describe writer — though it was never reached because the
tool wasn't exposed).  Net effect: one writer pre-§7c, one writer
post-§4.4 (SessionManager.save + runner.set_description target
different fields/files — verified pre-flight).

### Decision

Scope expanded from "single mechanical commit" → "single commit with
tier-flip + dead-code removal", still one commit but with
substantive architectural correction.  Audit-doc commit (this) lands
first per discipline #2; code commit follows immediately.

### Tests

7 regression-pin tests:

1. `PLUGIN_TIER` is `"runner"` in `shared/plugins/session/__init__.py`
   (AST/import pin).
2. Runner-side discover with `tier_filter="runner"` loads the session
   plugin (integration with `_configure_runtime_plugins`).
3. `_install_session_notification_callbacks` installs the description
   shim when a runner-side session plugin is present.
4. The shim emits `description_updated` NotificationFrame.
5. Daemon-side demuxer routes `description_updated` →
   `SessionDescriptionUpdatedEvent`.
6. Restore reverts the description callback to its pre-install value.
7. AST pin: `_setup_session_plugin` no longer wires
   `set_description_callback` (catches regression of the dead-code
   removal).

---

## Audit 2 — §4.7 multi-turn UX investigation (read-only)

**Plan reference:** `per_session_confined_runner_phase4_plan.md` §3.7.

**Trigger:** Phase 3 closure recap §"Known sub-gaps deferred
post-cycle-13" item 3 — "cumulative 16 PermissionRequestedEvents
+ Error: 1bcd84... rendering" flagged for investigation.

**Original framing** (plan §3.7):
> Read-only investigation.  Determine if the multi-turn loop is
> fixed by §4.1 / §4.2 OR if there's a separate gap.
>
> Acceptance: either (a) verified to be fixed by §4.1+§4.2 (closes
> the investigation), or (b) audit doc filed for a separate fix
> (promotes to §4.8 or follow-up).

**Constraint at audit time:** §4.1 (J.A `call_id` propagation) is
shipped (commit `1063c0c8`).  §4.2 (J.B `editable_metadata`) is
NOT yet shipped.  This audit therefore answers two sub-questions:

A. Does §4.1 alone account for the 16-event + UUID-error pattern?
B. Is there a residual gap that needs §4.8?

### Sub-question 1 — "cumulative 16 PermissionRequestedEvents"

**Finding:** Single producer.

Grep of `PermissionRequestedEvent(` across the entire server
codebase (excluding tests) returns exactly ONE producer:
`server/runner_rpc_handlers/prompt_operator.py:138`.  Each call to
`PromptOperatorHandler.handle()` produces exactly one
`PermissionRequestedEvent` (plus one companion
`PermissionInputModeEvent` per Path J).

Conclusion: **"16 PermissionRequestedEvents" = 16 distinct ASK
round-trips**.  Normal multi-turn behavior — N tool invocations
that need permission = N ASKs.  **Not a regression.**

The cycle-13 verdict's framing ("16 prompts may suggest a UX
degradation") was based on the assumption that some emit might
fire multiple times per ASK.  Verification rules that out.

### Sub-question 2 — "Error: 1bcd84... rendering"

**Finding:** Race condition in `respond_to_permission`'s
fall-through path.  Separate bug shape; orthogonal to J.A / J.B.

Trace:

1. TUI renders `ErrorEvent` via `rich_client.py:1795`:
   ```python
   display.add_system_message(
       f"Error: {event.error_type}: {event.error}",
       style="system_error_bold",
   )
   ```
2. The format `Error: <type>: <msg>` produces text like
   `Error: StateError: Unknown permission request: 1bcd84...`.
   The cycle-13 verdict abbreviated to "Error: 1bcd84...".
3. The producer is `core.py:4099-4102` in
   `JaatoServer.respond_to_permission`:
   ```python
   # Neither path resolved — unknown request.
   self.emit(ErrorEvent(
       error=f"Unknown permission request: {request_id}",
       error_type="StateError",
   ))
   ```
4. This fires when `respond_to_permission(request_id, response)` is
   called but neither resolution path matches:
   - Path 1 (Step 7.1): `PromptOperatorHandler.resolve_response`
     returns False (request_id not in `_pending`).
   - Path 2 (legacy): `_pending_permission_request_id` doesn't
     match.

**Race shape:** the client's `respond_to_permission` arrives AFTER
the request has been resolved/popped from `_pending`.  Possible
triggers:

- ASK resolved via timeout daemon-side; client's late 'y' arrives.
- ASK resolved by an auto-allow rule on a sibling code path; client's
  'y' arrives anyway.
- Multiple clients attached to the same session; one resolves; the
  others' resolves are stale.
- TUI race between the user typing 'y' and a daemon-side
  notification clearing the pending state.

**Architectural implication:** the fall-through error treats every
unmatched response as a hard error.  This is a strict policy that
fires on transient races where ignoring the stale response would
be correct.  Pre-§7c the same logic existed but the race window
was different (in-process call → no async-resolve race).  Post-§7c
the RPC-mediated path introduces new timing where the resolve and
the response can race more easily.

### Verdict

**§4.7 close status: PARTIAL.**

A. **"16 PermissionRequestedEvents"** — empirically explained;
   not a regression.  Section 6.1 of the plan's risk register can
   be downgraded for this specific symptom.

B. **"Error: 1bcd84..."** — separate bug shape (stale-response
   race in `respond_to_permission` fall-through).  NOT closed by
   §4.1 (J.A is TUI-side per-tool-block correlation; doesn't touch
   the daemon's resolve-state machine).  NOT closed by §4.2 (J.B
   is edit-and-approve schema lookup; orthogonal).  **Candidate
   §4.8 gap** if the race reproduces in post-§4.1+§4.2 integration
   testing.

### Recommendation

1. **Do NOT promote to §4.8 immediately.**  Two reasons:
   - §4.2 isn't shipped yet; can't verify in a clean post-§4.1+§4.2
     state.
   - The race may be transient (timing-specific to cycle-13's
     execution); needs reproduction before allocating fix-cycles.

2. **Add a §4.7 follow-up to the cycle-14 acceptance gate** (Phase
   4 plan §4 acceptance gate item 1, multi-turn test): observe
   whether the stale-response race reproduces post-§4.1+§4.2.  If
   yes → file §4.8 (race fix: idempotent `respond_to_permission`
   that drops stale responses without emitting ErrorEvent).  If no
   → close the investigation entirely.

3. **Tag the producer for future reference.** core.py:4099's
   fall-through is a hard-fail policy.  Phase 4+ guidance for that
   site: when implementing §4.8, the fix is likely demote-to-debug-
   log (stale responses are not errors; they're normal in
   async-resolve flows).  Preserve the error path for genuinely
   unknown requests (e.g., request_id never registered) — distinguish
   "stale" (was registered, now resolved) from "unknown" (never
   registered) via a small two-bucket state in the handler.

### What this audit does NOT decide

- The §4.8 fix itself.  Defer to cycle-14 verdict.
- Whether `respond_to_permission`'s fall-through error policy
  should be loosened for OTHER non-permission paths (clarification
  at core.py:4112/4131, reference selection at 4148).  Same shape;
  same potential race; same future-§4.8-style fix when those
  surface.

### Tests

None — this is a read-only audit per plan §3.7.  Tests will land
with the eventual §4.8 fix (if §4.7 follow-up at cycle-14 promotes
to §4.8).
