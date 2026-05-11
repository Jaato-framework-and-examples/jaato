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

---

## Audit 3 — §4.2 J.B `editable_metadata` schema lookup

**Plan reference:** `per_session_confined_runner_phase4_plan.md`
§3.2 + §5.1.

**Backlog reference:** Sub-gap J.B in
`project_backlog_path_j_sub_gaps_call_id_editable_metadata.md`.

**Plan's lean** (§5.1): Option A — `PromptPayload` carries
`editable_metadata: Optional[dict]` populated at the runner-side
ASK site.  Rationale: "matches the Phase 3 pattern of 'carry state
in envelope, not via daemon-side references'".

### Audit finding

The plan's Option A description implied the runner-side ASK site
would call `permission_plugin._get_tool_schema(tool_name)` to look
up the editable schema.  Verification shows **the lookup is
already done** by the time the channel sees the request:

1. `shared/plugins/permission/plugin.py:1411-1412`:
   ```python
   tool_schema = self._get_tool_schema(tool_name)
   editable = tool_schema.editable if tool_schema else None
   ```

2. `shared/plugins/permission/plugin.py:1461`:
   ```python
   request = PermissionRequest.create(
       ...,
       editable=editable,
   )
   ```

3. `shared/plugins/permission/channels.py:198`:
   ```python
   class PermissionRequest:
       ...
       editable: Optional[Any] = None  # EditableContent from types.py
   ```

So **`request.editable` is already populated runner-side** with the
`EditableContent` instance (or None).  The runner-RPC channel just
needs to read it and convert to dict.

### Decision: Option A — narrower than the plan suggested

The plan said "runner has the permission plugin instance already"
implying a schema lookup at the channel level.  In fact the lookup
ran earlier (in `check_permission`'s flow); the channel reads the
already-resolved `EditableContent` from the request.  This is
**cleaner than the plan version**:

- No new permission_plugin access inside the channel.
- No new schema-lookup at the ASK relay.
- Just a single field-read in the channel + dict conversion.

### Field-mapping (pre-§7c daemon-side hook reference)

`core.py:3062-3072` (the dead-letter daemon-side hook, untouched by
this audit) shows the canonical wire shape for
`PermissionInputModeEvent.editable_metadata`:

```python
editable_metadata = {
    "parameters": schema.editable.parameters,
    "format": schema.editable.format,
}
```

Two fields: `parameters` (list of editable param names) + `format`
(yaml/json/text/markdown).  `EditableContent.template` is NOT in
the pre-§7c wire shape — the template is an editor-rendering
concern that the runner consumes locally; the daemon/TUI don't
need it for input-mode signaling.

§4.2 produces the same shape for backward compat with TUI
consumers expecting the pre-§7c contract.

### Corrected scope (4 hops, parallel to §4.1)

A. **`shared/plugins/permission/runner_rpc_channel.py`** — read
   `request.editable` (Optional[EditableContent]); convert to dict
   shape `{"parameters": list, "format": str}`; thread to
   `PromptPayload`.  Defensive: when `request.editable` is None,
   `editable_metadata=None`.

B. **`shared/plugins/permission/types.py`** — add
   `editable_metadata: Optional[Dict[str, Any]] = None` field to
   `PromptPayload` + to_dict/from_dict.  Backward-compat: legacy
   wire dicts (without the field) deserialize to None.

C. **`server/runner_rpc_handlers/prompt_operator.py`** — thread
   `payload.editable_metadata` through to
   `PermissionInputModeEvent.editable_metadata` (currently
   hardcoded `None` per Path J.B backlog note).

D. **No daemon-side dead-code removal in scope** — the daemon-side
   hook at `core.py:3062-3072` was already a no-op post-§7c (the
   surrounding callback isn't wired to a live code path post-Path-J).
   The block is dead code but its removal is independent of §4.2;
   defer to a future "Phase 4 dead-code sweep" if needed.

### Tests (6 pins, parallel to §4.1)

1. `PromptPayload.editable_metadata` field default (None).
2. `PromptPayload.to_dict` includes `editable_metadata` key.
3. `PromptPayload` round-trip preserves populated dict.
4. `from_dict` backward-compat (legacy wire dict → None).
5. `RunnerRPCChannel.request_permission` reads
   `request.editable` and threads to
   `PromptPayload.editable_metadata` with canonical shape
   `{"parameters": list, "format": str}`.
6. `PromptOperatorHandler.handle` threads
   `payload.editable_metadata` to
   `PermissionInputModeEvent.editable_metadata` — flips the
   existing `test_input_mode_editable_metadata_is_none_path_j_b`
   pin from "is None" to "propagates".

### Rejected: Option B (handler reference)

Option B would have given `PromptOperatorHandler` a reference to
`server.permission_plugin` for a daemon-side schema lookup.
Rejected because:

- Adds daemon-side coupling that Phase 3 explicitly avoided
  (envelope-over-references pattern).
- The runner already has the schema; daemon-side lookup is
  redundant.
- `permission_plugin._get_tool_schema` requires a registry — the
  daemon's registry isn't tier-filtered (loads runner-tier plugins
  too post-Path-D), but the daemon-side permission plugin instance
  is configured with the daemon-side registry; the model loop runs
  RUNNER-side against the runner-side registry.  Wiring two
  registries to one handler invites drift.

### Coupling note (potential interaction with §4.4)

Per audit 1 (§4.4), the session plugin was flipped to runner-tier.
The session plugin doesn't declare `editable` on any of its tool
schemas today, so §4.4's tier-flip doesn't affect J.B's
correctness.  But the pattern matters: **§4.2 confirms the
"runner already has the data; envelope just propagates" pattern
that §4.4 also adopted**.  Envelope-over-references discipline
(Phase 4 plan §5.1 lean) is now empirically validated twice.

---

## Audit 4 — §4.3 §3.11 isolated-subagent opt-in (scope expansion)

**Plan reference:** `per_session_confined_runner_phase4_plan.md` §3.3.

**Backlog reference:** Plan §3.3 cites
`project_backlog_3_11_isolated_subagent_opt_in.md` as "shipped" —
**this file does not exist**.  Cross-grep confirms no production
sites for `agent_params.isolated` or `parent_runner_handle` (set
to `None` everywhere; default-share path NOT yet wired).  The
backlog reference is stale; first-pass audit context comes from
Phase 3 §7d audit Q6 (per_session_confined_runner_phase3_3c_rpc_surface.md:1778)
which explicitly punted §3.11 isolated-subagent opt-in as
"separate work, post-§7d".

**Plan's framing** (§3.3):
> Wire `agent_params.isolated: true` to spawn a fresh runner with
> a sub-AppArmor profile (`jaato-ws-{session_id}//{subagent_id}`)
> + sub-cgroup.  Default-share path already shipped in Phase 3.
>
> Files: subagent/plugin.py (branch), session_manager.py
> (`_spawn_isolated_runner`), apparmor.py (sub-profile generation).
> Audit + 1-2 implementation commits.

### Audit finding 1 — "Default-share path already shipped" is loose

Verification of the current subagent-spawn path:

- `shared/plugins/subagent/plugin.py:2674`: subagent creates its
  session via `self._runtime.create_session(...)` (the parent
  runner's runtime).  In-runtime, in-process — same Python process
  as the parent runner, same AppArmor profile, same cgroup.
- Ephemeral fan-out (`session_manager.py:5003`): constructs
  `BootstrapEnvelope(parent_runner_handle=None, ...)`.  The
  `parent_runner_handle` field exists on `BootstrapEnvelope`
  (envelope.py:281) but is never populated.

So "default-share" works today **by accident of in-process
co-location** — the parent's session_manager doesn't even use
`parent_runner_handle`.  The field was reserved for the future
shape where a subagent could optionally OPT INTO a separate
runner (and thus need an explicit parent-runner reference) but
the actual default-share isn't using it.

This is fine — co-located in-process sessions are the natural
default — but it means the plan's "default-share path already
shipped in Phase 3" is technically true via co-location, not via
the `parent_runner_handle` plumbing.

### Audit finding 2 — Subagent plugin runs RUNNER-side; can't directly call SessionManager

The plan says:
> subagent/plugin.py — branch on `agent_params.isolated`; …
> opt-in → call `SessionManager._spawn_isolated_runner(...)`

**This isn't directly callable.**  Verification:

- `shared/plugins/subagent/__init__.py:42` declares
  `PLUGIN_TIER = "runner"`.
- Path D's `_configure_runtime_plugins` loads the subagent plugin
  into the runner registry.
- The model loop invokes `spawn_subagent` runner-side; the
  `_spawn_subagent_async` thread runs runner-side; the
  `_runtime.create_session()` call (line 2674) runs runner-side.

`SessionManager` lives daemon-side.  The runner-side subagent
plugin can't reach `SessionManager._spawn_isolated_runner` via a
direct import — they're in different processes.

To spawn an isolated runner from the runner-side subagent plugin,
we need a **new runner→daemon RPC primitive**: e.g.,
`client.spawn_isolated_runner(profile, agent_id, ...)`.  The
daemon-side handler invokes `SessionManager._spawn_isolated_runner`,
spawns the new runner, returns a runner-handle (RPC channel
descriptor) to the calling runner.

This is an architectural escalation beyond the plan's "1-2
implementation commits" — it's a new RPC primitive class in the
shape of §7.1's PromptOperatorHandler.

### Audit finding 3 — Cross-runner forwarding semantics undefined

If the isolated subagent runs in a NEW runner process, the parent
runner needs to:

- Forward the subagent's prompt INTO the new runner
  (`session.send_message` over the sub-RPC channel?)
- Forward the subagent's output BACK to the parent's
  injection-queue (which the parent's model thread consumes for
  the subagent's final-summary result)
- Handle subagent termination (subagent finishes → tear down
  runner + sub-profile + sub-cgroup)
- Handle subagent failure (crash, timeout) without corrupting the
  parent's session state

None of this exists today.  Each is a non-trivial design point.

### Audit finding 4 — AppArmor sub-profile generation: only `change_profile` hat exists

Verification of `server/apparmor.py`:

- Line 184-205: `tool_hat` sub-profile exists for **in-runner**
  `change_profile` transitions (executor enters hat for tool
  execution; exits back to base post-tool).
- Line 444-489: subprocess child sub-profile work is tracked but
  not landed (per the comment at line 444:
  "Tracked fix: child subprofile (jaato-ws-S//child) for
  subprocesses").

So `aa-exec`-style spawning a sibling runner with a sub-profile
`jaato-ws-{session_id}//{subagent_id}` requires:
1. New per-subagent profile-template generation (similar to per-
   session, but with sub-profile syntax).
2. New `provision_profile` variant for sub-profiles.
3. New AppArmor profile-load handshake for the sub-profile.

The plan's "sub-profile generation (likely already supports
parent_name//{child} syntax; verify)" is empirically **not yet
supported for the spawn case**.  Only the `change_profile`
in-process transition supports `parent//{hat}`.

### Audit finding 5 — Cgroup sub-cgroup nesting: new infrastructure

Phase 3 §7d shipped per-session cgroup attach at runner spawn.
Sub-cgroup nesting per subagent requires:

1. Sub-cgroup creation under the parent session's cgroup.
2. New runner's `cgroup_attach` callable targets the sub-cgroup,
   not the session cgroup.
3. Cleanup on subagent termination (delete sub-cgroup).

The plan §3.3 doesn't mention cgroup work explicitly but §3.11's
scope (in parent design `per_session_confined_runner.md`) includes
sub-cgroup isolation.

### Decision: scope is materially larger than the plan estimated

Conservative re-estimate:

| Component | Plan estimate | Audit estimate |
|---|---|---|
| audit doc | 1 commit | 1 commit (this) |
| Runner→daemon RPC primitive | (not mentioned) | 1-2 commits + audit |
| `SessionManager._spawn_isolated_runner` helper | (1 commit) | 1 commit |
| AppArmor sub-profile generation | (1 commit, verify) | 1-2 commits + audit |
| Cgroup sub-cgroup attach | (not mentioned) | 1 commit + audit |
| Cross-runner forwarding semantics | (not mentioned) | 1-2 commits + audit |
| Subagent-plugin branch + lifecycle | (in 1 commit) | 1 commit |
| Tests (integration + unit) | (in commits) | 1-2 commits |
| **Total** | **2-3 commits** | **8-10 commits** |

This crosses the boundary from "Phase 4 carryover task" to a
**multi-cycle sub-track** comparable in scope to Phase 3's §7c
seat-flip arc.

### Option space + leans

**Option α — Full scope, multi-cycle Phase 4 sub-track.**
Implement the complete §3.11 vision: new RPC primitive, sub-
profile generation, sub-cgroup nesting, cross-runner forwarding.
Estimate: 8-10 commits across the rest of Phase 4.
- Pros: Phase 4 closure includes full isolation primitive.
- Cons: dominates Phase 4 budget; pushes §4.5 (perf baseline) and
  §4.6 (idle-shutdown) into Phase 5.

**Option β — Minimal opt-in stub, full implementation deferred.**
Land just the API surface (`agent_params.isolated` field; subagent-
plugin branch detects opt-in; opt-in path raises
`NotImplementedError` with a clear deferral message).  Tests pin
the opt-in detection + the explicit NotImplementedError.
- Pros: Closes Phase 4 §4.3 with a minimum viable API contract;
  unblocks future work without taking the full scope.
- Cons: Users get a "feature exists but doesn't work" surface.
  Risks misleading consumers if not clearly labeled experimental.

**Option γ — Defer §4.3 entirely to Phase 5.**
Document this audit as the deferral rationale.  Phase 4 closes at
§4.1/§4.2/§4.4/§4.5/§4.6/§4.7 (6 of original 7 tasks).
- Pros: Honest scope; the actual work belongs in Phase 5
  hardening since it has security implications (sub-profile
  generation, sub-cgroup nesting) that warrant hardening-cycle
  treatment.
- Cons: Phase 4 closes incomplete relative to its stated plan.

### Recommendation

**Lean: Option γ (defer to Phase 5).**

Rationale:

1. The plan's premise was wrong — the existence of
   `project_backlog_3_11_isolated_subagent_opt_in.md` was assumed
   but the file doesn't exist.  This is a "design phase not yet
   done" signal, not "ready to implement".
2. AppArmor sub-profile generation + sub-cgroup nesting are
   security-sensitive surfaces.  They belong in Phase 5
   (production hardening) per the parent design's phase split,
   not Phase 4.
3. Phase 4's natural value-density is the carryover + perf
   baseline + idle-shutdown — not a new architectural primitive.
4. Phase 3 audit-discipline #5 ("Inverse-virtue activations —
   cancel proposed work when audit reveals no consumer") may apply
   here too: no concrete consumer has asked for
   `agent_params.isolated: true` yet; it's a designed-in
   capability waiting for a use case.

Decision deferred to user; this audit captures the option space
+ lean.

### User decision (post-audit): Option α — Full scope

User selected **Option α** despite the worker's lean toward γ.
Rationale (user direction): proceed with the complete §3.11 vision
in Phase 4.  Implication accepted: §4.5 (perf baseline) and §4.6
(idle-shutdown) defer to Phase 5; Phase 4 closes with §4.1, §4.2,
§4.3 (this sub-track), §4.4, §4.7.

### Sub-commit decomposition (10 commits planned)

Mirrors Phase 3 §7c sub-step discipline.  Each sub-step gets its
own pre-implementation audit before code (per discipline #2), so
sub-step count below includes both audit + code commits.

| Sub-step | Scope | Commits |
|---|---|---|
| §4.3.0 | Audit doc (this commit) | 1 |
| §4.3.1 | `agent_params.isolated` field plumbing + opt-in detection (stub raises `NotImplementedError`) | 1 |
| §4.3.2 | Runner→daemon RPC primitive: `client.spawn_isolated_runner(...)` shape; daemon handler stub; runner-side client wrapper | audit + 1 |
| §4.3.3 | `SessionManager._spawn_isolated_runner` helper reusing `_spawn_session_runner_unconditional` machinery | 1 |
| §4.3.4 | AppArmor sub-profile generation: per-subagent profile template + provisioning hook | audit + 1 |
| §4.3.5 | Cgroup sub-cgroup nesting: creation + attach + cleanup | audit + 1 |
| §4.3.6 | Cross-runner forwarding semantics: prompt-forward INTO new runner; output-forward BACK to parent; termination + failure handling | audit + 1-2 |
| §4.3.7 | Subagent-plugin opt-in branch wires the full chain; §4.3.1's `NotImplementedError` stub replaced | 1 |
| §4.3.8 | Integration test gated on cgroup-writable host (both default-share AND opt-in flows) | 1 |
| §4.3.9 | Closure recap addendum: §3.11 status flip + Phase 4 progress update | 1 |
| **Total** | | **10 commits (or 14 with all audits)** |

Order rationale: bottom-up — primitive infrastructure first
(§4.3.2-§4.3.6), then the subagent-plugin opt-in branch wires them
together (§4.3.7), then integration test (§4.3.8) verifies the
whole chain.  §4.3.1 (stub) ships first as a tracer-bullet for the
API surface so subsequent commits have a clear seam.

### Sub-track risk register

1. **Phase 5 hardening implications.** Sub-profile generation +
   sub-cgroup nesting are security-sensitive surfaces.  Each
   sub-step's audit MUST document the hardening implications so
   Phase 5 has a clear cleanup runway.
2. **Cross-runner forwarding edge cases.** §4.3.6 has the highest
   uncertainty.  Termination semantics (parent dies mid-subagent,
   subagent dies mid-execution, both crash, network partition)
   need explicit per-case audit treatment.
3. **Test-fixture-cgroup-host coverage.** Phase 3 §7d tests gated
   on `_can_migrate_to(_find_writable_cgroup_parent())` — many CI
   hosts can't run these.  Phase 4 §4.3 inherits this gating;
   manual real-host verification is the load-bearing acceptance
   gate.
4. **No regression in default-share path.** Every §4.3 sub-step
   touching the subagent-plugin spawn path MUST preserve the
   default-share contract.  Regression-pin tests for the default
   case at each sub-step boundary.

### Sub-track acceptance gate (sub-§4.3 internal)

§4.3 closes when:

- `agent_params.isolated: true` spawns a separate runner with a
  sub-AppArmor profile + sub-cgroup, end-to-end.
- `agent_params.isolated: false` (or absent) uses the existing
  in-runtime default-share path, unchanged.
- Integration test covers both flows on a cgroup-writable host.
- Closure recap addendum updates §3.11 status from "post-§7d
  deferred" to "shipped via Phase 4 §4.3".

---

## Audit 5 — §4.3.2 runner→daemon spawn_isolated_runner RPC primitive

**Sub-step:** §4.3.2 of the §4.3 sub-track (decomposition recorded
in Audit 4).

**Original framing:** "Runner→daemon RPC primitive: `client.spawn_isolated_runner(...)`
shape; daemon handler stub; runner-side client wrapper".

**Why this needs a pre-implementation audit:** new daemon-tier
capability + new wire shape.  Settling the contract up-front means
§4.3.3-§4.3.7 can be written against a stable surface; reworking
the wire after the consumer plugins land would cascade through
every downstream sub-commit.

### Existing RPC machinery (load-bearing reference points)

The Phase 3 §3.2 runner→daemon RPC pattern is already shipped and
serves as the template for the §4.3.2 primitive:

| Component | Location | Role |
|---|---|---|
| Daemon-side handler module | `server/runner_rpc_handlers/<name>.py` | One file per RPC method; exports a `*Handler` class + a `register(rpc_server, handler)` convenience |
| Handler class | e.g., `PromptOperatorHandler` (prompt_operator.py:60) | Holds per-session state, has `async def handle(args) → dict`, registered by the convenience fn |
| Handler instantiation site | `server/core.py:set_runner_rpc()` (~L4595-4617) | Per-`JaatoServer` (per-session) at runner-spawn time; binds to `self.emit` and `rpc_client.rpc_server` |
| Runner-side wrapper | `server/runner/rpc_client.py:RunnerRPCClient` | One method per RPC; routes through `RunnerRPC.outgoing_call(method, args, timeout)`; raises `RunnerRPCError` on transport failure |
| Wrapper consumption | `registry.runner_rpc_client` (attached by registry-attribute pattern) | Runner-side plugins pull the client off the registry — same pattern as `add_reference_fragment` consumed by references plugin |

This audit follows the exact same pattern.  No new infrastructure
shape; just a new RPC method added through all four layers.

### Wire-shape contract

**Method name:** `subagent.spawn_isolated_runner`.  Dotted prefix
mirrors `client.prompt_operator` and `apparmor.add_reference_fragment`
— the prefix names the daemon-tier subsystem the runner is
calling into.  `subagent` because the eventual consumer is the
runner-side subagent plugin's opt-in branch (§4.3.7).

**Request args (dict, validated daemon-side):**

| Key | Type | Required | Purpose |
|---|---|---|---|
| `parent_session_id` | str | yes | Sanity echo; handler-bound id is authoritative |
| `subagent_id` | str | yes | Pre-generated by runner-side subagent plugin (today's `_next_agent_id` flow) |
| `profile_payload` | dict | yes | Serialized `SubagentProfile`: `model`, `provider`, `plugins`, `plugin_configs`, `system_instructions`, `preloaded_plugins`, `gc`, `max_turns`, `completion_payload_schema`, `completion_artifacts`, `suppress_base_instructions`, `env`.  Wire shape avoids passing the dataclass directly. |
| `task` | str | yes | First-turn prompt for the isolated runner |
| `workspace_path` | str | yes | Inherited from parent (§4.3 invariant: subagents always inherit workspace) |
| `agent_params` | dict | no | Forwarded `case_data` for `{{name}}` substitution / `RenderContext.agent_params`.  `isolated: true` will be stripped by the handler before passing through (it's a control flag, not template data). |
| `display_name` | str | no | Custom name (defaults to `profile_payload.name`) |
| `parent_agent_id` | str | no | For multi-hop subagent trees (subagent-of-subagent traceability) |

**Response shape:**

```jsonc
// Success
{
  "ok": true,
  "session_id": "<isolated runner's session id>",
  "subagent_id": "<echo>",
  "runner_pid": <int>,
  "apparmor_profile": "<sub-profile name>",  // empty string if not provisioned
  "cgroup_path": "<path>"                    // empty string if not provisioned
}

// Domain failure (validation reject, spawn failure, etc.)
{
  "ok": false,
  "error": "<reason>",
  "stage": "validation|sub_profile|spawn|sub_cgroup|forwarding"  // for diagnostics
}
```

Transport-level failures (handler crash, channel closed) raise
`RunnerRPCError` runner-side — same envelope pattern as
`add_reference_fragment`.

**Naming the "stage" enum:** explicit stage values let the
subagent plugin's error path produce a precise diagnostic ("the
sub-profile didn't provision" vs "the sub-cgroup didn't attach"
vs "the runner subprocess failed to start").  Aids §4.3.7 wire-up
debugging and Phase 5 hardening triage.

### Confused-deputy protection

The handler is bound at session-bootstrap time with the
**parent session's id** (same pattern as `AppArmorFragmentHandler`'s
session_id binding at apparmor_fragment.py:64-82).  The
`parent_session_id` echo in the request is sanity-checked against
the handler-bound id; mismatch raises a `ValueError` translated
to a typed envelope error.

This prevents the runner from spawning an isolated sub-runner
"under" a different parent session — important because the
sub-profile name template `jaato-ws-{parent_session}//{subagent}`
encodes the parent in the sub-profile name and a confused
parent_session_id would create a sub-profile under the wrong
parent's profile.

### Failure modes (each maps to a `stage` value)

| Stage | Trigger | Recovery |
|---|---|---|
| `validation` | Missing/malformed args, parent_session_id mismatch | Caller fixes args and retries; not retryable on identical call |
| `sub_profile` | AppArmor manager unavailable or sub-profile load failed (§4.3.4) | Caller can fall back to default-share (a fallback the §4.3.7 wire-up will offer) |
| `spawn` | Runner subprocess failed to start (resource limits, fork failure) | Caller retries after backoff; may be a transient cgroup issue |
| `sub_cgroup` | Sub-cgroup creation or migration failed (§4.3.5) | Caller can fall back to default-share or proceed without cgroup confinement (operator decision) |
| `forwarding` | Cross-runner forwarding setup failed (§4.3.6) | Spawn rolled back; caller retries fresh |

Each downstream sub-commit fills in its corresponding stage; the
§4.3.2 stub returns `validation`-success but `stage=spawn` with
"not yet implemented" for everything else.

### §4.3.2 commit scope

What lands in the code commit (one commit, audit + 1):

1. **New handler module** `server/runner_rpc_handlers/spawn_isolated_runner.py`:
   - `SpawnIsolatedRunnerHandler` class with `__init__(session_manager, parent_session_id)`,
     `async def handle(args)`, `shutdown()`.
   - Args validation (required keys present, types match,
     parent_session_id echo check).
   - Stub body: returns `{"ok": False, "error": "isolated-runner
     spawn machinery not yet implemented (Phase 4 §4.3.3-§4.3.7)",
     "stage": "spawn"}` for now.
   - `register(rpc_server, handler)` convenience.

2. **Runner-side wrapper** in `server/runner/rpc_client.py`:
   - New method `spawn_isolated_runner(...)` on `RunnerRPCClient`.
   - Builds args dict, routes through `outgoing_call`, parses
     envelope, returns the response dict.
   - Raises `RunnerRPCError` on transport failure; returns
     dict for domain failures (caller branches on `ok`).

3. **Registration site** in `server/core.py:set_runner_rpc()`:
   - Instantiate `SpawnIsolatedRunnerHandler(self._session_manager,
     parent_session_id=self.session_id)` alongside the existing
     `PromptOperatorHandler`.
   - Call `register_spawn_isolated_runner(rpc_client.rpc_server, handler)`.
   - Hold the handler ref on `self._spawn_isolated_runner_handler`
     so `shutdown()` can call its `shutdown()`.

4. **Tests** (three files mirroring Phase 3 §3.2 test patterns):
   - `server/runner_rpc_handlers/tests/test_spawn_isolated_runner.py`:
     handler unit tests (args validation, stub return, confused-
     deputy check, shutdown).
   - `server/runner/tests/test_rpc_client_spawn_isolated_runner.py`:
     wrapper unit tests (named-method routing, envelope parsing,
     error translation).
   - Optional: smoke test asserting that
     `JaatoServer.set_runner_rpc()` registers the handler on the
     rpc_server (parallel to existing prompt_operator registration
     test).

### Scope NOT in §4.3.2

To keep the commit reviewable:

- **No actual spawn logic** — the stub returns "not yet implemented"
  for everything beyond args validation.  Real spawn lives in §4.3.3.
- **No sub-profile generation** — that's §4.3.4.
- **No sub-cgroup attachment** — that's §4.3.5.
- **No cross-runner forwarding** — that's §4.3.6.
- **No subagent-plugin wire-up** — that's §4.3.7.  The §4.3.2
  commit only adds the runner-side wrapper method; nothing CALLS it
  yet (the §4.3.1 stub still fires).

### Phase 5 hardening implications (recorded for cleanup runway)

- The handler accepts `profile_payload` as a free-form dict.  A
  malicious or buggy runner-side plugin could inject unexpected
  profile fields.  Phase 5 should add a profile-payload allow-list
  (or move to a typed pydantic model on the wire) so the handler
  rejects unknown fields.  For now (Phase 4), the runner is trusted
  at this layer — same trust posture as Phase 3.
- `parent_session_id` echo is sanity-only; the handler-bound id is
  authoritative.  Aligned with `AppArmorFragmentHandler` pattern.
- `runner_pid` in the success response is informational; Phase 5
  hardening may want to remove it (process ids are a side-channel
  signal in some threat models).

### Sub-step risk register entry

Risk: the `profile_payload` serialization round-trip
(SubagentProfile → dict → JSON → dict → SubagentProfile) needs
parity with the existing in-process spawn path's resolved profile.
Mitigation: §4.3.3 (`_spawn_isolated_runner` daemon-side helper)
will reuse the same `SubagentProfile` reconstruction code path that
already exists for profile loading from disk.  §4.3.2 only needs
the wire shape; §4.3.3 will validate parity.

---

## Audit 6 — §4.3.4 sub-AppArmor profile generation

**Sub-step:** §4.3.4 of the §4.3 sub-track.

**Original framing** (Audit 4): "AppArmor sub-profile generation:
per-subagent profile template + provisioning hook | audit + 1".

**Why this needs a pre-implementation audit:** security-sensitive
surface — `subagent_id` becomes part of an AppArmor profile name +
filename + an in-kernel `change_profile` target.  Mis-sanitization
or careless cascade-teardown logic leaves orphaned profiles in the
kernel or enables profile-name injection.

### Existing AppArmor machinery (load-bearing reference points)

| Component | Location | Today's role | §4.3.4 usage |
|---|---|---|---|
| Profile file write + `apparmor_parser -r` reload | `AppArmorManager.provision_profile()` (apparmor.py:725) | Per-parent-session at session-init time | Same pattern, new method `provision_sub_profile()` |
| Profile file unlink + `apparmor_parser -R` unload | `_teardown_profile_impl()` (apparmor.py:835) | Per-parent-session at session-end | Same pattern, new method `teardown_sub_profile()` |
| `tool_hat` sub-profile body builder | `_build_tool_hat_subprofile()` (apparmor.py:1312) | Inline child profile in the parent's file, entered via `change_profile -> jaato-ws-{S}//tool_hat` | Reference for sub-profile rule-set authoring |
| Filename sanitization | `_safe_fragment_filename()` (apparmor.py:960) | Strips `[^A-Za-z0-9._-]` for reference-fragment filenames | Same regex for `subagent_id` |
| Profile name template | `get_profile_name(session_id)` → `jaato-ws-{session_id}` (apparmor.py:1206) | The kernel-visible identifier | Sub-profile NAME: `jaato-ws-{parent}//{sub}` |

### Sub-profile structure decision: standalone-with-prefix-name

AppArmor supports two flavors of `//`-named profiles:

1. **True "hat" sub-profiles** — declared INSIDE the parent's profile
   file as `profile name { ... }`.  Inherit nothing; entered via
   `change_hat` or `change_profile -> parent//hat`.  This is what
   `tool_hat` is today.
2. **Standalone profiles with `//` in their NAME** — declared in a
   separate file as `profile "jaato-ws-X//Y" { ... }`.  The kernel
   treats them as separate profiles; the `//` is purely a naming
   convention.

**§4.3.4 picks #2 (standalone-with-prefix-name).**  Rationale:

- **Avoids modifying the parent's loaded profile** at runtime.  Hat-
  based would require editing the parent's profile file + reloading,
  which affects the running parent process.
- **Cleaner lifecycle** — each sub-profile is a distinct file + a
  distinct kernel-level profile.  Teardown is symmetric to the
  parent's lifecycle.
- **Naming preserves parent design §4.3 intent** — the kernel
  identifier still reads `jaato-ws-{parent}//{subagent_id}`, so logs
  and `aa-status` output show the hierarchy.

Trade-off: the sub-profile is NOT structurally constrained to be a
subset of the parent.  An attacker who controls the daemon could
write a sub-profile with broader rules than the parent.  But the
daemon is the trust root in our threat model (the runner is the
LLM-driven untrusted process); same posture as today.

### Profile name + filename templates

| Surface | Template | Example |
|---|---|---|
| Kernel-visible profile name | `jaato-ws-{parent_session_id}//{subagent_id}` | `jaato-ws-S-A//agent-1` |
| Filename (no `/` allowed) | `jaato-ws-{parent_session_id}__sub_{sanitized_subagent_id}` | `jaato-ws-S-A__sub_agent-1` |

The filename matches `SessionManager._spawn_isolated_runner`'s
`isolated_session_id` template from §4.3.3 (`{parent}__sub_{sub}`)
— so callers can derive the filename from the isolated session id
without re-encoding the convention.

### subagent_id sanitization (CONFUSED-DEPUTY PROTECTION)

`subagent_id` originates runner-side (the subagent plugin's
`_next_agent_id` generator) but flows through the
`profile_payload` wire shape into AppArmor profile name + filename.
Without sanitization, a malicious runner could pass:

- `subagent_id = "agent-1\nprofile jaato-ws-evil { ... }"` —
  injection into the profile file, defining an unauthorized
  profile.
- `subagent_id = "../escape"` — path traversal in the filename.
- `subagent_id = "a"*1000` — DoS via giant filenames.

**Sanitization rule (§4.3.4):**

1. Length cap: 64 chars max.  Reject longer.
2. Character allow-list: `[A-Za-z0-9_-]` only.  Reject anything else.
3. Non-empty: reject empty / whitespace-only.

**Deliberately STRICT** — `_safe_fragment_filename`'s strategy
("collapse to `_`") would let `subagent_id="a/b"` become `a_b`,
masking a likely supervisor-side bug.  For sub-profile names we
REJECT instead so the runner-side caller sees the error.

Today's runner-side `_next_agent_id` produces `agent-N` (digits +
hyphen) — well within the allow-list.  Future callers that want
human-readable ids ("researcher", "executor") fit too.

### Sub-profile rule-set: conservative tightening

Default sub-profile body:

- **Workspace allows** — same as parent base (subagent inherits
  workspace per §4.3 invariant).  Without this the subagent
  literally can't do work.
- **Integrity deny rules** — same as parent base (mirrors
  `.jaato/agents/**`, `.jaato/profiles/**`, etc. write-denies).
- **Tool-hat-style read-denies** — same as parent's `tool_hat`
  (mirrors `.jaato/agents/**`, etc. read-denies for
  information-isolation between agents).
- **Drop external-reference admit** — no `add_reference_fragment`
  capability.  Isolated subagent can't broaden its allow-list at
  runtime.  Phase 5+ can re-add behind a supervisor-side opt-in.
- **Drop tool_hat sub-sub-profile** — flat profile, no further
  nesting.  Isolated subagent doesn't need finer-grained
  tool-execution scoping yet.

§4.3.7+ supervisor-side tightening flags (e.g., `agent_params.
isolated_workspace_subpath: "scratch"`) would narrow the workspace
allow further.  Out of scope for §4.3.4.

### Provisioning lifecycle

| Event | Action | Where |
|---|---|---|
| Handler routes valid request to helper (§4.3.3) | Helper validates + reconstructs profile, advances to `stage=sub_profile` | `SessionManager._spawn_isolated_runner` |
| §4.3.4: provision sub-profile | Call `AppArmorManager.provision_sub_profile(parent_session_id, sanitized_subagent_id, workspace_path)` | Inserted into helper after profile reconstruction |
| Provision success | Helper advances to `stage=sub_cgroup` (next stage waiting on §4.3.5) | Same helper, return shape |
| Provision failure | Return `stage=sub_profile` with kernel error | Helper short-circuits |
| §4.3.6 sub-runner shutdown | `AppArmorManager.teardown_sub_profile(parent, sub)` | Sub-runner cleanup path |
| Parent shutdown (§4.3.x) | Cascade: tear down all sub-profiles BEFORE the parent profile | New `_teardown_profile_impl` extension or §4.3.6 wire-up |

**Cascade safety**: parent teardown's existing pre-removal worker-
sweep doesn't know about sub-profiles.  §4.3.4 ships the
`teardown_sub_profile` method; §4.3.6 wires the cascade into
parent teardown when the sub-runner lifecycle is wired.

### §4.3.4 commit scope

1. **`AppArmorManager.provision_sub_profile`** + impl + tests.
   - Validates `subagent_id` (allow-list, length cap, non-empty).
   - Writes file `{profile_dir}/jaato-ws-{parent}__sub_{sub}`.
   - Renders sub-profile body via new `_render_sub_profile()` (mirrors
     `_render_profile` but flat + with the conservative-tightening
     rule set above).
   - Loads via `apparmor_parser -r`.

2. **`AppArmorManager.teardown_sub_profile`** + impl + tests.
   - Symmetric to `_teardown_profile_impl`.
   - Worker-sweep deferred to §4.3.6 (sub-runner lifecycle not wired
     yet).

3. **Helper extension** in `SessionManager._spawn_isolated_runner`:
   - After profile reconstruction succeeds, attempt
     `apparmor_manager.provision_sub_profile(...)`.
   - On success: return `stage=sub_cgroup` with `apparmor_profile`
     name populated.
   - On failure: return `stage=sub_profile` with kernel-error message.

4. **Tests**:
   - Sanitization: allow-list, length cap, non-empty (helper-level).
   - Profile name + filename template pinning.
   - Provision-success path → file written + parser invoked +
     return value True.
   - Provision-failure paths → bad subagent_id, kernel parser
     failure, write failure.
   - Teardown-success roundtrip.
   - Helper integration: stage=sub_profile → stage=sub_cgroup
     transition when AppArmor wired; helper returns stage=sub_profile
     with diagnostic when not.

### Scope NOT in §4.3.4

- Sub-cgroup creation — §4.3.5.
- Parent-teardown cascade wire-up — §4.3.6 (lifecycle wiring).
- Sub-runner self-confinement at spawn time — §4.3.5+§4.3.6 (the
  `_spawn_session_runner_unconditional(profile_name=...)` call).
- Sub-profile rule-set customization via supervisor flags — Phase 5+.
- Sub-profile `add_reference_fragment` support — Phase 5+ behind
  explicit opt-in.

### Phase 5 hardening surface (recorded for cleanup runway)

1. **Sub-profile rule-set is daemon-trusted** — the runner can request
   sub-profile creation; the daemon decides the rules.  Phase 5
   should let the SUPERVISOR (parent runner's subagent plugin)
   declare tightenings via the wire shape, with a daemon-side
   allow-list of permitted tightening keys.
2. **Sub-profile filename collision** — if a parent has two
   subagents with `subagent_id="agent-1"` (sequential spawns where
   the first finished and was torn down), the second reuses the same
   filename.  Today this is fine because teardown happens fully.
   Phase 5 should consider monotonic suffixing in case teardown
   races spawn.
3. **No sub-profile reload-on-fragment-add** — `add_reference_fragment`
   touches the parent's profile.  Sub-runner-initiated fragments
   would need to target the sub-profile; not wired (sub-profile has
   no `.refs.d/` companion dir).  Acceptable for §4.3.4 since
   sub-profile drops fragment admit anyway.
4. **`apparmor_parser -r` reload affects only the named file** —
   safe today, but if sub-profiles ever migrate to true hats (inside
   parent file), reload would affect parent.  Document the constraint.

### Sub-step risk register entry

Risk: sub-profile-load failure mid-spawn leaves a written file
without a kernel-loaded profile.  Mitigation: `provision_sub_profile`
unlinks the file on `apparmor_parser` failure (mirrors
`_provision_profile_impl`'s pattern).

---

## Audit 7 — §4.3.5 sub-cgroup creation + attach

**Sub-step:** §4.3.5 of the §4.3 sub-track.

**Original framing** (Audit 4): "Cgroup sub-cgroup nesting:
creation + attach + cleanup | audit + 1".

**Why this needs a pre-implementation audit:** cgroup v2 has
structural constraints that determined the design.  Naïve
"nest under parent" runs into the "no internal processes" rule;
need to settle the structure up-front so §4.3.6 forwarding
wire-up isn't blocked.

### Existing cgroup machinery (load-bearing reference points)

| Component | Location | Today's role | §4.3.5 usage |
|---|---|---|---|
| Provision + write limits | `CgroupsManager.provision_cgroup(session_id, config)` (cgroups.py:181) | Per-session cgroup at `/sys/fs/cgroup/jaato/jaato-ws-{session_id}` | Reused as-is, called with the isolated session id |
| Teardown via `cgroup.kill` + rmdir | `CgroupsManager.teardown_cgroup()` (cgroups.py:235) | Per-session cleanup | Reused as-is for sub-cgroup teardown |
| Attach pid (pre-exec callback) | `CgroupsManager.make_attach_callback(session_id)` (cgroups.py:294) | Subprocess.Popen `preexec_fn` for newly-spawned runner | Reused as-is in §4.3.6 sub-runner spawn |
| Name template | `get_cgroup_name(session_id)` → `jaato-ws-{session_id}` (cgroups.py:173) | Cgroup directory name | Same template, called with the isolated session id |

### Sub-cgroup structure decision: SIBLING, not NESTED

The obvious "true nesting" structure (`{parent_cgroup}/sub_{subagent}/`)
hits a cgroup v2 hard constraint: **the "no internal processes"
rule**.  A cgroup can either:

- Have processes in `cgroup.procs`, OR
- Have children with `cgroup.subtree_control` controllers enabled

NOT both.  The parent cgroup (`/sys/fs/cgroup/jaato/jaato-ws-{parent}/`)
already hosts the parent runner's pid.  To enable
`cgroup.subtree_control` for child resource delegation, we'd need
to MIGRATE the parent runner to a sub-cgroup of itself first (e.g.,
`{parent_cgroup}/main/`) — a non-trivial restructuring that
affects all of `provision_cgroup`'s callers.

**§4.3.5 picks SIBLING structure.**  Both parent + sub cgroups live
at `/sys/fs/cgroup/jaato/`:

- Parent: `/sys/fs/cgroup/jaato/jaato-ws-{parent}/`
- Sub:    `/sys/fs/cgroup/jaato/jaato-ws-{parent}__sub_{subagent}/`

Path leaf matches §4.3.3's `isolated_session_id` + §4.3.4's
AppArmor sub-profile filename — three identifiers, one template.

**Trade-off:** sub-cgroup limits don't compose under parent's
limits.  An isolated subagent declaring `memory.max=4G` gets 4G
even if the parent has `memory.max=2G`.  Per §4.3 design intent
("isolated runner" = SEPARATE confinement), this is the desired
behavior — the supervisor explicitly opted into isolation; bounds
are independent.

**Design intent alignment:** the parent design §4.3 use case is
"untrusted code-execution subagent inside an otherwise-trusted
workspace".  The supervisor's bounds are about its own work, NOT
about caging the subagent — that's what the SUB-cgroup is for.
Sibling structure makes the supervisor/subagent bounds independent,
which matches the threat model.

**Phase 5+ "true nesting":** if a future use case needs composable
bounds (sub-cgroup bounded by parent's bounds), the restructuring
(`provision_cgroup` migrates the runner to `/main/`, enables
`cgroup.subtree_control` on parent, creates child) lands as a
single coordinated change.  Out of scope for §4.3.5.

### Cgroup directory name template

| Surface | Template | Example |
|---|---|---|
| Cgroup directory leaf | `jaato-ws-{parent_session_id}__sub_{subagent_id}` | `jaato-ws-S-A__sub_agent-1` |
| Full cgroup path | `{cgroups_root}/jaato-ws-{parent}__sub_{subagent}/` | `/sys/fs/cgroup/jaato/jaato-ws-S-A__sub_agent-1/` |

Matches AppArmor sub-profile filename from §4.3.4 + the isolated
session id from §4.3.3 — a single template flows through all
three subsystems.

### subagent_id validation: hoisted to shared helper

§4.3.4 anchored `_validate_subagent_id` on `AppArmorManager`.
§4.3.5 needs the same validation for cgroup directory names —
strict allow-list `[A-Za-z0-9_-]`, 64-char cap, non-empty.

**Refactor:** hoist to module `server/subagent_id.py` (public
utility, not `_private`).  Both managers + the SessionManager
helper delegate to it.  Pure function — no state, no class.

**Why refactor as part of §4.3.5:** the function is small (~25
lines), the rules are identical, and divergence between AppArmor's
validation and cgroup's would be a latent confused-deputy bug.
Hoisting now is cheap; later it becomes harder.

### RuntimeLimits source + missing-limits handling

The sub-cgroup's limits come from
`profile.runtime_limits` (reconstructed via `build_inline_profile`
in `_spawn_isolated_runner`).

When the SubagentProfile declares no `runtime_limits`:

- Today's `provision_cgroup` skips cgroup creation entirely
  (`config.has_kernel_limits()` is False).
- For isolated subagents, this means: no sub-cgroup created.  The
  sub-runner subprocess will inherit the daemon's default cgroup
  (NOT the parent's cgroup, since sibling structure means parent
  and sub are at the same level).

**§4.3.5 behavior:**

- Profile has `runtime_limits` with kernel limits → create
  sub-cgroup.
- Profile has no `runtime_limits` → skip cgroup creation.
  Sub-runner inherits the daemon's default cgroup.

This is a documented gap: an isolated subagent without explicit
limits has no kernel-level resource bounding.  AppArmor isolation
(filesystem + capabilities) still applies; cgroup isolation does
not.

**Phase 5+ hardening**: default `RuntimeLimits` for isolated
subagents (e.g., 2 GiB memory, 128 pids).  Out of scope for §4.3.5.

### Cleanup safety on partial-failure

The §4.3.5 sub-step exists in the middle of a multi-stage
provisioning chain:

```
§4.3.3: profile reconstruction
§4.3.4: sub-AppArmor profile provision   ← can succeed
§4.3.5: sub-cgroup provision             ← can fail HERE
§4.3.6: sub-runner spawn + forwarding
```

If §4.3.5 fails AFTER §4.3.4 succeeded, the sub-AppArmor profile
is loaded but the sub-runner will never spawn.  Next attempt with
the same subagent_id would see the AppArmor sub-profile already
loaded (idempotent re-load is fine) but it's still cleaner to
roll back on failure.

**§4.3.5 rollback:** on cgroup provision failure, call
`apparmor_manager.teardown_sub_profile(parent, subagent)` to
clean up the just-loaded AppArmor sub-profile.  Best-effort —
teardown failure logs but doesn't change the §4.3.5 return shape.

### Stage advance: sub_cgroup → forwarding

| Outcome | Return stage |
|---|---|
| cgroups unavailable (no kernel cgroups v2, or `_apparmor_manager` not wired in SessionManager) | `stage=forwarding`, `cgroup_path=""` (documented gap — no cgroup isolation) |
| Profile has no `runtime_limits` | `stage=forwarding`, `cgroup_path=""` (documented gap) |
| `provision_cgroup` succeeded | `stage=forwarding`, `cgroup_path=<full path>` |
| `provision_cgroup` failed | `stage=sub_cgroup`, AppArmor sub-profile torn down |

Note: `stage=forwarding` is the NEXT stage (§4.3.6's body).  The
helper's "advance past sub_cgroup" happens both on cgroup success
AND on cgroup-not-applicable (graceful degradation).  Only kernel
errors stop at `stage=sub_cgroup`.

### §4.3.5 commit scope

1. **Hoist `_validate_subagent_id`** from `AppArmorManager` to
   `server/subagent_id.py`.  Pure-function module.  Update
   `AppArmorManager._validate_subagent_id` to delegate (preserves
   classmethod API for backwards compat with §4.3.4 tests).

2. **Helper extension** in `SessionManager._spawn_isolated_runner`:
   - After AppArmor sub-profile provisioning succeeds (existing
     §4.3.4 path), resolve cgroups manager via new
     `_resolve_cgroups_manager()` helper (mirrors
     `_resolve_apparmor_manager`).
   - If cgroups available + `profile.runtime_limits` has kernel
     limits: call `cgroups_manager.provision_cgroup(
     isolated_session_id, profile.runtime_limits)`.
   - On success → `stage=forwarding`, `cgroup_path=<full path>`.
   - On failure → tear down sub-AppArmor profile + return
     `stage=sub_cgroup` with cgroup-kernel error.
   - If cgroups unavailable OR no `runtime_limits` → `stage=forwarding`
     with `cgroup_path=""`.

3. **`SessionManager._resolve_cgroups_manager()` helper**:
   - Returns `getattr(self, "_cgroups_manager", None)`.
   - Mirrors `_resolve_apparmor_manager`.

4. **Tests**:
   - Hoisted validation function (existing §4.3.4 test pins still
     pass because they exercise the public surface).
   - Helper extension: cgroups available + has limits → stage=forwarding
     with non-empty cgroup_path.
   - Helper extension: cgroups available + no limits → stage=forwarding
     with empty cgroup_path.
   - Helper extension: cgroups unavailable → stage=forwarding with
     empty cgroup_path.
   - Helper extension: cgroups failure → stage=sub_cgroup + AppArmor
     teardown invoked.

### Scope NOT in §4.3.5

- Sub-runner subprocess spawn — §4.3.6 (passes `cgroup_attach=...`
  to `RunnerSpawner.spawn`).
- Cross-runner forwarding (prompt-forward IN, output-forward OUT)
  — §4.3.6.
- Parent-teardown cascade for sub-AppArmor + sub-cgroup — §4.3.6
  (lifecycle wiring).
- Default `RuntimeLimits` for isolated subagents — Phase 5+.
- True nested cgroup structure (parent restructure + subtree_control)
  — Phase 5+.

### Phase 5 hardening surface (recorded for cleanup runway)

1. **Default `RuntimeLimits` for isolated subagents** — without
   profile-declared limits, sub-runner has no kernel-level resource
   bounding.  Phase 5 should add a conservative default (2 GiB / 128
   pids / cpu.weight=100) applied when the supervisor's opt-in
   requested isolation but didn't supply limits.
2. **Nested-cgroup structure** — sibling structure makes parent +
   sub bounds independent.  Use cases that need composable bounds
   (sub bounded by parent) need the parent-restructure (`/main/`
   sub-cgroup + `subtree_control`).  Coordinated change, not
   incremental.
3. **Cgroup teardown coordination with sub-runner shutdown** —
   §4.3.6 wires this; §4.3.5 ships only the cgroup creation.
4. **Cgroup-leak audit at session shutdown** — if a sub-runner
   crashes before parent shutdown, the sub-cgroup may be left with
   zombie processes.  `cgroup.kill` handles atomic termination;
   pre-§4.3.6 rollback path covers immediate failures.  Phase 5
   should add an audit pass at parent teardown that finds + reaps
   any leaked sub-cgroups by name pattern.

### Sub-step risk register entry

Risk: cgroup-create failure mid-spawn leaves the AppArmor
sub-profile loaded.  Mitigation: §4.3.5 rollback calls
`teardown_sub_profile` on failure path.  Risk: rollback's own
teardown fails — log + continue; AppArmor sub-profile remains
loaded.  Next session attempt sees the loaded profile and
provision_sub_profile is idempotent (re-load via apparmor_parser
-r is fine), so no operator action needed.


