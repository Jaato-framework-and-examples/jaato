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
