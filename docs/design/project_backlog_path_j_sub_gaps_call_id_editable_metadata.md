# Backlog: Path J sub-gaps — call_id propagation + editable_metadata schema lookup

> **Status**: Filed as Phase 3 closure follow-ups.  Identified at audit
> time (Path J pre-implementation audit, commit `36e2a9f9`) and
> deliberately deferred to keep Path J's critical-path fix surgical.
>
> Cycle 13 verified the ASK round-trip works end-to-end without these
> fields populated (basic permission flow renders + resolves
> correctly).  Cycle-13 verdict also flagged a possible "cumulative 16
> PermissionRequestedEvents + Error: 1bcd84... rendering" degraded-UX
> symptom in multi-turn workflows — investigate whether either J.A or
> J.B explains it before closing this backlog.

## Context

Path J (cycle 12 → commit `1a8a37e4`) closed Layer 10 by emitting
`PermissionInputModeEvent` companion alongside `PermissionRequestedEvent`
from `PromptOperatorHandler.handle()`.  Pre-§7c both events were emitted
from the daemon-side permission plugin's hook at `core.py:3105` with two
fields the post-§7c emit does NOT populate.

## Sub-gap J.A — `call_id` propagation

**What**: `PermissionInputModeEvent.call_id` is currently `None`.

**Why missing**: `PromptPayload` (the runner→daemon ASK payload at
`shared/plugins/permission/types.py:26`) doesn't carry a `call_id` field.
The runner-side permission plugin's ASK origin has the call_id but
doesn't propagate it through the RPC payload.

**Impact**: TUI's per-tool-block correlation (which tool's prompt is
showing in the popup) may degrade when parallel tools are in flight.
Single-tool flows unaffected.  Cycle-13 verified single-tool ASK works.

**Fix shape**:
1. Add `call_id: Optional[str] = None` field to `PromptPayload` +
   round-trip through `to_dict`/`from_dict`.
2. Runner-side permission plugin populates `call_id` from the
   active tool-call context when constructing the payload.
3. `PromptOperatorHandler.handle()` passes `payload.call_id` to
   `PermissionInputModeEvent.call_id`.
4. Tests pin the propagation through both directions.

**Estimated effort**: ~30 LoC, 1 audit + code commit.

## Sub-gap J.B — `editable_metadata` schema lookup

**What**: `PermissionInputModeEvent.editable_metadata` is currently
`None`.

**Why missing**: Pre-§7c the daemon-side hook at `core.py:3093-3099`
looked up the editable schema via:
```python
schema = server.permission_plugin._get_tool_schema(tool_name)
if schema and schema.editable:
    editable_metadata = {
        "parameters": schema.editable.parameters,
        "format": schema.editable.format,
    }
```

`PromptOperatorHandler` doesn't have a reference to
`server.permission_plugin` (it's constructed in `JaatoServer.set_runner_rpc`
with only the `emit_event` callback per Step 7.1).

**Impact**: The TUI's "edit and approve" flow can't activate — users
who want to edit tool args before approving have to deny + re-prompt
the model instead.  Default approve/deny flow unaffected.

**Fix shape (Option A)**: pass `permission_plugin` reference into
`PromptOperatorHandler.__init__`.  Schema lookup happens inside
`handle()` before the emit.

**Fix shape (Option B)**: include `editable_metadata` in `PromptPayload`
itself — runner-side permission plugin includes schema metadata in the
ASK payload.  Avoids daemon-side schema-plugin coupling but requires
the runner's schema to match the daemon's (both Path D-loaded the same
runner-tier plugin set, so this should hold).

**Decision**: Option B is architecturally cleaner (single source of
truth: the runner's permission plugin owns the schema knowledge).
Option A is faster.  Re-evaluate when the editing UX is reprioritized.

**Estimated effort**: Option A ~40 LoC, Option B ~60 LoC.

## Triggering conditions for prioritization

These sub-gaps stay deferred until ONE of:

1. **Multi-turn workflow surfaces UX degradation** — cycle 13's "16
   PermissionRequestedEvents + Error: 1bcd84..." suggests this may
   already be an active impact zone.  Investigate before closing.
2. **Edit-and-approve flow becomes load-bearing** — currently rare in
   practice; users typically deny + re-prompt.  If a feature surfaces
   that depends on it, prioritize J.B.
3. **Parallel-tool workflows reach Phase 4+ critical path** — when
   multiple tools fire concurrently, J.A's call_id correlation becomes
   structural.

## Related artifacts

- Path J audit: `docs/design/per_session_confined_runner_phase3_3c_rpc_surface.md`
  §"Path J pre-implementation audit (cycle 12 — sixth chain)"
- Path J code: commit `1a8a37e4`
- Path J tests: `jaato-server/server/runner_rpc_handlers/tests/test_permission_input_mode_emit_path_j.py`
  (pins `call_id is None` and `editable_metadata is None` explicitly
  so future fixes break these pins and force re-audit)
