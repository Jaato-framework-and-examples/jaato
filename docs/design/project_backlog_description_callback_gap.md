# Backlog: daemon-side description-callback hook is silently broken post-6.6.4.3b

> **Status**: Pre-existing gap surfaced by §7c step 6.6.4.4 audit (Finding 2).
> Not in scope for the §7c series.  See
> [`per_session_confined_runner_phase3_3c_rpc_surface.md`](per_session_confined_runner_phase3_3c_rpc_surface.md)
> for full context.

## Problem

`JaatoServer._setup_session_plugin` wires `on_description_changed` on the
**daemon-side** `session_plugin` instance:

```python
# jaato-server/server/core.py:2360-2367
if hasattr(session_plugin, 'set_description_callback'):
    def on_description_changed(session_id: str, description: str) -> None:
        self.emit(SessionDescriptionUpdatedEvent(
            session_id=session_id,
            description=description,
        ))
    session_plugin.set_description_callback(on_description_changed)
```

The `_on_description_changed` callback fires from inside the
`session_plugin`'s `set_description` tool handler:

```python
# jaato-server/shared/plugins/session/file_session.py:743-747
if self._current_session_id and self._on_description_changed:
    try:
        self._on_description_changed(self._current_session_id, description)
    except Exception:
        pass  # Don't fail if callback fails
```

Post-§7c step 6.6.4.3b seat-flip, the runner has its own
`session_plugin` instance constructed from the bootstrap envelope's
plugin list.  When the model invokes the `set_description` tool, it fires
**the runner-side instance's** callback — the daemon-side instance's
callback is never invoked.

Effect: `SessionDescriptionUpdatedEvent` no longer emits when the model
sets a description.  Sessions retain whatever description was set
pre-seat-flip, or none at all.

## What still works vs what's broken

| Surface | Status |
|---|---|
| Daemon-side `session_plugin` registration in `PluginRegistry` (line 2371) | ✅ works (registry-side wiring) |
| Daemon-side description-callback wiring (line 2366) | ⚠️ wired but never fires (callback is on the wrong instance) |
| Runner-side `session_plugin` instantiation at bootstrap | ✅ works |
| Runner-side `set_description` tool execution | ✅ works (model can set descriptions) |
| Persistence layer's storage of new descriptions | ✅ works (runner-side plugin writes to disk) |
| Daemon emission of `SessionDescriptionUpdatedEvent` for new descriptions | ❌ **broken** |
| Client UI updating session-picker labels in real-time | ❌ **broken** (event never fires) |

## Audit context

Surfaced during the §7c step 6.6.4.4 implementation-review audit.  The
hook was wired in `_setup_session_plugin` and survives intentionally
even after the audit's safe-only WIRING deletions — see the comment
block added in commit `9ea2f827`:

> Note (6.6.4.4 audit Finding 2, deferred): the daemon-side
> ``set_description_callback`` below is wired on the daemon-side
> ``session_plugin`` instance — but the model invokes ``set_description``
> runner-side, firing the runner-side instance's callback.  Daemon
> never sees it.  This regression is pre-existing from 6.6.4.3b, not
> caused by 6.6.4.4.  Fix planned via a new ``description_updated``
> NotificationFrame event_type.

## Likely fix shapes

### Option A — new `description_updated` NotificationFrame event_type

Extends the §7c step 6.6.4.1 protocol from 8 event_types to 9:

| Event_type | Source | Payload |
|---|---|---|
| `description_updated` (new) | runner-side `session_plugin._on_description_changed` shim | `{session_id: str, description: str}` |

Runner-side install-machinery additions in
`_install_session_notification_callbacks`:

```python
# Mirror existing _on_subscribed pattern at lines 2776-2785.
session_plugin = ... # from session.runtime.registry?
if session_plugin and hasattr(session_plugin, 'set_description_callback'):
    originals["description_updated"] = session_plugin._on_description_changed

    def _desc_cb(session_id: str, description: str) -> None:
        try:
            rpc.emit_notification(
                request_id=request_id,
                event_type=rpc._NOTIF_DESCRIPTION_UPDATED,
                payload={"session_id": session_id, "description": description},
            )
        except Exception:
            logger.exception("description_updated notify raised")
    session_plugin.set_description_callback(_desc_cb)
```

Daemon-side demuxer adds a 9th branch in
`_build_send_message_notification_handler`:

```python
if event_type == "description_updated":
    server.emit(SessionDescriptionUpdatedEvent(
        session_id=payload.get("session_id", "") or "",
        description=payload.get("description", "") or "",
    ))
    return
```

Pros: matches the established 8-event protocol exactly; minimal new
wire infrastructure; preserves the existing daemon-side
`SessionDescriptionUpdatedEvent` shape.

Cons: runner-side install needs to find the `session_plugin` instance
in the runner's runtime/registry (one-time lookup at session-init).

### Option B — direct event emission via existing mechanism

If a separate event-bus channel exists between runner and daemon
(beyond NotificationFrames), wire the runner-side
`session_plugin._on_description_changed` to emit directly.  Less likely
to be cleaner than Option A.

## Open questions

1. **Where is the runner-side `session_plugin` instance accessible from
   `RunnerRPC._handle_session_send_message`?**  Probably via
   `session._runtime.registry.get_plugin('session')` — verify.
2. **Does `set_description` ever fire outside a `send_message` call?**
   If the tool can be invoked from a non-send_message RPC path
   (unlikely but possible), the per-call install pattern misses it.
   Alternative: wire at runner bootstrap, not per-call.
3. **Should the daemon-side `_setup_session_plugin` description-callback
   wiring be deleted once the fix lands?**  Yes — the daemon-side hook
   becomes dead-weight once the runner-side path is wired.

## Files to touch (when scheduled)

- `jaato-server/server/runner/envelope.py` — new
  `_NOTIF_DESCRIPTION_UPDATED = "description_updated"` constant
- `jaato-server/server/runner/rpc.py` — extend
  `_install_session_notification_callbacks` /
  `_restore_session_notification_callbacks`; new constant
- `jaato-server/server/core.py` — extend
  `_build_send_message_notification_handler` with 9th branch; delete
  the daemon-side `set_description_callback` wiring at line 2366 once
  runner-side fix lands
- Tests: extend
  `jaato-server/server/runner/tests/test_session_send_message_notification_emit.py`
  with description-callback emission test; extend
  `jaato-server/server/tests/test_send_message_seat_flip_643b.py` with
  daemon-side demuxer branch test
