# Phase 4 §D — agent_params propagation to runner (third §7c envelope gap)

**Discovered**: 2026-05-11 during the §C smoke test.  After the §B + §C
fixes, the documenter agent successfully bootstrapped, authenticated
with openrouter, and made multiple HTTP completions — but its `cli`
tool calls were targeting `tmux send-keys -t 0` (window `0` =
operator's first claude pane) instead of the harness-provided `7:1`
(the `tui-manual-build` window).  Keystrokes leaked into the
operator's interactive panel.

## Root cause

Third class of post-§7c envelope-propagation gap (alongside §B env
vars and §C plugin_configs):

`runner_spawn.build_session_envelope` hard-coded
``agent_params={}`` on the wire envelope:

```python
return SessionInitEnvelope(
    ...
    agent_params={},  # ← hard-coded; throws away the caller-supplied dict
    ...
)
```

The walker's call

```python
await client.create_session(
    profile="documenter", agent="documenter",
    agent_params={"feature_id": ..., "tmux_pane": "7:1", ...},
)
```

→ landed in `SessionManager._create_session_impl(agent_params=...)`
→ flowed into `_resolve_agent(agent_name, agent_params, ...)` for the
  `{{name}}` template substitution pass (line 1869)
→ went nowhere else.

The `{{!py:scripts/prefetch_documenter_brief.py}}` placeholder in the
agent's persona markdown was **not** expanded during `_resolve_agent`
(which only handles bare `{{name}}` placeholders, line 358).  The
`{{!py:...}}` markers survived through to the runner-side
`JaatoSession.configure()` step (`jaato_session.py:1873-1881`), where
`expand_py_placeholders(self._system_instruction, ctx, ...)` runs with
``ctx = build_render_context(self, agent_params=self._agent_params)``.

But `self._agent_params` came from
``runtime.create_session(agent_params=envelope.agent_params or None)``
in the runner's session-build path (`runner/session.py:489`).  With
``envelope.agent_params == {}`` (the hard-coded value),
``self._agent_params`` ended up empty, the prefetch script's
`params.get("tmux_pane")` returned ``None``, and the prefetch emitted
its `[prefetch error: agent_params is missing required keys: ...]`
text block.

Nemotron, faced with a brief that explicitly told it the target was
missing, fell back to a reasonable guess: ``tmux send-keys -t 0``.
Window ``0`` is the operator's first claude pane.  Hence the
linefeed-leak.

## Fix (Option D-1)

Forward `agent_params` through the existing envelope chain.  Zero
runner-side changes — `runtime.create_session(agent_params=...)` is
already wired; it just had nothing to forward.

### Schema-level

`BootstrapEnvelope` (daemon-internal, never crosses the wire) gains
``agent_params: Dict[str, str] = field(default_factory=dict)``.

`SessionInitEnvelope` already had ``agent_params`` (schema v1).  No
wire-format change needed — only the writer path.

### Path-level

1. `SessionManager._create_session_impl`: pass the originating
   ``agent_params`` into the new `BootstrapEnvelope.agent_params`
   field.
2. `SessionManager._construct_and_initialize_server`: stash a copy
   on the per-session JaatoServer via ``server._agent_params =
   dict(envelope.agent_params or {})``.  The JaatoServer is per-
   session (one instance per session, GC'd at session end) — not
   centralized daemon state, just the per-session daemon-side handle
   holding session data during the envelope-build window.
3. `runner_spawn.build_session_envelope`: read
   ``getattr(server, "_agent_params", {})`` and pass to
   ``SessionInitEnvelope(agent_params=...)``.

Total: ~15 LoC across 3 files.

### Why not D-2 (daemon-side prefetch expansion)

Considered and rejected: rendering `{{!py:...}}` placeholders daemon-
side would centralize session-specific work (rendering with session-
specific agent_params) on the daemon process.  The user's
architectural principle: *"daemon should not centralize what is
session specific, but only shared plugin implementation"*.  D-1
preserves the rule — session-specific rendering happens runner-side
where the per-session JaatoSession owns the agent_params.

## Test plan

Four regression pins in `test_agent_params_passdown_phase4d.py`:

1. ``test_envelope_agent_params_empty_when_server_has_no_stash`` —
   pre-§D behaviour preserved when no agent_params is supplied.
2. ``test_envelope_agent_params_forwarded_from_server_stash`` — when
   `server._agent_params` is populated, the wire envelope carries
   the same dict verbatim.
3. ``test_envelope_agent_params_dict_is_independent_copy`` —
   defensive deep-copy so mutations to the envelope's view don't
   leak back into the server's stash (or vice versa across
   subsequent uses).
4. ``test_session_init_envelope_agent_params_round_trip`` — the wire
   format's `to_dict`/`from_dict` survives the §D-typical payload.

Real-provider integration: rerun the harness with the exit-menu
sidecar deleted; documenter should target `tmux send-keys -t 7:1 ...`
exclusively, never leaking to window `0`.

## Out of scope

- **Disk-restore / WS-standalone / ephemeral subagent paths**: these
  go through different envelope builders.  Only IPC create_session
  (the harness path) is patched here.  Other paths can land in a
  follow-up if/when they show similar gaps.
- **Subagent fan-out `agent_params`**: `spawn_subagent(...)` uses a
  parallel path through `JaatoRuntime.create_session(agent_params=...)`
  which already works — no envelope round-trip.  Same wire field on
  `SessionInitEnvelope` is used when spawning a subagent runner
  (Phase 4 §3.11 isolated mode); covered by the existing tests.
