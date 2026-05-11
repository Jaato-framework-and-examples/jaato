# Phase 4 §C — profile.plugin_configs propagation to runner (backlog §3.3c.X)

**Discovered**: 2026-05-11 during the harness smoke test (post-Phase 4 §B
env-fix).  The documenter agent received its first response from
openrouter+nemotron correctly (FunctionCall extracted by the provider,
parts_count=1, finish_reason=TOOL_USE), then **hung at the permission
check**:

```
[21:31:24.049] SESSION_STREAMING_COMPLETE parts_count=1 finish=FinishReason.TOOL_USE
[21:31:24.099] SESSION_PART[0] text=None fc=cli_based_tool
[21:31:24.129] SESSION_TOOL_START name=cli_based_tool
[21:31:24.130] PERMISSION check_permission: acquiring policy lock for ASK on cli_based_tool
[21:31:24.131] PERMISSION acquiring channel lock... tool_schema=True, editable=True
                ── hang ──
```

The documenter profile sets `plugin_configs.permission.policy.defaultPolicy: allow`
but the runner-side `PermissionPlugin` initialised with hardcoded
`defaultPolicy: "ask"` and tried to ASK the walker, which doesn't respond
to permission events.

## Root cause

`runner_spawn.py:build_session_envelope` lines 195-208 only attach
per-plugin config to **plugins that appear in `profile.plugins:`**:

```python
plugin_configs_dict = dict(getattr(profile, "plugin_configs", {}) or {})
for name in names:                          # names = ["cli", "file_edit"]
    entry: dict = {"name": name, "preload": name in preloaded}
    cfg = plugin_configs_dict.get(name)
    if cfg:
        entry["config"] = dict(cfg)
    plugin_specs.append(entry)
```

Plugins that aren't in `profile.plugins:` but are auto-loaded by the runner
(`permission`, the `gc_*` strategy, framework formatters, etc.) **silently
drop their configs**.

A comment in `runner/session.py:251-262` already documents this as backlog
§3.3c.X: *"Profile-supplied `plugin_configs["permission"]` overrides aren't
currently in the envelope."*  The daemon-side path (`core.py:1788-1793`)
applies the same profile config correctly — runner side just doesn't see it.

## Fix shape (Option B-clean per user direction)

Promote per-plugin configs from a per-entry sidecar to a top-level
envelope field so **all** profile configs propagate, not just configs for
plugins in `profile.plugins`.  Drop the per-entry `config` field
entirely — no overlap to manage.

### Schema changes

`SessionInitEnvelope`:

- **Add** `plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)`.
  Carries the full `profile.plugin_configs` dict verbatim (one entry per
  plugin name, value is the per-plugin config dict).
- **Remove** `config` key from each `plugins[i]` entry. Entries become
  `{"name": str, "preload": bool}`.
- **Bump** `SESSION_ENVELOPE_VERSION` from `1` → `2`.  Forward-compat is
  intact: a new runner reading an old envelope (v1) sees an empty
  `plugin_configs` (default factory). An old runner reading a new envelope
  (v2) refuses per the existing `version > SESSION_ENVELOPE_VERSION` guard,
  which is the right behaviour for shape-breaking changes.

### build_session_envelope (`runner_spawn.py:185-256`)

```python
plugin_configs_dict = dict(getattr(profile, "plugin_configs", {}) or {})
for name in names:
    plugin_specs.append({"name": name, "preload": name in preloaded})
# Carry the whole plugin_configs map, not just configs for listed plugins.
```

`plugin_configs=plugin_configs_dict` is passed into the `SessionInitEnvelope`
ctor.

### Runner-side `_configure_runtime_plugins` (`runner/session.py:226-272`)

Two changes:

1. **envelope.plugin_configs replaces entry["config"]** (the two reader
   sites on lines 231 and 463 — see below).
2. **PermissionPlugin gets the profile override** mirroring daemon-side
   `core.py:1788-1793`:

```python
permission_init_config = {
    "channel_type": "queue",
    "channel_config": {"use_colors": False},
    "workspace_path": workspace_path,
    "policy": {
        "defaultPolicy": "ask",
        "whitelist": {"tools": [], "patterns": []},
        "blacklist": {"tools": [], "patterns": []},
    },
}
profile_perm_config = envelope.plugin_configs.get("permission")
if profile_perm_config:
    permission_init_config.update(profile_perm_config)
permission_plugin.initialize(permission_init_config)
```

Shallow-merge semantics match daemon-side at `core.py:1793`.

### Runner-side `_build_session` (`runner/session.py:455-475`)

Already builds a `plugin_configs` local dict from `entry.get("config")`.
Migrate to read from `envelope.plugin_configs` instead.

## Migration cost

Three call sites total:

| File | Line | Direction | Change |
|------|------|-----------|--------|
| `runner_spawn.py` | 200-208 | writer | drop `entry["config"]`, pass `plugin_configs` dict to envelope |
| `runner/session.py` | 229-234 | reader | iterate `envelope.plugin_configs` instead of `entry.config` |
| `runner/session.py` | 458-465 | reader | iterate `envelope.plugin_configs` instead of `entry.config` |
| `runner/session.py` | 252-262 | NEW | merge `envelope.plugin_configs.get("permission")` into init |
| `session_envelope.py` | 124, 145, 160, 198-217 | schema | add field, bump version, update to_dict/from_dict |

Total: ~30 LoC of changes.

## Test plan

Three regression pins:

1. **Envelope round-trip** with non-empty `plugin_configs`:
   - `to_dict` includes the new field; `from_dict` rebuilds it intact.
   - `to_dict` plugin entries have no `config` key.
   - schema_version=2 round-trips; v1 envelope deserializes with empty
     plugin_configs.

2. **build_session_envelope** with a profile that declares
   `plugin_configs.permission` (matching the documenter profile):
   - `envelope.plugin_configs["permission"]["policy"]["defaultPolicy"] == "allow"`.
   - `envelope.plugins[i]` has no `config` key.

3. **_configure_runtime_plugins** with envelope.plugin_configs.permission set:
   - The constructed `PermissionPlugin` ends up with `defaultPolicy="allow"`.
   - Without the override, fallback "ask" applies.

4. **Discipline-#9 real-provider integration**: rerun the harness with
   `exit-menu` sidecar deleted; verify the documenter executes its first
   `cli_based_tool` call instead of hanging on permission ASK.

## Out of scope

- **Other auto-loaded plugins with profile overrides** (none today; this
  fix unblocks them generically when they appear).
- **Daemon-side runner_spawn**'s reading of profile.permission: the
  daemon already applies it correctly at `core.py:1788-1793`; this fix
  brings the runner to parity.
- **Per-entry config field removal in tests**: the partition test
  `test_build_session_envelope.py` will need a small update to assert the
  new shape; that's a one-line change.
- **The cli plugin's tool name** `cli_based_tool` instead of `cli` —
  pre-existing oddity, not in scope.
