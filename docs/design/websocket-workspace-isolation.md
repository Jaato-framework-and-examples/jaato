# WebSocket Workspace Management & AppArmor Isolation

## Problem Statement

When clients connect via IPC, both client and server share a filesystem — the
client's `cwd` is a valid workspace path on the server. When clients connect via
WebSocket, the client is typically on a different host. The client's local paths
are meaningless to the server. The server must provision and manage workspaces
on behalf of remote clients, and enforce isolation between them.

## Design Goals

1. **Server-provisioned workspaces** — WS clients get an isolated directory on
   the server, auto-created on session creation.
2. **Per-client isolation** — one client cannot access another's workspace,
   enforced at the kernel level via AppArmor.
3. **Template system** — server admin pre-configures credential templates that
   new workspaces inherit.
4. **Lifecycle management** — workspaces are reaped when sessions expire.
5. **Backward compatibility** — IPC mode is unchanged. The existing
   "select workspace" flow for WS still works alongside auto-provisioning.

## Architecture Overview

```
JaatoWSServer
  ├── WorkspaceProvisioner (NEW)
  │   ├── provision(session_id, template?) → WorkspaceInfo
  │   ├── teardown(session_id)
  │   ├── reap_expired(max_age)
  │   └── get_workspace(session_id) → Optional[WorkspaceInfo]
  │
  ├── AppArmorManager (NEW)
  │   ├── provision_profile(session_id, workspace_path)
  │   ├── teardown_profile(session_id)
  │   ├── wrap_command(session_id, command) → prefixed command
  │   └── is_available() → bool
  │
  ├── WorkspaceManager (EXISTING, enhanced)
  │   ├── Per-client tracking (replaces single _selected_workspace)
  │   └── Integration with WorkspaceProvisioner
  │
  └── Per-client state
      └── _client_workspaces: Dict[client_id, WorkspaceInfo]
```

### Directory Layout

```
{workspace_root}/
├── templates/                     # Admin-configured templates
│   ├── default/
│   │   ├── .env                   # Provider credentials
│   │   └── .jaato/
│   │       └── profiles/          # Default agent profiles
│   └── research/                  # Optional named templates
│       ├── .env
│       └── .jaato/
├── sessions/                      # Auto-provisioned per-session
│   ├── {session_id_1}/            # Isolated workspace
│   │   ├── .env                   # Copied from template
│   │   ├── .jaato/
│   │   │   ├── sessions/          # Session persistence
│   │   │   └── profiles/
│   │   └── ...                    # Agent-created files
│   ├── {session_id_2}/
│   └── ...
└── workspaces/                    # Manually created (existing flow)
    ├── my-project/
    └── another-project/
```

## Implementation Status

### Phase 1: WorkspaceProvisioner ✅

**File: `jaato-server/server/workspace_provisioner.py`**

Handles creation, tracking, and cleanup of auto-provisioned workspace
directories for WebSocket sessions. 22 unit tests passing.

Key implementation details:
- `ProvisionedWorkspace` dataclass with session_id, path, template, timestamps, client_id
- Thread-safe with `threading.Lock`, manifest persistence in `{root}/sessions/manifest.json`
- Daemon reaper thread with configurable interval and `on_teardown` callback
- Template copying via `shutil.copytree` from `{root}/templates/{name}/`

**Key decisions:**
- Templates are copied (not symlinked) so each workspace is fully independent.
- `.env` files in templates can contain credentials the admin sets up once.
- `reap_expired()` is called by a background thread/timer, similar to the
  interactive shell reaper pattern.

### Phase 2: AppArmorManager ✅

**File: `jaato-server/server/apparmor.py`**

Manages AppArmor profiles for workspace confinement. Designed to be optional —
when AppArmor is not available, the system falls back to directory-level
sandboxing (existing behavior). 22 unit tests passing.

Key implementation details:
- `PROFILE_TEMPLATE` class constant with format placeholders for session_id, workspace_path, venv_path, sessions_root
- Profile grants workspace rw, venv ro, denies sibling workspaces, allows network outbound, denies ptrace/mount/sys_admin
- `is_available()` checks: Linux, apparmor_parser on PATH, aa-exec on PATH, kernel module loaded, profile dir writable (result cached)
- `wrap_command()` for argv lists, `wrap_shell_command()` for shell strings with proper quote escaping
- Graceful cleanup on `apparmor_parser` failure during provisioning

**Key decisions:**
- **Fail-open**: If AppArmor is unavailable, the system works exactly as before
  with directory-level sandboxing. `is_available()` is checked once at startup.
- **Profile directory**: `/etc/apparmor.d/jaato/` — requires the server process
  to have write access (or use a sudoers rule for `apparmor_parser`).
- **`ix` transitions**: Subprocess executions inherit the profile, so child
  processes (git, python, curl) are also confined.
- **Deny sibling workspaces**: The `deny {sessions_root}/**` rule with a more
  specific allow for the session's own directory ensures cross-session isolation.

### Phase 3: CLI Plugin Integration ✅

**Modified: `jaato-server/shared/plugins/cli/plugin.py`**

- Added `_apparmor_wrapper` (argv) and `_apparmor_shell_wrapper` (shell string) attributes
- `set_apparmor_wrapper(argv_wrapper, shell_wrapper)` method
- Both `_execute()` and `_execute_streaming()` apply the appropriate wrapper before `Popen()`
- Shell mode uses `_apparmor_shell_wrapper`, non-shell mode uses `_apparmor_wrapper`
- No-op when wrappers are None (existing behavior preserved)

### Phase 4: Interactive Shell Plugin Integration ✅

**Modified: `jaato-server/shared/plugins/interactive_shell/plugin.py`**

- Added `_apparmor_shell_wrapper` attribute
- `set_apparmor_wrapper(shell_wrapper)` method
- Wraps spawn command string through wrapper before `ShellSession()` creation
- The wrapper prepends `aa-exec -p jaato-ws-{session_id} -- /bin/sh -c '...'`

### Phase 5: WebSocket Server Integration ✅

**Modified: `jaato-server/server/websocket.py`**

Key changes implemented:

1. **`__init__`**: Accepts `apparmor` (Optional[bool]), `default_template`, `workspace_max_age`
2. **`start()`**: Initializes `WorkspaceProvisioner` and `AppArmorManager` (with auto/required/disabled logic), starts reaper with `on_teardown` callback
3. **Per-client tracking**: `_client_provisioned: Dict[client_id, str]` maps clients to session IDs
4. **`_initialize_server_for_workspace()`**: Called from `_handle_config_update()` — auto-provisions workspace, creates `JaatoServer` with env_file, initializes in executor, applies AppArmor wrappers
5. **`provision_workspace()`**: Provisions via provisioner, sets up AppArmor profile
6. **`get_apparmor_wrappers()`**: Returns (argv_wrapper, shell_wrapper) closures
7. **Client disconnect**: Calls `_workspace_manager.remove_client()`, pops `_client_provisioned`
8. **Reaper**: Background thread calling `reap_expired()` + AppArmor teardown via callback
9. **CLI flags**: `--apparmor/--no-apparmor`, `--workspace-template`, `--workspace-max-age`
10. **`get_server_info()`**: Includes provisioned_workspaces count, available_templates, apparmor_available

### Phase 6: Session Manager Integration ✅

**Modified: `jaato-server/server/session_manager.py`**

1. **`Session` dataclass**: Added `provisioned: bool = False`
2. **`create_session()`**: Accepts `provisioned: bool = False` param
3. **`_save_session()`**: Persists `provisioned` flag in `metadata['provisioned']`
4. **`_load_session()`**: Restores `provisioned` flag from metadata

### Phase 7: JaatoServer Integration ✅

**Modified: `jaato-server/server/core.py`**

- `set_apparmor_wrapper(argv_wrapper, shell_wrapper)` method
- Propagates to CLI plugin (both wrappers) and interactive_shell plugin (shell wrapper)
- Uses `registry.get_plugin()` with `hasattr` guard for safety

### Phase 8: Workspace Reaper ✅

**Integrated into `WorkspaceProvisioner`**

- `start_reaper(interval, max_age, on_teardown)` — daemon thread
- `stop_reaper()` — signals shutdown via `threading.Event`
- `on_teardown` callback enables AppArmor profile cleanup for reaped sessions
- Called by WS server's `start()`, stopped by `stop()`

## Files Summary

| File | Action | Status | Description |
|------|--------|--------|-------------|
| `server/workspace_provisioner.py` | **NEW** | ✅ | Auto-provisioning, templates, reaper |
| `server/apparmor.py` | **NEW** | ✅ | AppArmor profile management |
| `server/websocket.py` | MODIFY | ✅ | Per-client workspaces, provisioning integration |
| `server/workspace_manager.py` | MODIFY | ✅ | Per-client tracking, integration with provisioner |
| `server/session_manager.py` | MODIFY | ✅ | Provisioned workspace awareness |
| `server/core.py` | MODIFY | ✅ | AppArmor wrapper propagation to plugins |
| `shared/plugins/cli/plugin.py` | MODIFY | ✅ | AppArmor command wrapping |
| `shared/plugins/interactive_shell/plugin.py` | MODIFY | ✅ | AppArmor command wrapping |
| `shared/tests/test_workspace_provisioner.py` | **NEW** | ✅ | 22 unit tests |
| `shared/tests/test_apparmor.py` | **NEW** | ✅ | 22 unit tests |
| `deploy/apparmor/` | **NEW** | DEFERRED | Example profiles, setup script |

## Configuration

### Server CLI flags (new)

```
--workspace-root PATH       # (existing) Root for workspaces
--apparmor / --no-apparmor   # Enable/disable AppArmor (default: auto-detect)
--workspace-max-age SECONDS  # Reaper max age (default: 86400)
--workspace-template NAME    # Default template name (default: "default")
```

### Environment variables (new)

| Variable | Purpose | Default |
|----------|---------|---------|
| `JAATO_WS_APPARMOR` | Enable AppArmor (`true`/`false`/`auto`) | `auto` |
| `JAATO_WS_WORKSPACE_MAX_AGE` | Max workspace age in seconds | `86400` |
| `JAATO_WS_DEFAULT_TEMPLATE` | Default template for provisioning | `default` |
| `JAATO_WS_PROFILE_DIR` | AppArmor profile directory | `/etc/apparmor.d/jaato` |

## Security Considerations

1. **AppArmor as defense-in-depth**: Directory sandboxing (file_edit, CLI
   plugins) remains the first line. AppArmor is the kernel-enforced backstop.

2. **Template credential management**: `.env` files in templates contain API
   keys. The templates directory should be readable only by the server user.
   Consider: per-client credentials via an auth layer (future work).

3. **Resource exhaustion**: AppArmor does not limit CPU/memory/disk. For
   production deployments, combine with cgroups (via systemd resource controls
   on the server service) and filesystem quotas.

4. **Profile privilege**: Loading AppArmor profiles requires either root or a
   sudoers rule. Alternative: pre-generate a pool of profiles with wildcard
   session IDs, avoiding runtime privilege needs.

5. **Escape vectors**: `ix` transitions mean child processes inherit the
   profile. However, if a confined process can write to a location that an
   unconfined process executes from, that's an escape. The deny rules on
   sibling workspaces and system directories mitigate this.

## Deployment Model

```
# Recommended: dedicated OS user
useradd --system --home-dir /srv/jaato --shell /bin/false jaato

# Directory setup
mkdir -p /srv/jaato/workspaces/{templates/default,sessions}
chown -R jaato:jaato /srv/jaato

# AppArmor profile directory
mkdir -p /etc/apparmor.d/jaato
chown jaato:jaato /etc/apparmor.d/jaato

# Sudoers rule for profile management (if not running as root)
echo "jaato ALL=(root) NOPASSWD: /sbin/apparmor_parser" > /etc/sudoers.d/jaato

# Start server
sudo -u jaato .venv/bin/python -m server \
    --web-socket :8080 \
    --workspace-root /srv/jaato/workspaces \
    --daemon
```

## Testing Status

1. ✅ **Unit tests**: `WorkspaceProvisioner` — 22 tests (provision, teardown, reap, templates, manifest persistence, activity tracking)
2. ✅ **Unit tests**: `AppArmorManager` — 22 tests (profile generation, `is_available()` mock, command/shell wrapping, provision/teardown with mocked subprocess)
3. ⬚ **Integration tests**: WS connect → provision → CLI command → verify
   confinement (requires AppArmor-enabled CI runner or mock) — DEFERRED
4. ✅ **Fallback tests**: Graceful degradation when AppArmor unavailable (covered by apparmor unit tests)

## Deferred Work

- **Deployment artifacts**: `deploy/apparmor/` directory with setup scripts,
  sudoers template, and example workspace templates. Needed for production
  deployment but not for the core feature.
- **WebSocket integration test**: End-to-end test covering WS connect → config
  update → workspace provisioning → AppArmor wrapping → message flow. Requires
  mocking the full WS server lifecycle.
- **Session deletion hook**: When a session is deleted via session manager,
  notify provisioner/apparmor for cleanup. Currently cleanup only happens via
  reaper or client disconnect.

## Future Work

- **Per-client authentication**: OAuth/API-key auth for WS clients, binding
  workspaces to authenticated identities instead of session IDs
- **Persistent workspaces**: Allow authenticated users to reconnect to their
  workspace across sessions
- **cgroup integration**: CPU/memory/disk limits per workspace via systemd
  scopes or direct cgroup management
- **Container mode**: Optional Docker/Podman backend as alternative to AppArmor
  for environments without kernel MAC
- **macOS sandboxing**: `sandbox-exec` profile generation for macOS deployments
- **Disk quotas**: Per-workspace filesystem quotas via project quotas (XFS) or
  quota tools (ext4)
