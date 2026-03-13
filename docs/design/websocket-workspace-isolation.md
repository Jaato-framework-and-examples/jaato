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

## Implementation Plan

### Phase 1: WorkspaceProvisioner

**New file: `jaato-server/server/workspace_provisioner.py`**

Handles creation, tracking, and cleanup of auto-provisioned workspace
directories for WebSocket sessions.

```python
@dataclass
class ProvisionedWorkspace:
    session_id: str
    path: str                       # Absolute path
    template: Optional[str]         # Template name used
    created_at: str                 # ISO timestamp
    last_activity: str              # ISO timestamp
    client_id: Optional[str]        # Owning client (if any)

class WorkspaceProvisioner:
    """Provisions isolated workspace directories for remote sessions.

    Each provisioned workspace is a subdirectory under
    ``{workspace_root}/sessions/{session_id}/``. Templates from
    ``{workspace_root}/templates/{name}/`` are copied into new workspaces
    to provide initial configuration (.env, .jaato/).

    Lifecycle:
        provision() → workspace created, template applied
        get_workspace() → lookup by session_id
        update_activity() → bump last_activity timestamp
        teardown() → remove workspace directory
        reap_expired() → remove workspaces exceeding max_age
    """

    def __init__(self, workspace_root: str, default_template: str = "default"):
        ...

    def provision(
        self,
        session_id: str,
        client_id: Optional[str] = None,
        template: Optional[str] = None,
    ) -> ProvisionedWorkspace:
        """Create an isolated workspace directory for a session.

        1. Create {workspace_root}/sessions/{session_id}/
        2. Copy template contents (if template dir exists)
        3. Create .jaato/ subdirectory
        4. Return ProvisionedWorkspace with absolute path
        """
        ...

    def teardown(self, session_id: str) -> None:
        """Remove a provisioned workspace and its contents."""
        ...

    def reap_expired(self, max_age_seconds: int = 86400) -> List[str]:
        """Remove workspaces not accessed within max_age. Returns removed IDs."""
        ...

    def get_workspace(self, session_id: str) -> Optional[ProvisionedWorkspace]:
        ...

    def list_workspaces(self) -> List[ProvisionedWorkspace]:
        ...

    def update_activity(self, session_id: str) -> None:
        ...
```

**Key decisions:**
- Templates are copied (not symlinked) so each workspace is fully independent.
- `.env` files in templates can contain credentials the admin sets up once.
- `reap_expired()` is called by a background thread/timer, similar to the
  interactive shell reaper pattern.

### Phase 2: AppArmorManager

**New file: `jaato-server/server/apparmor.py`**

Manages AppArmor profiles for workspace confinement. Designed to be optional —
when AppArmor is not available, the system falls back to directory-level
sandboxing (existing behavior).

```python
class AppArmorManager:
    """Manages AppArmor profiles for per-session workspace confinement.

    When available, provides kernel-enforced filesystem isolation:
    - Each session gets a profile restricting access to its workspace
    - CLI and interactive shell commands are wrapped with aa-exec
    - Profiles are created on provision and removed on teardown

    When AppArmor is not available (non-Linux, not installed, or
    insufficient privileges), all methods are no-ops and
    ``is_available()`` returns False. Callers should check availability
    and fall back to directory-level sandboxing.

    Profile naming: ``jaato-ws-{session_id}``
    """

    PROFILE_TEMPLATE = '''
#include <tunables/global>

profile jaato-ws-{session_id} flags=(attach_disconnected) {{
  #include <abstractions/base>
  #include <abstractions/nameservice>
  #include <abstractions/python>

  # ---- workspace: read-write ----
  {workspace_path}/   rw,
  {workspace_path}/** rwkl,

  # ---- shared read-only resources ----
  {venv_path}/           r,
  {venv_path}/**         r,
  {venv_path}/bin/*      ix,

  # ---- temp files scoped to session ----
  /tmp/jaato-{session_id}-** rw,

  # ---- deny sibling workspaces ----
  deny {sessions_root}/ rw,
  deny {sessions_root}/** rw,

  # ---- basic system access ----
  /usr/bin/**          ix,
  /usr/local/bin/**    ix,
  /bin/**              ix,
  /usr/lib/**          rm,
  /lib/**              rm,
  /etc/ld.so.cache     r,
  /etc/passwd          r,
  /etc/nsswitch.conf   r,
  /proc/self/**        r,
  /dev/null            rw,
  /dev/urandom         r,
  /dev/pts/*           rw,

  # ---- network: outbound only ----
  network inet  stream,
  network inet6 stream,
  network inet  dgram,   # DNS
  network inet6 dgram,
  deny network raw,

  # ---- deny dangerous capabilities ----
  deny ptrace,
  deny mount,
  deny capability sys_admin,
  deny capability net_admin,
  deny capability sys_ptrace,
}}
'''

    def __init__(
        self,
        workspace_root: str,
        venv_path: Optional[str] = None,
        profile_dir: str = "/etc/apparmor.d/jaato",
    ):
        """Initialize AppArmor manager.

        Args:
            workspace_root: Root directory containing sessions/.
            venv_path: Path to Python venv (for read-only access).
            profile_dir: Directory to write profile files.
                Defaults to /etc/apparmor.d/jaato.
        """
        ...

    def is_available(self) -> bool:
        """Check if AppArmor is available and we can manage profiles.

        Returns True only when:
        - Running on Linux
        - apparmor_parser is on PATH
        - Profile directory is writable
        - /sys/kernel/security/apparmor exists
        """
        ...

    def provision_profile(
        self,
        session_id: str,
        workspace_path: str,
    ) -> bool:
        """Create and load an AppArmor profile for a session.

        Writes the profile file to {profile_dir}/jaato-ws-{session_id}
        and loads it with apparmor_parser -r.

        Returns True on success, False on failure (logged, not raised).
        """
        ...

    def teardown_profile(self, session_id: str) -> bool:
        """Unload and remove an AppArmor profile.

        Runs apparmor_parser -R to unload, then deletes the profile file.
        Returns True on success, False on failure.
        """
        ...

    def wrap_command(self, session_id: str, command: list) -> list:
        """Wrap a command to run under the session's AppArmor profile.

        Returns: ["aa-exec", "-p", "jaato-ws-{session_id}", "--"] + command

        If AppArmor is not available, returns the original command unchanged.
        """
        ...

    def get_profile_name(self, session_id: str) -> str:
        """Return the AppArmor profile name for a session."""
        return f"jaato-ws-{session_id}"
```

**Key decisions:**
- **Fail-open**: If AppArmor is unavailable, the system works exactly as before
  with directory-level sandboxing. `is_available()` is checked once at startup.
- **Profile directory**: `/etc/apparmor.d/jaato/` — requires the server process
  to have write access (or use a sudoers rule for `apparmor_parser`).
- **`ix` transitions**: Subprocess executions inherit the profile, so child
  processes (git, python, curl) are also confined.
- **Deny sibling workspaces**: The `deny {sessions_root}/**` rule with a more
  specific allow for the session's own directory ensures cross-session isolation.

### Phase 3: CLI Plugin Integration

**Modified: `jaato-server/shared/plugins/cli/plugin.py`**

The CLI plugin needs to know when it's running in a WS session with AppArmor
so it can wrap subprocess calls.

```python
# New method on CLIPlugin:
def set_apparmor_wrapper(
    self,
    wrapper: Optional[Callable[[list], list]],
) -> None:
    """Set an optional command wrapper for AppArmor confinement.

    When set, all subprocess executions will be passed through the
    wrapper function, which prepends aa-exec or similar confinement.
    When None, commands execute without additional confinement.

    Called by JaatoServer during session initialization when the
    session is created for a WebSocket client with AppArmor enabled.
    """
    self._apparmor_wrapper = wrapper
```

Changes to `_execute_command()`:
- Before `subprocess.Popen()`, if `self._apparmor_wrapper` is set, transform
  the command through it.
- For shell mode commands, the wrapper wraps the entire shell invocation:
  `aa-exec -p profile -- /bin/sh -c "command"`.
- For non-shell mode, the wrapper prefixes the argv list.

### Phase 4: Interactive Shell Plugin Integration

**Modified: `jaato-server/shared/plugins/interactive_shell/session.py`**

Same pattern — when an AppArmor wrapper is configured, `shell_spawn` wraps
the pexpect command.

```python
# In ShellSession.__init__ or spawn():
if self._apparmor_wrapper:
    command = self._apparmor_wrapper(command)
```

The `pexpect.spawn()` call already takes the command as a string or list,
so the wrapper just needs to prepend `aa-exec -p jaato-ws-{session_id} --`.

### Phase 5: WebSocket Server Integration

**Modified: `jaato-server/server/websocket.py`**

The WS server gains workspace provisioning on session creation.

Key changes:

1. **`__init__`**: Accept optional `apparmor` flag (default: auto-detect).

2. **`start()`**: Initialize `WorkspaceProvisioner` and `AppArmorManager`.
   Start workspace reaper background task.

3. **Per-client workspace tracking**: Replace single `_selected_workspace`
   with `_client_workspaces: Dict[client_id, ProvisionedWorkspace]`.

4. **New handler: `_handle_session_create`**: When a WS client creates a
   session:
   - Provision workspace via `WorkspaceProvisioner.provision()`
   - Create AppArmor profile via `AppArmorManager.provision_profile()`
   - Pass provisioned workspace path to `SessionManager.create_session()`
   - Wire `AppArmorManager.wrap_command()` into the session's CLI and
     interactive shell plugins

5. **Client disconnect**: On disconnect, if no sessions reference the
   workspace, schedule it for cleanup (with configurable grace period).

6. **Workspace reaper task**: Periodic asyncio task that calls
   `WorkspaceProvisioner.reap_expired()` and
   `AppArmorManager.teardown_profile()` for cleaned-up workspaces.

### Phase 6: Session Manager Integration

**Modified: `jaato-server/server/session_manager.py`**

Add awareness of provisioned workspaces:

1. **`create_session()`**: Accept optional `provisioned: bool` flag. When True,
   the workspace_path is server-managed and should not be overridden by
   client config.

2. **`_save_session()`**: Persist `provisioned` flag in session state so
   restored sessions know their workspace is server-managed.

3. **Session cleanup hooks**: When a session is deleted, notify the
   `WorkspaceProvisioner` and `AppArmorManager` to clean up.

### Phase 7: JaatoServer Integration

**Modified: `jaato-server/server/core.py`**

1. **`initialize()`**: After plugin wiring, if an `apparmor_wrapper` is
   provided, call `cli_plugin.set_apparmor_wrapper()` and
   `interactive_shell_plugin.set_apparmor_wrapper()`.

2. **New method: `set_apparmor_wrapper(wrapper)`**: Stores the wrapper and
   propagates it to relevant plugins.

### Phase 8: Workspace Reaper

**Integrated into `WorkspaceProvisioner`**

```python
class WorkspaceProvisioner:
    ...
    def start_reaper(
        self,
        interval_seconds: int = 3600,
        max_age_seconds: int = 86400,
        on_teardown: Optional[Callable[[str], None]] = None,
    ) -> threading.Thread:
        """Start background thread that periodically reaps expired workspaces.

        Args:
            interval_seconds: How often to check (default: hourly).
            max_age_seconds: Max age before reaping (default: 24h).
            on_teardown: Callback for each reaped session_id (for AppArmor
                cleanup, etc.).

        Returns the daemon thread (auto-stops on process exit).
        """
        ...
```

## New/Modified Files Summary

| File | Action | Description |
|------|--------|-------------|
| `server/workspace_provisioner.py` | **NEW** | Auto-provisioning, templates, reaper |
| `server/apparmor.py` | **NEW** | AppArmor profile management |
| `server/websocket.py` | MODIFY | Per-client workspaces, provisioning integration |
| `server/workspace_manager.py` | MODIFY | Per-client tracking, integration with provisioner |
| `server/session_manager.py` | MODIFY | Provisioned workspace awareness |
| `server/core.py` | MODIFY | AppArmor wrapper propagation to plugins |
| `shared/plugins/cli/plugin.py` | MODIFY | AppArmor command wrapping |
| `shared/plugins/interactive_shell/plugin.py` | MODIFY | AppArmor command wrapping |
| `shared/plugins/interactive_shell/session.py` | MODIFY | Wrap pexpect spawn command |
| `deploy/apparmor/` | **NEW** | Example profiles, setup script |

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

## Testing Strategy

1. **Unit tests**: `WorkspaceProvisioner` — provision, teardown, reap, templates
2. **Unit tests**: `AppArmorManager` — profile generation, `is_available()` mock
3. **Integration tests**: WS connect → provision → CLI command → verify
   confinement (requires AppArmor-enabled CI runner or mock)
4. **Fallback tests**: Verify graceful degradation when AppArmor unavailable

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
