# Implementation Plan: WebSocket Workspace Management & AppArmor Isolation

## Overview
Server-provisioned workspaces for WebSocket clients with kernel-enforced isolation via AppArmor. Remote clients get auto-created isolated directories; CLI/shell commands are confined to their workspace.

## Design Document
Full design: [docs/design/websocket-workspace-isolation.md](docs/design/websocket-workspace-isolation.md)

## Implementation Steps

### Step 1: WorkspaceProvisioner (`server/workspace_provisioner.py`) — NEW
- `ProvisionedWorkspace` dataclass (session_id, path, template, timestamps, client_id)
- `WorkspaceProvisioner` class:
  - `provision(session_id, client_id?, template?)` — creates `{root}/sessions/{session_id}/`, copies template
  - `teardown(session_id)` — removes workspace directory tree
  - `reap_expired(max_age_seconds)` — finds and removes stale workspaces
  - `get_workspace(session_id)`, `list_workspaces()`, `update_activity(session_id)`
  - `start_reaper(interval, max_age, on_teardown)` — daemon thread
- Template resolution: copies from `{root}/templates/{name}/` if exists
- Manifest file `{root}/sessions/manifest.json` for persistence across restarts
- Unit tests in `server/tests/test_workspace_provisioner.py`

### Step 2: AppArmorManager (`server/apparmor.py`) — NEW
- `AppArmorManager` class:
  - `is_available()` — checks Linux + apparmor_parser + writable profile dir + /sys/kernel/security/apparmor
  - `provision_profile(session_id, workspace_path)` — renders template, writes to profile dir, loads with `apparmor_parser -r`
  - `teardown_profile(session_id)` — unloads with `apparmor_parser -R`, deletes file
  - `wrap_command(session_id, command)` — returns `["aa-exec", "-p", profile_name, "--"] + command`
  - `get_profile_name(session_id)` — returns `jaato-ws-{session_id}`
- Profile template as class constant with format placeholders
- Graceful degradation: all methods are no-ops when `is_available()` is False
- Optional sudo support: if direct write fails, try via `sudo apparmor_parser`
- Unit tests in `server/tests/test_apparmor.py` (mocked subprocess calls)

### Step 3: CLI Plugin — AppArmor wrapper support
- Add `_apparmor_wrapper: Optional[Callable[[list], list]] = None` attribute
- Add `set_apparmor_wrapper(wrapper)` method
- In `_execute_command()`:
  - Non-shell mode: `cmd = self._apparmor_wrapper(cmd)` before `Popen()`
  - Shell mode: wrap as `aa-exec -p profile -- /bin/sh -c "original command"`
- No changes when wrapper is None (existing behavior preserved)

### Step 4: Interactive Shell Plugin — AppArmor wrapper support
- Add `_apparmor_wrapper` to `InteractiveShellPlugin`
- Add `set_apparmor_wrapper(wrapper)` method
- In `_execute_spawn()`: wrap the spawn command through the wrapper
- Pass wrapper to `ShellSession` constructor → applied in `_spawn_process()`

### Step 5: JaatoServer — wrapper propagation
- Add `set_apparmor_wrapper(wrapper)` method on `JaatoServer`
- Stores `_apparmor_wrapper` and propagates to CLI plugin and interactive_shell plugin
- Called by WS server after session creation for WebSocket-originated sessions

### Step 6: WebSocket Server Integration
- New init params: `apparmor` (bool/auto), `default_template`
- `start()`: initialize `WorkspaceProvisioner` + `AppArmorManager`, start reaper task
- Per-client workspace tracking: `_client_workspaces: Dict[client_id, ProvisionedWorkspace]`
- Enhanced `_handle_workspace_select()` / new `_handle_session_create()`:
  - Call `provisioner.provision(session_id, client_id, template)`
  - Call `apparmor.provision_profile(session_id, workspace_path)`
  - Create session with provisioned workspace_path
  - Wire `apparmor.wrap_command(session_id, ...)` into the session's server
- Client disconnect handler: schedule workspace cleanup (with grace period)
- Workspace reaper: async task calling `provisioner.reap_expired()` + `apparmor.teardown_profile()`

### Step 7: WorkspaceManager enhancements
- Replace single `_selected_workspace` with `_client_workspaces: Dict[client_id, WorkspaceInfo]`
- `select_workspace(name, client_id)` — per-client selection
- `get_selected_workspace(client_id)` — per-client lookup
- Integration point with `WorkspaceProvisioner` for auto-provisioned workspaces

### Step 8: Session Manager awareness
- `create_session()`: accept `provisioned: bool = False` flag
- When `provisioned=True`, workspace_path is server-managed, not overridable by client config
- Persist `provisioned` flag in session state metadata
- Session deletion hook: notify provisioner/apparmor for cleanup

### Step 9: Deployment artifacts
- `deploy/apparmor/README.md` — setup instructions
- `deploy/apparmor/setup.sh` — creates user, directories, sudoers rule
- Example template in `deploy/apparmor/templates/default/.env.example`

### Step 10: Tests
- `server/tests/test_workspace_provisioner.py` — provision, teardown, templates, reaper, manifest persistence
- `server/tests/test_apparmor.py` — profile generation, availability detection, command wrapping (mocked)
- `server/tests/test_websocket_workspace.py` — integration: WS connect → provision → verify workspace path in session
