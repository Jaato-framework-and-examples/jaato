# Implementation Plan: WebSocket Workspace Management & AppArmor Isolation

## Overview
Server-provisioned workspaces for WebSocket clients with kernel-enforced isolation via AppArmor. Remote clients get auto-created isolated directories; CLI/shell commands are confined to their workspace.

## Design Document
Full design: [docs/design/websocket-workspace-isolation.md](docs/design/websocket-workspace-isolation.md)

## Implementation Steps

### Step 1: WorkspaceProvisioner (`server/workspace_provisioner.py`) — NEW ✅
- `ProvisionedWorkspace` dataclass (session_id, path, template, timestamps, client_id)
- `WorkspaceProvisioner` class:
  - `provision(session_id, client_id?, template?)` — creates `{root}/sessions/{session_id}/`, copies template
  - `teardown(session_id)` — removes workspace directory tree
  - `reap_expired(max_age_seconds)` — finds and removes stale workspaces
  - `get_workspace(session_id)`, `list_workspaces()`, `update_activity(session_id)`
  - `start_reaper(interval, max_age, on_teardown)` — daemon thread
- Template resolution: copies from `{root}/templates/{name}/` if exists
- Manifest file `{root}/sessions/manifest.json` for persistence across restarts
- Unit tests: 22 tests in `shared/tests/test_workspace_provisioner.py` — all passing

### Step 2: AppArmorManager (`server/apparmor.py`) — NEW ✅
- `AppArmorManager` class:
  - `is_available()` — checks Linux + apparmor_parser + writable profile dir + /sys/kernel/security/apparmor
  - `provision_profile(session_id, workspace_path)` — renders template, writes to profile dir, loads with `apparmor_parser -r`
  - `teardown_profile(session_id)` — unloads with `apparmor_parser -R`, deletes file
  - `wrap_command(session_id, command)` — returns `["aa-exec", "-p", profile_name, "--"] + command`
  - `wrap_shell_command(session_id, command)` — returns shell-escaped `aa-exec` invocation
  - `get_profile_name(session_id)` — returns `jaato-ws-{session_id}`
- Profile template as class constant with format placeholders
- Graceful degradation: all methods are no-ops when `is_available()` is False
- Unit tests: 22 tests in `shared/tests/test_apparmor.py` — all passing (mocked subprocess)

### Step 3: CLI Plugin — AppArmor wrapper support ✅
- Added `_apparmor_wrapper` and `_apparmor_shell_wrapper` attributes
- Added `set_apparmor_wrapper(argv_wrapper, shell_wrapper)` method
- Both `_execute()` and `_execute_streaming()` apply wrappers before `Popen()`
- No changes when wrappers are None (existing behavior preserved)

### Step 4: Interactive Shell Plugin — AppArmor wrapper support ✅
- Added `_apparmor_shell_wrapper` to `InteractiveShellPlugin`
- Added `set_apparmor_wrapper(shell_wrapper)` method
- Wraps spawn command through wrapper before `ShellSession()` creation

### Step 5: JaatoServer — wrapper propagation ✅
- Added `set_apparmor_wrapper(argv_wrapper, shell_wrapper)` method on `JaatoServer`
- Propagates to CLI plugin (both wrappers) and interactive_shell plugin (shell wrapper)
- Uses `registry.get_plugin()` with `hasattr` guard for safety

### Step 6: WebSocket Server Integration ✅
- New init params: `apparmor` (bool/auto), `default_template`, `workspace_max_age`
- `start()`: initializes `WorkspaceProvisioner` + `AppArmorManager`, starts reaper with on_teardown callback
- Per-client workspace tracking via `_client_provisioned: Dict[client_id, str]`
- `provision_workspace()`: provisions via provisioner, sets up AppArmor profile
- `get_apparmor_wrappers()`: returns (argv_wrapper, shell_wrapper) closures
- **Integration wiring**: `_handle_config_update()` calls `_initialize_server_for_workspace()` which:
  - Auto-provisions workspace if provisioner is configured
  - Creates `JaatoServer` with correct `env_file` and `workspace_path`
  - Initializes server in executor
  - Applies AppArmor wrappers via `server.set_apparmor_wrapper()`
- Client disconnect: calls `_workspace_manager.remove_client()`, pops `_client_provisioned`
- CLI flags: `--apparmor/--no-apparmor`, `--workspace-template`, `--workspace-max-age`
- `get_server_info()`: includes provisioned_workspaces count, available_templates, apparmor_available

### Step 7: WorkspaceManager enhancements ✅
- Added `_client_workspaces: Dict[str, str]` for per-client tracking
- `select_workspace(name, client_id)` — per-client selection with legacy fallback
- `get_selected_workspace(client_id)` — per-client lookup with legacy fallback
- Added `remove_client(client_id)` method

### Step 8: Session Manager awareness ✅
- `Session` dataclass: added `provisioned: bool = False`
- `create_session()`: accepts `provisioned: bool = False` param
- `_save_session()`: persists provisioned flag in `metadata['provisioned']`
- `_load_session()`: restores provisioned flag from metadata

### Step 9: Deployment artifacts — DEFERRED
- `deploy/apparmor/README.md` — setup instructions
- `deploy/apparmor/setup.sh` — creates user, directories, sudoers rule
- Example template in `deploy/apparmor/templates/default/.env.example`

### Step 10: Tests — PARTIAL ✅
- ✅ `shared/tests/test_workspace_provisioner.py` — 22 tests (provision, teardown, templates, reaper, manifest persistence)
- ✅ `shared/tests/test_apparmor.py` — 22 tests (profile generation, availability detection, command/shell wrapping, provision/teardown mocked)
- ⬚ `shared/tests/test_websocket_workspace.py` — integration test (WS connect → provision → verify workspace path) — DEFERRED

## Remaining Work

### Deferred Items
1. **Deployment artifacts** (Step 9) — `deploy/apparmor/` directory with setup scripts, sudoers template, and example workspace templates. Needed for production deployment but not for the core feature.
2. **WebSocket integration test** (Step 10) — End-to-end test covering WS connect → config update → workspace provisioning → AppArmor wrapping → message flow. Requires mocking the full WS server lifecycle.
3. **Session deletion hook** — When a session is deleted via session manager, notify provisioner/apparmor for cleanup. Currently cleanup only happens via reaper or client disconnect.
