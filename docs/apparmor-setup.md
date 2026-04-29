# AppArmor Workspace Isolation

Jaato's WebSocket server can confine each session to its own workspace directory using Linux AppArmor. When enabled, CLI commands and interactive shell sessions executed by the model cannot access files outside the session's workspace — enforced at the kernel level.

## When to use

AppArmor isolation is for **multi-tenant deployments** where the jaato server accepts WebSocket connections from untrusted or semi-trusted clients (dashboards, web components, remote teams). Each client's session gets a kernel-enforced sandbox.

For local IPC usage (single user, `jaato` TUI), AppArmor is unnecessary by default — the user already has full filesystem access.

For **orchestrator-driven IPC harnesses** where the agent itself is the threat surface (LLM-driven tool calls in a sandbox directory, model-generated paths that need kernel enforcement), AppArmor is opt-in via the SDK:

```python
from jaato_sdk.client import IPCClient

client = IPCClient(
    socket_path="/tmp/jaato.sock",
    workspace_path="/path/to/sandbox",       # rw inside the profile
    config_root="/path/to/project/.jaato",   # readonly inside the profile
    apparmor=True,                           # opt-in confinement
)
```

When `apparmor=True`, the daemon provisions a per-session AppArmor profile (same machinery as the WS path). The agent's tool plugins (`cli`, `file_edit`, `interactive_shell`, etc.) can read / write inside `workspace_path`, read inside `config_root` and `~/.jaato/`, but cannot escape to arbitrary filesystem locations.

Default remains `False` so the long-standing IPC behavior (sessions run unconfined for the local user's TUI) is unchanged.

When AppArmor is unavailable on the host (non-Linux, kernel module not loaded, `apparmor_parser` missing) the session falls back to running unconfined — but **not silently**. The daemon always emits a `SystemMessageEvent` to the client describing the outcome:

- `[apparmor] confinement applied (workspace=..., config_root=...)` (style `"info"`) when enforcement is in effect.
- `[apparmor] requested but AppArmor is unavailable on this host (...) — running unconfined` (style `"warning"`) when it isn't.
- `[apparmor] profile provisioning failed (see daemon log) — running unconfined` (style `"warning"`) when provisioning fails.

Surface these in your IPC client's event loop so the user can see at a glance whether kernel confinement is really active, instead of having to tail `/tmp/jaato.log`.

## Prerequisites

- **Linux** with AppArmor kernel module loaded
- `apparmor_parser` and `aa-exec` on `PATH` (usually from `apparmor-utils`)
- A writable profile directory (default: `/etc/apparmor.d/jaato`)

### Install on Ubuntu/Debian

```bash
sudo apt install apparmor apparmor-utils
```

### Verify AppArmor is active

```bash
# Kernel module loaded?
ls /sys/kernel/security/apparmor
# → If this directory exists, AppArmor is active

# Tools available?
which apparmor_parser aa-exec
```

## Server setup

### 1. Create the profile directory

The server writes per-session AppArmor profiles to `/etc/apparmor.d/jaato/`. Create it and grant write access to the jaato server user:

```bash
sudo mkdir -p /etc/apparmor.d/jaato
sudo chown jaato:jaato /etc/apparmor.d/jaato
```

If the server runs as your own user:

```bash
sudo mkdir -p /etc/apparmor.d/jaato
sudo chown $USER /etc/apparmor.d/jaato
```

### 2. Grant apparmor_parser permissions

The server needs to load and unload profiles without sudo. Add a sudoers rule:

```bash
# /etc/sudoers.d/jaato-apparmor
jaato ALL=(root) NOPASSWD: /sbin/apparmor_parser
```

Or, if using a non-root user that can write to the profile directory and `apparmor_parser` runs as root, configure the server to use sudo:

```bash
# Alternative: the server calls apparmor_parser directly
# if the profile directory is writable and the process has
# CAP_MAC_ADMIN capability (rare outside containers).
```

In practice, most deployments write profiles as the jaato user and load them via a helper script with sudo.

> **Note:** The server uses `--cache-loc ~/.jaato/apparmor-cache/` when invoking `apparmor_parser`, so the system-level cache at `/var/cache/apparmor` (which requires root) is not needed.

### 3. Start the server

Via `jaato-server`, AppArmor confinement activates automatically when the server starts with a WebSocket listener and all prerequisites are met — no extra flags are needed:

```bash
jaato-server --web-socket :8089 --daemon
```

If you run the WebSocket server standalone, you can explicitly control AppArmor:

```bash
# Auto-detect (default)
python -m server.websocket --host 0.0.0.0 --port 8089 --workspace-root ~/.jaato/workspaces

# Explicitly enable — logs a warning if prerequisites are missing
python -m server.websocket --host 0.0.0.0 --port 8089 --workspace-root ~/.jaato/workspaces --apparmor

# Explicitly disable
python -m server.websocket --host 0.0.0.0 --port 8089 --workspace-root ~/.jaato/workspaces --no-apparmor
```

Check the log to confirm which mode is active:

```bash
grep -i apparmor /tmp/jaato.log
# → "AppArmor confinement enabled" or "AppArmor confinement not available"
```

## How it works

### Per-session profiles

When a WS client creates a session, the server:

1. Provisions an isolated workspace directory under `~/.jaato/workspaces/sessions/{session_id}/`
2. Generates an AppArmor profile named `jaato-ws-{session_id}`
3. Loads the profile via `apparmor_parser -r`
4. Wraps all CLI and interactive shell commands with `aa-exec -p jaato-ws-{session_id}`

### What the profile allows

| Resource | Access |
|----------|--------|
| Session workspace (`~/.jaato/workspaces/sessions/{id}/`) | Read-write |
| Python venv (server's `sys.prefix`) | Read-only |
| Session temp files (`/tmp/jaato-{id}-*`) | Read-write |
| System binaries (`/usr/bin/`, `/bin/`) | Execute (inherit) |
| Network (TCP/UDP outbound) | Allowed |

### What the profile denies

| Resource | Denied |
|----------|--------|
| Other sessions' workspaces | Read and write |
| Raw sockets | All |
| ptrace (debugging other processes) | All |
| mount/umount | All |
| `CAP_SYS_ADMIN`, `CAP_NET_ADMIN` | All |

### Lifecycle

- **Created**: When `session.new` provisions a workspace
- **Active**: For the duration of the session
- **Removed**: When the workspace reaper cleans up expired sessions (default: 24h), or when the session is explicitly closed

## Graceful degradation

If AppArmor is unavailable (non-Linux, tools not installed, permissions missing), the server falls back to **directory-level sandboxing** — the CLI plugin restricts paths to the workspace directory via application-level checks. This provides defense-in-depth but is not kernel-enforced.

The server logs which mode is active at startup:

```
AppArmor confinement enabled          → kernel-enforced isolation
AppArmor confinement not available    → directory sandboxing only
```

## Extension fragments

Daemon extensions (e.g. `jaato_premium.reactors`) sometimes need additional grants for state files they own — `~/.jaato/handoff_gates.json` for the reactor's HandoffGate registry, future cluster-state files, etc. Patching those paths into the public profile template would leak extension-specific knowledge into the framework, so the contract is **fragments**:

The profile template ends with:

```
include if exists "~/.jaato/apparmor-fragments/*.rules"
```

Each extension drops a `*.rules` file at startup (or installation time) into `~/.jaato/apparmor-fragments/`. The `apparmor_parser` splices every file there into every confined session.

**Example** — the premium reactor's fragment grants its handoff-gates state file:

```
# ~/.jaato/apparmor-fragments/premium-reactor.rules
@{HOME}/.jaato/handoff_gates.json     rwk,
@{HOME}/.jaato/.handoff_gates.*.tmp   rwk,
```

**Conventions**:

- Filename pattern: `<extension-name>.rules`. One file per extension keeps cleanup obvious.
- Only put rules inside (no `profile { ... }` wrapper — these are spliced into the existing profile).
- Use `@{HOME}` for the user's home directory; AppArmor expands it at parse time.
- Empty / missing dir is a no-op thanks to `if exists`, so unconfined deployments and extension-less builds keep working.

When adding a new fragment, restart any running sessions for the new rules to take effect — `apparmor_parser` reads the include at profile-load time, not on every transition.

## Troubleshooting

### "AppArmor confinement not available" / "required but not available"

The first message appears when auto-detection finds missing prerequisites. The second appears only when `--apparmor` is passed explicitly to the standalone WS server. In both cases, check:

```bash
# Is the kernel module loaded?
cat /sys/kernel/security/apparmor/profiles | head -5

# Are tools installed?
dpkg -l | grep apparmor-utils

# Is the profile directory writable?
ls -la /etc/apparmor.d/jaato/
```

### Profile load failures

If a session fails to start with AppArmor errors:

```bash
# Check the profile syntax
sudo apparmor_parser -p /etc/apparmor.d/jaato/jaato-ws-{session_id}

# Check kernel log for denials
dmesg | grep apparmor | tail -20

# Or use aa-status
sudo aa-status | grep jaato
```

### Commands blocked unexpectedly

AppArmor denials appear in the kernel log:

```bash
dmesg | grep "apparmor=\"DENIED\""
```

Common causes:
- Tool needs access to a path not in the profile (add it to the template in `server/apparmor.py`)
- Python package outside the venv path (install in the server's venv)

## Docker / container considerations

AppArmor inside containers requires the host to allow nested profiles. Most container runtimes (Docker, Podman) support this with `--security-opt apparmor=unconfined` on the container, then loading profiles inside.

For Kubernetes, use the AppArmor annotations on the pod spec. The jaato server's profile directory must be writable inside the container.

If running in a container without AppArmor support, the server auto-detects and falls back to directory sandboxing.
