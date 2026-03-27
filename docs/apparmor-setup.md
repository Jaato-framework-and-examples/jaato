# AppArmor Workspace Isolation

Jaato's WebSocket server can confine each session to its own workspace directory using Linux AppArmor. When enabled, CLI commands and interactive shell sessions executed by the model cannot access files outside the session's workspace — enforced at the kernel level.

## When to use

AppArmor isolation is for **multi-tenant deployments** where the jaato server accepts WebSocket connections from untrusted or semi-trusted clients (dashboards, web components, remote teams). Each client's session gets a kernel-enforced sandbox.

For local IPC usage (single user, `jaato` TUI), AppArmor is unnecessary — the user already has full filesystem access.

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
