# Runtime Limits (cgroup v2)

Jaato's WebSocket server can apply **per-session resource caps** — memory, PIDs, CPU share, plus app-layer wall-clock and output limits — to every subprocess a session launches. Enforced via Linux cgroup v2 for the kernel-level subset; the rest applied at the Python layer by the CLI / interactive_shell plugins.

This is the **runtime limits** axis, orthogonal to **sandboxing** (AppArmor):

| Axis | Question answered | Mechanism |
|------|-------------------|-----------|
| Sandboxing | What can this session **touch**? | AppArmor profile per session — see [apparmor-setup.md](apparmor-setup.md) |
| Runtime limits | How much can this session **consume**? | cgroup v2 + app-layer caps — this doc |

The two are independent. A session can have one, both, or neither.

## When to use

Runtime limits are for **multi-tenant deployments** where the jaato server accepts WebSocket connections from untrusted or semi-trusted clients and operators need predictable resource ceilings — preventing a single session from OOM-killing the host, fork-bombing the box, or starving siblings on CPU.

For local IPC usage (single user, jaato TUI), runtime limits are unnecessary — the user already controls their own machine.

## Prerequisites

- **Linux** with cgroup v2 mounted (the unified hierarchy at `/sys/fs/cgroup`)
- A parent cgroup directory the jaato server can write to
- The `memory`, `pids`, and `cpu` controllers delegated to that parent via `cgroup.subtree_control`

### Verify cgroup v2 is active

```bash
# v2 unified hierarchy present?
ls /sys/fs/cgroup/cgroup.controllers
# → If this file exists, cgroup v2 is mounted.

# Available controllers?
cat /sys/fs/cgroup/cgroup.controllers
# → memory, pids, cpu must appear in the list.
```

If you see `/sys/fs/cgroup/memory/`, `/sys/fs/cgroup/pids/`, etc. as separate directories, you're on **cgroup v1** and this feature won't work. Most modern distros (Ubuntu 22.04+, Debian 11+, Fedora 31+) default to v2.

## Operator setup

### 1. Create the parent cgroup directory

The server writes per-session cgroups under a parent directory (default `/sys/fs/cgroup/jaato`). Create it and grant write access to the jaato server user:

```bash
sudo mkdir /sys/fs/cgroup/jaato
sudo chown jaato:jaato /sys/fs/cgroup/jaato
```

If the server runs as your own user:

```bash
sudo mkdir /sys/fs/cgroup/jaato
sudo chown $USER /sys/fs/cgroup/jaato
```

### 2. Delegate controllers

cgroup v2 requires the parent's `cgroup.subtree_control` to enable each controller before its child cgroups can use them. This needs root **once at boot**:

```bash
echo "+memory +pids +cpu" | sudo tee /sys/fs/cgroup/jaato/cgroup.subtree_control
```

To make this persist across reboots, add a systemd drop-in or a `tmpfiles.d` entry. Example for systemd:

```ini
# /etc/systemd/system/jaato-cgroup.service
[Unit]
Description=Provision jaato cgroup parent
DefaultDependencies=no
After=sysinit.target
Before=basic.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/mkdir -p /sys/fs/cgroup/jaato
ExecStart=/bin/chown jaato:jaato /sys/fs/cgroup/jaato
ExecStart=/bin/sh -c 'echo "+memory +pids +cpu" > /sys/fs/cgroup/jaato/cgroup.subtree_control'

[Install]
WantedBy=basic.target
```

```bash
sudo systemctl enable --now jaato-cgroup.service
```

### Alternative: systemd user slice

If the server runs under a systemd user session, point the server at the user's delegated slice instead of `/sys/fs/cgroup/jaato`:

```bash
jaato-server \
  --web-socket :8089 \
  --cgroups-root "/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/jaato.slice"
```

You'll need to create that path with `loginctl enable-linger` and a user-level `tmpfiles.d` rule, but you avoid touching the system cgroup root.

### 3. Start the server

Auto-detect mode is the default — when the parent cgroup exists with controllers delegated, the feature activates:

```bash
jaato-server --web-socket :8089 --daemon
```

Explicit control is available too:

```bash
# Auto-detect (default)
python -m server.websocket --host 0.0.0.0 --port 8089 --workspace-root ~/.jaato/workspaces

# Explicitly enable — logs a warning if prerequisites are missing
python -m server.websocket --host 0.0.0.0 --port 8089 --workspace-root ~/.jaato/workspaces --cgroups

# Explicitly disable (app-layer caps still apply)
python -m server.websocket --host 0.0.0.0 --port 8089 --workspace-root ~/.jaato/workspaces --no-cgroups

# Custom cgroup root
python -m server.websocket --host 0.0.0.0 --port 8089 --workspace-root ~/.jaato/workspaces --cgroups-root /sys/fs/cgroup/my-jaato
```

Check the log:

```bash
grep -i "cgroup" /tmp/jaato.log
# → "Cgroups runtime limits enabled (root=/sys/fs/cgroup/jaato)"
# → "Cgroups runtime limits not available — falling back to no kernel limits"
```

## Profile usage

Add a `runtime_limits` field to any session profile (`.jaato/profiles/<name>.json`), parallel to `gc`:

```json
{
  "name": "build-and-test",
  "description": "Heavyweight session for compile + test cycles",
  "model": "claude-sonnet-4-20250514",
  "plugins": ["cli", "interactive_shell", "file_edit"],
  "runtime_limits": {
    "memory_max_mb": 4096,
    "pids_max": 1024,
    "cpu_weight": 200,
    "tool_timeout_seconds": 600,
    "max_output_bytes": 1048576
  }
}
```

```json
{
  "name": "code-review",
  "description": "Lightweight read-mostly session",
  "plugins": ["cli", "filesystem_query", "web_search"],
  "runtime_limits": {
    "memory_max_mb": 512,
    "pids_max": 128,
    "cpu_weight": 50,
    "tool_timeout_seconds": 60,
    "max_output_bytes": 262144
  }
}
```

### Field reference

| Field | Type | Layer | What it does |
|-------|------|-------|--------------|
| `memory_max_mb` | int (positive) | Kernel | Written to `memory.max`. Process tree gets OOM-killed if it exceeds this. |
| `pids_max` | int (positive) | Kernel | Written to `pids.max`. `fork()` returns EAGAIN beyond this count. |
| `cpu_weight` | int 1–10000 | Kernel | Written to `cpu.weight` (default 100). Relative scheduling weight against sibling cgroups. |
| `tool_timeout_seconds` | float (positive) | App | Wall-clock cap on each subprocess tool call. SIGTERM with 2s grace, then SIGKILL. |
| `max_output_bytes` | int (positive) | App | Override of the default stdout/stderr cap in CLI tool results. |

All fields are optional — set only the ones you want to enforce. Validation runs at profile load time, so typos and out-of-range values fail fast with a clear error rather than crashing mid-session.

### Inheritance

`runtime_limits` participates in profile inheritance with **scalar-override** semantics, same as `gc`. Multiple parents must agree (or the child must override) to avoid a conflict error.

```json
{
  "name": "strict-build",
  "inherits": ["build-and-test"],
  "runtime_limits": {
    "memory_max_mb": 8192,
    "pids_max": 1024,
    "cpu_weight": 200,
    "tool_timeout_seconds": 600,
    "max_output_bytes": 1048576
  }
}
```

## How it works

### Per-session cgroups

When a WS client creates a session with a profile that sets `runtime_limits`, the server:

1. Creates `/sys/fs/cgroup/jaato/jaato-ws-{session_id}/`
2. Writes `memory.max`, `pids.max`, `cpu.weight` from the profile
3. Plumbs an attach callback into the executor — every `subprocess.Popen` the session launches uses it as `preexec_fn`, so the forked child joins the cgroup between `fork()` and `exec()`
4. Plumbs an event-snapshot reader for OTel telemetry (see below)
5. On session teardown (workspace reaper or explicit close), kills any remaining processes via `cgroup.kill` (kernel ≥ 5.14) and removes the directory

### Subprocess-side enforcement

| Plugin | Path | Behaviour with limits set |
|--------|------|---------------------------|
| `cli` | foreground (`run_command`) | `preexec_fn` attaches child to cgroup; `tool_timeout_seconds` enforced via watchdog timer; `max_output_bytes` overrides the static cap. |
| `cli` | background (Popen+threads) | Same `preexec_fn` attach; `tool_timeout_seconds` via `proc.wait(timeout=...)` with SIGTERM-grace-SIGKILL; result dict gains `timed_out: true` + `timeout_seconds` keys. |
| `interactive_shell` | `shell_spawn` | `preexec_fn` passed to pexpect/popen_spawn so the PTY child joins the cgroup. `tool_timeout_seconds` is **not** applied per-call (PTY sessions are inherently long-lived; `max_lifetime` plugin config is the relevant ceiling). |

### Graceful degradation

If cgroup v2 is unavailable (cgroup v1 host, missing controllers, non-writable root, non-Linux), the server falls back to **app-layer-only** enforcement: `tool_timeout_seconds` and `max_output_bytes` still apply, but kernel-level caps are skipped.

The server logs which mode is active at startup:

```
Cgroups runtime limits enabled (root=/sys/fs/cgroup/jaato)
Cgroups runtime limits not available — falling back to no kernel limits (app-layer caps still apply)
```

## OTel telemetry

When the cgroup catches a runaway, the event surfaces as attributes on the tool span — no extra wiring needed. The executor snapshots `cgroup.events` before and after every tool call and injects positive deltas into the result's `_telemetry` dict, which the existing OTel forwarder lifts onto the span.

| Attribute | When it appears | Meaning |
|-----------|-----------------|---------|
| `jaato.cgroup.oom_kill_delta` | OOM-killer activated during this tool call | N processes inside the cgroup were killed by the OOM-killer |
| `jaato.cgroup.oom_delta` | Same, separate counter | N cgroup-level OOM events triggered |

Both only appear when the delta is non-zero — successful tool calls produce no extra attributes, keeping spans clean.

**Attribution caveat**: when multiple tool calls run concurrently in the same per-session cgroup (parallel tool execution), an OOM in tool A may also show up as a non-zero delta on a parallel tool B that straddled the event. Operators correlating spans with `dmesg | grep -i oom` can disambiguate.

## Troubleshooting

### "Cgroups runtime limits not available" at startup

The auto-detection found a missing prerequisite. Check each:

```bash
# 1. Is cgroup v2 mounted?
ls /sys/fs/cgroup/cgroup.controllers

# 2. Does the parent directory exist and is it writable?
ls -la /sys/fs/cgroup/jaato/

# 3. Are the required controllers delegated?
cat /sys/fs/cgroup/jaato/cgroup.subtree_control
# Must include: memory pids cpu

# 4. What did the server log?
grep -i "cgroup" /tmp/jaato.log
```

The log message after `Cgroups runtime limits not available` includes the specific reason (e.g. `controllers ['cpu'] not delegated`).

### Profile rejected at load time

If a profile fails to load with a `runtime_limits` error, the validation message is in the log:

```
Invalid runtime_limits in profile 'build-and-test': cpu_weight=99999 out of range [1, 10000]
```

The `validate_profile()` function delegates to `RuntimeLimits.from_dict()`, so all field-level errors from the dataclass surface here.

### Inspecting a live session's cgroup

```bash
# Find the session's cgroup
ls /sys/fs/cgroup/jaato/

# What processes are in it?
cat /sys/fs/cgroup/jaato/jaato-ws-20260425_120000/cgroup.procs

# Current memory usage vs limit
cat /sys/fs/cgroup/jaato/jaato-ws-20260425_120000/memory.current
cat /sys/fs/cgroup/jaato/jaato-ws-20260425_120000/memory.max

# OOM-kill events so far
cat /sys/fs/cgroup/jaato/jaato-ws-20260425_120000/cgroup.events
```

### Subprocesses not joining the cgroup

If `cgroup.procs` is empty even though processes are running:

- The plugin might not be using `preexec_fn` (check it implements `set_runtime_limits`).
- The cgroup might not have been provisioned (check `runtime_limits` is set on the profile and the server log shows `Cgroup runtime limits applied to session`).
- The kernel might be refusing the write — check `dmesg | tail` for cgroup-related errors.

### OOM kills not appearing in spans

The OTel attributes only show when an event actually fires during a tool call. If the model writes a `cat huge.bin` that fits within `memory.max`, no kill happens and no span attribute is set. To verify the wiring, force a kill:

```bash
# Set a tiny memory cap on a test profile and run something memory-hungry
# in a session attached to that profile. Then check both:
dmesg | grep -i oom
# AND the OTel exporter — the span for that tool call should have
# jaato.cgroup.oom_kill_delta > 0
```

## Composing with AppArmor

The two features stack cleanly:

- **Both enabled** — kernel-enforced filesystem allow-list (AppArmor) **plus** kernel-enforced consumption caps (cgroups). Strongest configuration; recommended for multi-tenant WS.
- **AppArmor only** — sessions can read/write only their workspace, but a runaway process can still consume host RAM/CPU.
- **Cgroups only** — sessions are bounded in resource use but can read/write any path the server user can. Useful when sandboxing is provided by another layer (e.g. running the server inside a container).
- **Neither** — single-user IPC default. Trust-by-transport.

Both features auto-detect, fall back gracefully, and use the same `--{feature}` / `--no-{feature}` tristate flags.

## See also

- [AppArmor Workspace Isolation](apparmor-setup.md) — the orthogonal sandboxing axis.
- [`server/cgroups.py`](../jaato-server/server/cgroups.py) — `RuntimeLimits` dataclass + `CgroupsManager` lifecycle.
- [`shared/runtime_limits.py`](../jaato-server/shared/runtime_limits.py) — the dataclass shared between server and subagent profile schema.
- [OpenTelemetry design](opentelemetry-design.md) — span hierarchy this hooks into.
