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

There are three supported paths, in decreasing order of preference. Start at (1) unless you know you need (2) or (3).

### 1. Recommended: run jaato-server as a systemd unit with `Delegate=yes`

On any systemd-managed host this is the only setup that doesn't fight systemd. systemd creates the cgroup, chowns the three delegate files (`cgroup.procs`, `cgroup.subtree_control`, `cgroup.threads`) to the service user atomically at unit start, enables the controllers, and rebuilds the tree across reboots and `daemon-reload`s. No manual `chown` dance, no controllers-not-available surprises.

Create the unit:

```ini
# /etc/systemd/system/jaato-server.service
[Unit]
Description=jaato-server
After=network.target

[Service]
Type=simple
User=jaato
Group=jaato
ExecStart=/usr/local/bin/jaato-server --web-socket :8089
Restart=on-failure

# This line is what makes per-session cgroups work without root:
# systemd hands the service its own cgroup with cpu/memory/pids
# delegated to the User= account.
Delegate=yes
# Equivalently, be explicit about which controllers to delegate:
# Delegate=cpu memory pids

[Install]
WantedBy=multi-user.target
```

Enable and verify:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now jaato-server
systemd-cgls /sys/fs/cgroup/system.slice/jaato-server.service
ls -la /sys/fs/cgroup/system.slice/jaato-server.service/
# → directory + cgroup.procs + cgroup.subtree_control + cgroup.threads
#   should all be owned by jaato:jaato
cat /sys/fs/cgroup/system.slice/jaato-server.service/cgroup.controllers
# → must include: cpu memory pids
```

Then point jaato at the delegated cgroup:

```bash
sudo systemctl edit jaato-server
# Add an Environment= or extend ExecStart=:
#   ExecStart=/usr/local/bin/jaato-server --web-socket :8089 \
#       --cgroups-root /sys/fs/cgroup/system.slice/jaato-server.service
```

The default `--cgroups-root` (`/sys/fs/cgroup/jaato`) is the manual-fallback path; with `Delegate=yes` the systemd-managed path is the right target.

### 2. Developer / interactive testing under your own user

When you're not running jaato as a system service — e.g. ad-hoc testing as your normal login uid — the cgroup that systemd delegates to you isn't `user-$UID.slice` (that one is owned by root by design); it's one level deeper at:

```
/sys/fs/cgroup/user.slice/user-$UID.slice/user@$UID.service/
```

`user@$UID.service` is a separate unit with `Delegate=yes` baked in by systemd. It's started for you when `loginctl enable-linger` is set (so it survives between SSH sessions). Verify ownership:

```bash
loginctl enable-linger $USER   # one-time, persists across reboots
ls -la /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/
# → cgroup.procs / cgroup.subtree_control / cgroup.threads owned by you
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers
# → cpu memory pids
```

Point jaato at a sub-slice you create inside it:

```bash
USER_CG="/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service"
mkdir "$USER_CG/jaato"
# Enable controllers in the user@.service's subtree_control so the
# child cgroup we just made actually has cpu/memory/pids available.
echo "+cpu +memory +pids" > "$USER_CG/cgroup.subtree_control"
jaato-server --web-socket :8089 --cgroups-root "$USER_CG/jaato"
```

Common mistake: writing into `user-$UID.slice/` directly. That directory is root:root — by systemd design — and your `mkdir`/`chown` will fail with `EACCES`. The delegated subtree starts one level deeper at `user@$UID.service/`.

### 3. Manual fallback: top-level `/sys/fs/cgroup/jaato` outside systemd's tree

Only use this when (1) and (2) aren't an option (e.g. running outside systemd, or in a tightly controlled container init). cgroup v2 technically allows it, but on a systemd host the daemon may log warnings about an unmanaged cgroup and a `systemctl daemon-reload` may rebuild parts of the tree.

The minimum recipe — note the four `chown` lines, not just one:

```bash
sudo mkdir /sys/fs/cgroup/jaato
sudo chown jaato:jaato /sys/fs/cgroup/jaato
# Without these three the user can't actually populate or configure
# the cgroup — chowning just the directory is insufficient.
sudo chown jaato:jaato /sys/fs/cgroup/jaato/cgroup.procs \
                      /sys/fs/cgroup/jaato/cgroup.subtree_control \
                      /sys/fs/cgroup/jaato/cgroup.threads

# A controller is only available *inside* a cgroup if its parent has
# enabled it in subtree_control.  For /sys/fs/cgroup/jaato/ to expose
# cpu/memory/pids, the *root* cgroup must list them in its
# subtree_control.  On systemd hosts these are usually already there
# because systemd needs them; print and only add what's missing.
cat /sys/fs/cgroup/cgroup.subtree_control
echo "+cpu +memory +pids" | sudo tee /sys/fs/cgroup/cgroup.subtree_control
```

To persist this across reboots, prefer a systemd oneshot over `tmpfiles.d` so the four chowns and the `subtree_control` write happen in order:

```ini
# /etc/systemd/system/jaato-cgroup.service — manual-fallback only
[Unit]
Description=Provision jaato cgroup parent (manual fallback)
DefaultDependencies=no
After=sysinit.target
Before=basic.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/mkdir -p /sys/fs/cgroup/jaato
ExecStart=/bin/chown jaato:jaato /sys/fs/cgroup/jaato
ExecStart=/bin/chown jaato:jaato /sys/fs/cgroup/jaato/cgroup.procs /sys/fs/cgroup/jaato/cgroup.subtree_control /sys/fs/cgroup/jaato/cgroup.threads
ExecStart=/bin/sh -c 'echo "+memory +pids +cpu" > /sys/fs/cgroup/cgroup.subtree_control'

[Install]
WantedBy=basic.target
```

```bash
sudo systemctl enable --now jaato-cgroup.service
```

### What's NOT a knob

A few things people reach for that don't help:

- **Kernel boot parameters** (`cgroup_no_v1=all`, `systemd.unified_cgroup_hierarchy=1`) — these are about cgroup v1 vs v2 generally, not delegation. Modern Ubuntu/Fedora/Debian default to v2 already.
- **PAM modules** — PAM doesn't gate cgroup access.
- **`chmod 777` on a cgroupfs file** — should work in principle (cgroupfs honours mode bits), but if you found it didn't actually change the mode, you were almost certainly not root (running inside a container, user namespace, or WSL where the cgroup root is read-only). Verify with `stat`, not `ls`, immediately after.

The only knobs that matter are: (a) `loginctl enable-linger` for the user-session path, (b) `Delegate=` on the unit for the system-service path, (c) the manual chown-the-directory-plus-three-files dance for the fallback.

### 4. Start the server

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

The auto-detection found a missing prerequisite. Substitute `$ROOT` with whichever cgroups-root applies to your setup (`/sys/fs/cgroup/system.slice/jaato-server.service` for the recommended path, `/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/jaato` for interactive testing, `/sys/fs/cgroup/jaato` for the manual fallback). Check each:

```bash
ROOT=/sys/fs/cgroup/system.slice/jaato-server.service   # adjust for your setup

# 1. Is cgroup v2 mounted?
ls /sys/fs/cgroup/cgroup.controllers

# 2. Does the cgroups-root exist and are the three delegate files
#    owned by the server user (not root)?
ls -la "$ROOT/cgroup.procs" "$ROOT/cgroup.subtree_control" "$ROOT/cgroup.threads"

# 3. Are the required controllers available *inside* the cgroups-root?
#    (Driven by the PARENT's subtree_control — cgroup.controllers
#    here lists what the parent has actually delegated down.)
cat "$ROOT/cgroup.controllers"
# Must include: cpu memory pids

# 4. What did the server log?
grep -i "cgroup" /tmp/jaato.log
```

The log message after `Cgroups runtime limits not available` includes the specific reason (e.g. `controllers ['cpu'] not delegated`). The two most common causes:

- **Delegate files still root-owned** — for the manual-fallback path, you chowned the directory but forgot the three files; re-run the chown for `cgroup.procs` / `cgroup.subtree_control` / `cgroup.threads`. With `Delegate=yes` this should never happen — if it does, your unit didn't actually take effect (`systemctl status jaato-server` for the cgroup line).
- **Controllers missing from `cgroup.controllers`** — the controller is enabled in the *parent's* `subtree_control`, not the cgroup's own. For the manual-fallback path, that means writing to `/sys/fs/cgroup/cgroup.subtree_control` (the root cgroup), not `/sys/fs/cgroup/jaato/cgroup.subtree_control`.

### Profile rejected at load time

If a profile fails to load with a `runtime_limits` error, the validation message is in the log:

```
Invalid runtime_limits in profile 'build-and-test': cpu_weight=99999 out of range [1, 10000]
```

The `validate_profile()` function delegates to `RuntimeLimits.from_dict()`, so all field-level errors from the dataclass surface here.

### Inspecting a live session's cgroup

`$ROOT` is whichever path you set in `--cgroups-root` (or the auto-detected default — see the operator-setup section above for which root applies to your install).

```bash
ROOT=/sys/fs/cgroup/system.slice/jaato-server.service   # adjust as needed
SESSION=jaato-ws-20260425_120000                         # from the session log

# What processes are in it?
cat "$ROOT/$SESSION/cgroup.procs"

# Current memory usage vs limit
cat "$ROOT/$SESSION/memory.current"
cat "$ROOT/$SESSION/memory.max"

# OOM-kill events so far
cat "$ROOT/$SESSION/cgroup.events"
```

`systemd-cgls` is also handy for the recommended path:

```bash
systemd-cgls /sys/fs/cgroup/system.slice/jaato-server.service
# → tree view of the daemon's cgroup with every per-session child
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
