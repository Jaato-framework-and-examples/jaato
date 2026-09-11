# Interactive Shell Plugin

The interactive shell plugin lets the model spawn persistent sessions and drive any user-interactive command by reading output and sending input.

## Overview

Unlike the `cli` plugin (which uses `subprocess` for non-interactive commands), this plugin provides a real pseudo-terminal where the model can read output and send input back and forth. It uses idle-based output detection — it reads until the process stops producing output (~500ms of silence), then returns whatever appeared.

## Tools

All tools have `discoverability="discoverable"` (on-demand loading).

| Tool | Purpose |
|------|---------|
| `shell_spawn` | Start a new interactive process. Returns `session_id` + initial output. |
| `shell_input` | Send text to an existing session (by `session_id`). |
| `shell_read` | Read pending output without sending input. |
| `shell_control` | Send control keys: `c-c`, `c-d`, `c-z`, `c-l`. |
| `shell_close` | Terminate a session. Returns exit status. |
| `shell_list` | List all active sessions with status, command, and age. |

## Platform Backends

The plugin selects a backend at import time based on the platform:

| Platform | Backend | Method | PTY |
|----------|---------|--------|-----|
| Unix / macOS | `pexpect.spawn` | PTY via `pty.fork()` | Full |
| Windows (native) | `wexpect.spawn` | Windows console APIs + named pipes | Console |
| MSYS2 / Git Bash | `pexpect.PopenSpawn` | `subprocess.Popen` with piped I/O | None |

The active backend is stored in `session._BACKEND` (`'pexpect'`, `'popen_spawn'`, or `'wexpect'`).

### MSYS2 Backend Selection

When running under MSYS2 (detected via `shared.path_utils.is_msys2_environment()`), the plugin tries backends in priority order:

1. **`pexpect.spawn`** — Full PTY support. Only works with MSYS Python (which provides the `pty` module). MINGW Python does not have `pty`.
2. **`pexpect.PopenSpawn`** — Subprocess pipes with a background reader thread and `Queue`-based timeout. Works with MINGW Python. This is the typical MSYS2 path.
3. **`wexpect`** — Last resort. May hang with Cygwin-based MSYS2 executables (see below).

#### Why wexpect hangs on MSYS2

wexpect spawns processes via Windows console APIs and communicates through named pipes. Cygwin-based MSYS2 executables (bash, python3, etc.) don't write to those pipes in the expected way. The `read_nonblocking()` call does a blocking `win32file.ReadFile` that never returns, causing `shell_spawn` to hang indefinitely.

### PopenSpawn Limitations

When the `popen_spawn` backend is active (typical on MSYS2 with MINGW Python), there are known limitations compared to a full PTY:

| Capability | PTY (pexpect/wexpect) | PopenSpawn |
|---|---|---|
| Terminal detection (`isatty()`) | `True` | **`False`** — child sees piped stdin/stdout |
| Terminal dimensions (rows/cols) | Respected | **Not available** — `rows`/`cols` params are ignored |
| Password prompt hiding | Input is hidden | **Input is visible** — no terminal echo control |
| Programs that require a TTY | Work normally | **May skip interactive prompts** or behave differently |
| Timeout / idle detection | Works | Works (via background reader thread + Queue) |
| Process lifecycle (spawn, send, close) | Works | Works |

**Practical impact:**
- REPLs (python, node, psql) work — they have their own prompt logic that handles piped I/O.
- Debuggers (gdb, pdb) work for basic usage.
- Password prompts (ssh, sudo) may not hide input and some programs may refuse to prompt at all when `isatty()` returns `False`.
- Programs that call `isatty()` and change behavior (e.g., colored output, progress bars) will see a non-TTY and may produce plainer output.

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_sessions` | int | `8` | Maximum concurrent sessions |
| `max_lifetime` | float | `600` | Session lifetime ceiling in seconds |
| `max_idle` | float | `300` | Max idle seconds before reaping |
| `idle_timeout` | float | `0.5` | Seconds of silence for output settling |
| `workspace_root` | str | `None` | Working directory for spawned processes, and the containment boundary (see below). `None` = no sandboxing |
| `require_confinement` | bool | `false` | Refuse every `shell_spawn` when no AppArmor child profile is active |
| `scrub_secret_env` | str/list | `default` | Secret env-var globs stripped from every spawn ([Secret Env Scrubbing](../../../../CLAUDE.md)) |
| `agent_name` | str | `None` | Agent context for trace logging |

## Path Containment (#722, #503)

This plugin spawns a real PTY, so it is strictly more capable than `cli` —
and until #722 it enforced nothing, while `cli` refused every path outside
the workspace. `shell_spawn("cat /etc/hostname")` therefore read a file
`cli` had just declined. Three layers answer that, and they are
deliberately not equal:

| Layer | Covers | On a string it cannot parse |
|-------|--------|-----------------------------|
| `shell_spawn`'s `command` | the workspace check `cli` applies, same analyzer (`shared/plugins/command_containment.py`) | **refuses** — the spawn command IS a shell command |
| `shell_input`'s `text` | the same check, best effort | **allows** — typed text may be Python, SQL or a password; refusing every non-shell string would refuse most legitimate input |
| AppArmor child profile | everything, kernel-enforced | n/a |

Only the third is a boundary in the strict sense: a live PTY's working
directory drifts under `cd`, and a program inside it can name paths
through channels no string check sees. The string layers refuse the
direct attempt and claim nothing more — concretely, the analyzer does not
descend into the quoted argument of `sh -c`, so `sh -c 'cat /etc/shadow'`
is not caught (a gap shared with `cli`, whose analyzer this is).

When the runner installed no AppArmor child-profile transition the plugin
**says so once per session at WARNING** rather than running unconfined in
silence, which was the pre-#722 behaviour. `require_confinement: true`
takes the stronger posture and refuses to spawn at all — the same shape as
the `notebook` plugin's in-process-exec gate. The default is `false`
because a PTY child is a separate process, the same risk class as a `cli`
subprocess, which does not fail closed either.

The spawn `cwd` is the workspace root, is not model-controllable, and is
verified against `workspace_root` inside `ShellSession` before the process
starts (the `cwd` half of #503).

## Session Lifecycle

Sessions have configurable max lifetime (default 600s) and max idle time (default 300s). A background reaper thread periodically closes expired sessions.

```
shell_spawn(command="python3")
  → session_id="session_0", output="Python 3.11.0\n>>> "

shell_input(session_id="session_0", input="print('hello')\n")
  → output="hello\n>>> "

shell_close(session_id="session_0")
  → exit_status=0
```

## Architecture

```
plugin.py          InteractiveShellPlugin — tool schemas, executors, reaper thread
  │
  ├── session.py   ShellSession — wraps pexpect/wexpect/PopenSpawn with idle detection
  │     │
  │     └── Backend selection (module-level):
  │           IS_MSYS2 → pexpect.spawn > PopenSpawn > wexpect
  │           IS_WINDOWS → wexpect
  │           else → pexpect.spawn
  │
  └── ansi.py      ANSI escape sequence stripping for clean model output
```

## Trace Logging

The plugin writes trace messages to `JAATO_TRACE_LOG` (env var) or `<tempdir>/rich_client_trace.log`. Traces include the active backend and MSYS2 detection status:

```
[14:32:01.123] [InteractiveShell] initialize: ..., backend=popen_spawn, msys2=True
[14:32:05.456] [InteractiveShell] spawn: id=session_0, cmd=bash --norc, backend=popen_spawn
```

On MSYS2 with MINGW Python, `tempfile.gettempdir()` typically resolves to `C:\Users\<user>\AppData\Local\Temp` (Windows path), not `/tmp`. Set `JAATO_TRACE_LOG` explicitly if the default location is hard to find.

## Dependencies

| Package | Platform | Purpose |
|---------|----------|---------|
| `pexpect>=4.8.0` | All | PTY sessions (Unix), PopenSpawn (MSYS2) |
| `wexpect>=4.0.0` | Windows | Windows console sessions (native Windows) |
| `pywin32>=220` | Windows | wexpect dependency (Windows APIs) |
| `psutil>=5.0.0` | Windows | wexpect dependency (process management) |
| `setuptools` | Windows | wexpect dependency (`pkg_resources`) |
