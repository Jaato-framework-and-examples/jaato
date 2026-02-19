"""Interactive shell plugin for driving user-interactive commands.

This plugin provides tools that let the model spawn persistent sessions
and interact with programs requiring back-and-forth input: REPLs, password
prompts, wizards, debuggers, SSH sessions, etc.

The model reads whatever the process outputs and decides what to type next —
no expect patterns needed. The intelligence is in the model, not the tool.

Platform backends (selected at import time in ``session.py``):

- **Unix / macOS**: ``pexpect.spawn`` — full PTY via ``pty.fork()``.
- **Windows (native)**: ``wexpect.spawn`` — Windows console APIs + named pipes.
- **MSYS2 / Git Bash**: ``pexpect.PopenSpawn`` — ``subprocess.Popen`` with piped
  stdin/stdout.  No real PTY (child ``isatty()`` returns ``False``, no terminal
  dimensions), but reliable timeout behaviour.  wexpect is avoided because its
  Windows console APIs hang with Cygwin-based MSYS2 executables.

See ``README.md`` in this directory for the full limitations table.
"""

from .plugin import InteractiveShellPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'InteractiveShellPlugin',
    'create_plugin',
    'PLUGIN_KIND',
]
