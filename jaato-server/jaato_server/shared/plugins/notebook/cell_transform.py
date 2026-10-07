"""IPython-style ``!shell`` line support for notebook cells.

The notebook kernels exec cells as plain Python (``ast.parse`` + ``exec``), so a
bare ``!cmd`` line — valid in IPython/Jupyter — is a ``SyntaxError`` here.  This
module rewrites each line matching ``!<cmd>`` into a call to a runtime helper
that runs ``<cmd>`` through the shell and streams its output into the cell,
matching IPython's ``!`` shell escape **per-line** inside multi-line cells.

The helper runs in the kernel's own process, inheriting its environment — so
under the subprocess kernel (which is launched with the workspace venv
activated) ``!pip install X`` targets the tool-venv, exactly like a plain
``pip`` command would.

Scope / limitations:
- ``!cmd`` execution only.  IPython's ``x = !cmd`` output-capture form and
  ``%magic`` lines are NOT handled.
- Line-based with triple-quoted-string awareness: a line that begins (after
  indentation) with ``!`` is treated as shell UNLESS it lies inside a
  ``\"\"\"``/``'''`` string.  It is not a full tokenizer, so exotic cases
  (a ``!`` line continued from a bracket/backslash on the previous line) fall
  outside its awareness — the same practical envelope as a simple transformer.
"""

import subprocess
import sys
import re
from typing import Dict, List, Tuple

SHELL_HELPER_NAME = "__jaato_run_shell__"

_SHELL_LINE = re.compile(r"^([ \t]*)!(.*)$")
_TRIPLES = ('"""', "'''")


def _update_triple_state(line: str, state):
    """Advance the triple-quoted-string state machine across one line.

    ``state`` is ``None`` (outside any triple string) or the active delimiter
    (``'\"\"\"'`` / ``\"'''\"``).  Heuristic (no escape / nested-quote handling);
    good enough to avoid rewriting ``!`` lines that live inside a multi-line
    string.
    """
    i, n = 0, len(line)
    while i < n:
        if state is None:
            for t in _TRIPLES:
                if line.startswith(t, i):
                    state = t
                    i += 3
                    break
            else:
                i += 1
        else:
            if line.startswith(state, i):
                state = None
                i += 3
            else:
                i += 1
    return state


def transform_cell_source(code: str) -> str:
    """Rewrite ``!cmd`` lines to ``__jaato_run_shell__("cmd")`` for exec.

    Non-shell lines are unchanged, and one input line maps to one output line,
    so traceback line numbers and the last-expression rule still line up.
    """
    out: List[str] = []
    state = None
    for line in code.split("\n"):
        if state is None:
            m = _SHELL_LINE.match(line)
            if m:
                out.append(f"{m.group(1)}{SHELL_HELPER_NAME}({m.group(2)!r})")
                continue  # a rewritten line opens no string
        out.append(line)
        state = _update_triple_state(line, state)
    return "\n".join(out)


def strip_shell_lines(code: str) -> Tuple[str, List[Tuple[int, str]]]:
    """For static analysis: replace ``!cmd`` lines with ``<indent>pass``.

    Returns the stripped code (parseable Python, line numbers preserved) plus
    the extracted ``(1-based line number, command)`` pairs so the analyzer can
    risk-flag each shell command while still analyzing the rest of the cell.
    """
    out: List[str] = []
    cmds: List[Tuple[int, str]] = []
    state = None
    for lineno, line in enumerate(code.split("\n"), start=1):
        if state is None:
            m = _SHELL_LINE.match(line)
            if m:
                cmds.append((lineno, m.group(2)))
                out.append(f"{m.group(1)}pass")
                continue
        out.append(line)
        state = _update_triple_state(line, state)
    return "\n".join(out), cmds


def _run_shell(cmd: str):
    """Runtime helper for a rewritten ``!cmd`` line.

    Runs ``cmd`` through the shell, streaming merged stdout+stderr line-by-line
    to ``sys.stdout`` (which the kernel redirects into the cell output).
    Inherits the kernel process environment.  Returns ``None`` (so a trailing
    ``!cmd`` displays nothing, like IPython).
    """
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
    finally:
        proc.stdout.close()
        rc = proc.wait()
    if rc != 0:
        sys.stderr.write(f"[shell] '{cmd}' exited with status {rc}\n")
    return None


def inject_shell_helper(namespace: Dict) -> None:
    """Make the ``!``-line helper available in a cell namespace (idempotent)."""
    namespace.setdefault(SHELL_HELPER_NAME, _run_shell)
