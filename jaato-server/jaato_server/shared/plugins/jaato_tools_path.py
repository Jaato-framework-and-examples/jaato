"""jaato's own command-line tools on the model's subprocess PATH (#1273).

The ``jaato-sdk`` skill tells a model to answer questions about the
framework by running ``jaato-doctor`` and ``jaato-scaffold`` -- "from the
same Python environment as the daemon", because both introspect the
installed build.  A daemon started as ``<venv>/bin/python -m jaato_server``
without its venv activated does not have ``<venv>/bin`` on its ``PATH``, and
a model-driven subprocess inherits that ``PATH`` (``cli``'s
``_build_subprocess_env``), so neither command resolves.  The model cannot
name them by path either: #1202 allows an executable by path only when its
directory is on that same ``PATH``.

Putting ``<venv>/bin`` on the ``PATH`` would fix it and is refused: that
directory's ``python`` and ``pip`` would let model-driven commands modify
the daemon's own environment (as root, on a root daemon) wherever nothing
earlier on the ``PATH`` shadows them.  So only the tools in
:data:`JAATO_TOOL_NAMES` are exposed, as symlinks in a directory of their
own, APPENDED to the ``PATH`` so they can never shadow a binary already on
it.

Why symlinks, in the session tmpdir
------------------------------------
Under AppArmor, exec is mediated on the RESOLVED path.  The target is
``{venv_path}/bin/<tool>``, which the runner profile already grants ``ix``
(the runner runs from that venv), while the tmpdir is ``rw`` without exec --
so a wrapper script there would be refused and a symlink is not.  The
tmpdir is the session's own (#1171), writable by a confined runner, and
never inside the workspace, so nothing appears in the Files panel or in git.

The allow-list is deliberate: ``jaato-server`` starts and stops the daemon
and ``jaato`` is the TUI; neither belongs in a model's hands.

Known limitation: a symlink cannot reset the environment, so a tool started
this way sees the subprocess ``PYTHONPATH``, which a workspace venv prepends
its site-packages to (``apply_venv_to_env``).  A conflicting package the
model installed there could shadow one of jaato's own dependencies.
"""

from __future__ import annotations

import logging
import os
import sysconfig
import tempfile
import threading
from typing import Dict, MutableMapping, Optional

logger = logging.getLogger(__name__)

#: The jaato console scripts a model may run: the two introspection tools
#: the ``jaato-sdk`` skill names.  An allow-list, never "every jaato script".
JAATO_TOOL_NAMES = ("jaato-doctor", "jaato-scaffold")

#: Directory name under the session tmpdir.
_TOOLS_DIRNAME = "jaato-tools"

_lock = threading.Lock()
# tmpdir -> built directory (or None when nothing could be exposed).  Keyed
# by tmpdir because a pool slot serves several sessions, each with its own
# pinned tmpdir (#1171).
_built: Dict[str, Optional[str]] = {}


def _scripts_dir() -> Optional[str]:
    """The running interpreter's console-script directory (``<venv>/bin``)."""
    try:
        return sysconfig.get_path("scripts")
    except (KeyError, ValueError):
        return None


def _build(tmpdir: str) -> Optional[str]:
    """Create or refresh the symlink directory under ``tmpdir``.

    Returns its path, or ``None`` when no allow-listed tool is installed next
    to this interpreter (a daemon run from a source checkout via
    ``PYTHONPATH`` has no console scripts) or the directory could not be
    written.  Never raises: a missing convenience must not fail a command.
    """
    if os.name == "nt":
        return None
    scripts = _scripts_dir()
    if not scripts:
        return None
    targets = {
        name: os.path.join(scripts, name)
        for name in JAATO_TOOL_NAMES
        if os.access(os.path.join(scripts, name), os.X_OK)
    }
    if not targets:
        return None
    directory = os.path.join(tmpdir, _TOOLS_DIRNAME)
    try:
        os.makedirs(directory, exist_ok=True)
        for name, target in targets.items():
            link = os.path.join(directory, name)
            try:
                current = os.readlink(link)
            except OSError:
                current = None
            if current == target:
                continue
            if os.path.lexists(link):
                os.unlink(link)
            os.symlink(target, link)
    except OSError as exc:
        logger.warning(
            "jaato tools: could not expose %s under %s (%s: %s); the model "
            "will not find them on PATH",
            ", ".join(targets), directory, type(exc).__name__, exc,
        )
        return None
    return directory


def jaato_tools_dir() -> Optional[str]:
    """The directory holding the jaato tool symlinks, built once per tmpdir."""
    tmpdir = tempfile.gettempdir()
    with _lock:
        cached = _built.get(tmpdir)
        # Rebuilt when the directory vanished (a session tmpdir is reaped).
        if tmpdir not in _built or (cached and not os.path.isdir(cached)):
            _built[tmpdir] = _build(tmpdir)
        return _built[tmpdir]


def append_path_entry(env: MutableMapping[str, str], directory: str) -> None:
    """Append ``directory`` to ``env['PATH']`` unless it is already an entry.

    Appended, never prepended: an entry added here supplies a name the
    ``PATH`` lacks and can never shadow one it already has.
    """
    current = env.get("PATH", "")
    if directory in current.split(os.pathsep):
        return
    # Concatenated rather than re-joined, so the existing value (empty
    # entries included, which POSIX reads as the cwd) is left exactly as is.
    env["PATH"] = f"{current}{os.pathsep}{directory}" if current else directory


def apply_jaato_tools_to_env(env: MutableMapping[str, str]) -> None:
    """Append the jaato tools directory to ``env['PATH']`` when there is one."""
    directory = jaato_tools_dir()
    if directory:
        append_path_entry(env, directory)
