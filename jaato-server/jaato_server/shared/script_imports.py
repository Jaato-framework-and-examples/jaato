"""Where a runner-side script's imports resolve, and a per-call way to widen it (#1644).

Completion processors and ``{{!py:...}}`` prefetch scripts are imported by
:func:`~jaato_server.shared.script_loader.load_script_symbol` INSIDE the
runner's own interpreter, so a third-party ``import`` in either resolves
against the daemon's environment.  A cascade that brings its own Python (a
per-cascade venv, the #1322 ``.pth``-bridge pattern) is isolated everywhere
except here: the model's subprocesses run in the workspace tool-venv
(#1274), the client's host tools import from the same site-packages, and
these two script kinds had only two wrong options:

- get the package into the daemon's venv, shared by every workspace;
- ``sys.path.insert`` from the script, which changes the path of a runner
  slot that serves several sessions in turn (#1171), so the NEXT session's
  imports change too.

This module is the third option, and it is opt-in::

    plugin_configs:
      cli:
        workspace_venv: .jaato/tool-venv   # or the managed default
        script_imports: workspace_venv     # default: daemon

With it, a processor's or prefetch script's imports may also resolve from
the session's workspace venv, under four rules (:data:`RESOLUTION_RULES`,
which ``explain completion`` and ``explain prefetch`` render verbatim):

1. **The daemon's environment answers first.**  The venv's site-packages
   (and the plain directory lines of its ``.pth`` files) are APPENDED to
   ``sys.path``, so a package the daemon already has, or has imported,
   resolves to the daemon's copy.  The venv fills gaps; it does not
   replace a version.  Genuine version isolation needs the script out of
   process, which is not what this is.
2. **jaato's own packages are never taken from the venv.**  A meta-path
   finder installed for the scope answers every top-level ``jaato*`` name
   from the daemon's path only, and refuses it when the daemon has none.
   This is #1322's rule for the tool-venv, held here by a check rather
   than by ordering alone.
3. **The scope lasts for the load and for each call, and no longer.**  The
   appended entries are removed when the call returns, and every module
   loaded from the venv's directories is taken out of ``sys.modules`` and
   PARKED with the scope, to be restored on its next entry.  So a slot's
   next session, which does not hold this scope, cannot import them, while
   the session that does keeps one module object across its calls (a
   processor's module-level state survives).
4. **No grant is added.**  Under confinement the runner reads the venv with
   whatever its profile already grants on that path: the tool-venv lives in
   the workspace, which the runner may read, so pure-Python packages load;
   a C extension needs an exec-mapping grant the runner's base profile
   does not carry on the workspace, exactly as for the tool's own
   subprocesses, and a venv outside the workspace is not readable at all.

Costs, stated:

- **Scoped calls are serialised per process.**  ``sys.path`` and
  ``sys.modules`` belong to the interpreter, so an active scope holds one
  process-wide lock from entry to exit.  Two sessions sharing a runner (a
  parent and its in-process subagents) run their scoped scripts one at a
  time.  Nothing takes the lock when no session opted in.
- **Other threads see the venv while a call is inside the scope.**  An
  unscoped import elsewhere in the process during that window could find a
  module only the venv has; what it loads is parked at exit with the rest.
- **The tool-venv is model-writable.**  A gate that imports from it trusts
  what the model may have ``pip install``-ed there, for any name the
  daemon does not already provide.  That is why this is opt-in, and why a
  cascade that needs its gate's imports to be its author's should point
  ``workspace_venv`` at a venv the model does not write.
- **Editable installs through an import hook are not seen.**  A ``.pth``
  line that is an ``import`` statement (the finder-style editable install)
  would install a process-wide hook, so it is skipped; a ``.pth`` that names
  a directory is followed.
"""

from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.machinery
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: The ``plugin_configs.cli`` key that opts a session's scripts into its
#: workspace venv.  Beside ``workspace_venv`` because it names the same
#: venv; the cli plugin declares it in its config schema so ``validate``
#: checks the value.
SCRIPT_IMPORTS_KEY = "script_imports"

#: Default: imports resolve in the runner's interpreter, as before #1644.
SCRIPT_IMPORTS_DAEMON = "daemon"

#: Imports may also resolve from the session's ``workspace_venv``.
SCRIPT_IMPORTS_WORKSPACE_VENV = "workspace_venv"

#: The closed vocabulary, with what each value means.  The cli schema's
#: ``enum`` and the ``explain`` topics read this.
SCRIPT_IMPORTS_VALUES: Dict[str, str] = {
    SCRIPT_IMPORTS_DAEMON: (
        "the runner's own interpreter only (the default)"
    ),
    SCRIPT_IMPORTS_WORKSPACE_VENV: (
        "also the session's workspace_venv site-packages, appended for the "
        "load and each call only"
    ),
}

#: The four rules, one line each, rendered by ``explain completion`` and
#: ``explain prefetch`` so the two topics cannot word them differently.
RESOLUTION_RULES: Tuple[str, ...] = (
    "the daemon's environment answers first: the venv's site-packages are "
    "APPENDED to sys.path, so a package the daemon has resolves to the "
    "daemon's copy (the venv fills gaps, it does not replace a version)",
    "jaato's own packages (any top-level name starting 'jaato') are never "
    "taken from the venv",
    "the venv is on sys.path for the load and for each call only; modules "
    "loaded from it are parked with the session's scope, so a slot's next "
    "session cannot import them",
    "no confinement grant is added: the runner reads the venv with what its "
    "profile already grants (the tool-venv is in the workspace, so pure "
    "Python loads; a C extension needs an exec-mapping grant the base "
    "profile does not carry there)",
)

#: Name of the ``.pth`` the tool-venv's pip bridge writes (#1322).  It names
#: only a directory holding the runner's pip, which the runner already has.
_PIP_BRIDGE_PTH = "_jaato_runner_bridge.pth"

#: One lock for every active scope in the process: ``sys.path`` and
#: ``sys.modules`` are interpreter-wide.  Reentrant so a scoped call that
#: loads another scoped script on the same thread does not deadlock.
_SCOPE_LOCK = threading.RLock()

#: Per-thread stack of the scopes currently entered, so re-entering the
#: scope already on top is a no-op rather than a second append.
_ACTIVE = threading.local()


def _is_protected(name: str) -> bool:
    """Whether a top-level module name is jaato's own (rule 2)."""
    return name.split(".", 1)[0].startswith("jaato")


class _ProtectJaatoNames(importlib.abc.MetaPathFinder):
    """Meta-path finder that answers ``jaato*`` names from the daemon's path.

    Installed at the front of ``sys.meta_path`` for the scope's duration.
    For a top-level jaato name it searches ``sys.path`` minus the scope's
    directories, so the venv cannot supply one, and raises when the daemon
    has none rather than letting the venv's copy through.  Every other name,
    and every submodule lookup (which goes through the parent package's
    ``__path__``, never ``sys.path``), falls through to the normal finders.
    """

    def __init__(self, scope_dirs: Tuple[str, ...]) -> None:
        self._scope_dirs = scope_dirs

    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        """Return the daemon's spec for a top-level jaato name, else defer."""
        if path is not None or not _is_protected(fullname):
            return None
        daemon_path = [p for p in sys.path if _norm(p) not in self._scope_dirs]
        spec = importlib.machinery.PathFinder.find_spec(fullname, daemon_path)
        if spec is None:
            raise ModuleNotFoundError(
                f"{fullname!r} is jaato's own package and is never imported "
                f"from the session's workspace venv (#1644); the daemon's "
                f"environment does not provide it",
                name=fullname,
            )
        return spec


def _norm(path: str) -> str:
    """A comparable form of a ``sys.path`` entry or module origin."""
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return path


def _pth_dirs(site_dir: str) -> List[str]:
    """The directory lines of the ``.pth`` files in ``site_dir``, in order.

    ``import`` lines are skipped: running them would install process-wide
    hooks (the finder-style editable install), which a per-call scope cannot
    undo.  The pip bridge is skipped too, since it names only the runner's
    own pip.  A relative line resolves against ``site_dir``, as ``site`` does.
    """
    out: List[str] = []
    try:
        names = sorted(n for n in os.listdir(site_dir) if n.endswith(".pth"))
    except OSError:
        return out
    for name in names:
        if name == _PIP_BRIDGE_PTH:
            continue
        try:
            with open(os.path.join(site_dir, name), encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("import ", "import\t")):
                continue
            path = line if os.path.isabs(line) else os.path.join(site_dir, line)
            if os.path.isdir(path):
                out.append(path)
    return out


def _module_dirs(module: ModuleType) -> List[str]:
    """Where a loaded module came from: its file, else its package path."""
    found: List[str] = []
    origin = getattr(module, "__file__", None)
    if isinstance(origin, str):
        found.append(origin)
        return found
    for entry in list(getattr(module, "__path__", None) or ()):
        if isinstance(entry, str):
            found.append(entry)
    return found


@dataclass
class ScriptImportScope:
    """One session's widened import scope (rules 1 to 3).

    Built by :func:`resolve_script_import_scope`; entered with ``with`` or
    applied to a callable with :meth:`wrap`.  Holds the modules loaded from
    its directories between entries (``_parked``), which is what keeps them
    out of ``sys.modules`` for every other session on the slot.

    Attributes:
        venv_path: The resolved workspace venv.
        dirs: The directories appended to ``sys.path``: the venv's
            site-packages, then the directory lines of its ``.pth`` files.
    """

    venv_path: str
    dirs: Tuple[str, ...]
    _parked: Dict[str, ModuleType] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._norm_dirs = tuple(_norm(d) for d in self.dirs)

    def describe(self) -> str:
        """One line naming where this scope's imports may resolve."""
        return (f"daemon environment, then {', '.join(self.dirs)} "
                f"(workspace venv {self.venv_path})")

    def owns(self, module: ModuleType) -> bool:
        """Whether every location of ``module`` lies under this scope's dirs."""
        locations = _module_dirs(module)
        if not locations:
            return False
        for loc in locations:
            n = _norm(loc)
            if not any(n == d or n.startswith(d + os.sep) for d in self._norm_dirs):
                return False
        return True

    def __enter__(self) -> "ScriptImportScope":
        """Append the dirs, restore parked modules, guard jaato names."""
        stack = getattr(_ACTIVE, "stack", None)
        if stack is None:
            stack = _ACTIVE.stack = []
        _SCOPE_LOCK.acquire()
        if stack and stack[-1][0] is self:
            stack.append((self, None))
            return self
        try:
            # Candidates for parking at exit are the names that are new by
            # then, restored ones included (snapshot taken before the
            # restore).  A module loaded before entry did not come from
            # directories only this scope puts on the path.
            before = set(sys.modules)
            for name, module in self._parked.items():
                sys.modules.setdefault(name, module)
            self._parked = {}
            added = [d for d in self.dirs if d not in sys.path]
            sys.path.extend(added)
            guard = _ProtectJaatoNames(self._norm_dirs)
            sys.meta_path.insert(0, guard)
            importlib.invalidate_caches()
        except BaseException:
            _SCOPE_LOCK.release()
            raise
        stack.append((self, (added, guard, before)))
        return self

    def __exit__(self, *exc: Any) -> None:
        """Remove what was added and park the venv's modules."""
        stack = _ACTIVE.stack
        _, state = stack.pop()
        try:
            if state is None:
                return
            added, guard, before = state
            try:
                sys.meta_path.remove(guard)
            except ValueError:
                pass
            for d in added:
                try:
                    sys.path.remove(d)
                except ValueError:
                    pass
            for name in [n for n in sys.modules if n not in before]:
                module = sys.modules.get(name)
                if module is not None and self.owns(module):
                    self._parked[name] = sys.modules.pop(name)
        finally:
            _SCOPE_LOCK.release()

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` run inside this scope on every call."""
        @functools.wraps(fn)
        def scoped(*args: Any, **kwargs: Any) -> Any:
            with self:
                return fn(*args, **kwargs)
        scoped.__jaato_import_scope__ = self  # type: ignore[attr-defined]
        return scoped


def resolve_script_import_scope(
    cli_config: Optional[Mapping[str, Any]],
    workspace_path: Optional[str],
) -> Tuple[Optional[ScriptImportScope], str]:
    """The scope a session's ``plugin_configs.cli`` block asks for.

    Args:
        cli_config: The session's own ``plugin_configs.cli`` block (the
            managed ``workspace_venv`` default is already folded in on the
            runner path, #1274).
        workspace_path: The session workspace, against which a relative
            ``workspace_venv`` resolves.

    Returns:
        ``(scope, where)``: ``scope`` is ``None`` when the session did not
        opt in or the opt-in cannot be honoured; ``where`` says where the
        scripts' imports resolve, for the log line and diagnostics.
    """
    from .plugins.workspace_venv import resolve_venv_path, venv_site_packages

    cfg = cli_config if isinstance(cli_config, Mapping) else {}
    mode = cfg.get(SCRIPT_IMPORTS_KEY) or SCRIPT_IMPORTS_DAEMON
    if mode == SCRIPT_IMPORTS_DAEMON:
        return None, "daemon environment"
    if mode != SCRIPT_IMPORTS_WORKSPACE_VENV:
        return None, (f"daemon environment (unknown {SCRIPT_IMPORTS_KEY} "
                      f"{mode!r}; one of {sorted(SCRIPT_IMPORTS_VALUES)})")
    try:
        venv = resolve_venv_path(cfg.get("workspace_venv"), workspace_path)
    except ValueError as exc:
        return None, f"daemon environment ({exc})"
    if not venv:
        return None, ("daemon environment (script_imports asks for the "
                      "workspace venv and the session has no workspace_venv)")
    site_dir = venv_site_packages(venv)
    if site_dir is None:
        return None, (f"daemon environment (workspace venv {venv} has no "
                      f"site-packages yet)")
    scope = ScriptImportScope(venv_path=venv,
                              dirs=tuple([site_dir] + _pth_dirs(site_dir)))
    return scope, scope.describe()


def script_import_scope_for(session: Any) -> Optional[ScriptImportScope]:
    """The scope for ``session``'s scripts, kept on the session once found.

    Kept so the parked modules belong to the session for its whole life: a
    processor loaded once and called on every ``signal_completion`` keeps
    one module object.  A miss is NOT kept: the tool-venv is created by the
    first command that needs it, so a prefetch at session-prep may find no
    venv that a completion processor later does.  ``None`` when the session
    did not opt in, or the venv is not there yet.  Where the imports resolve
    is logged at INFO when a scope is first built, and once at WARNING when
    the opt-in cannot be honoured.
    """
    if session is None:
        return None
    # Read from the instance dict: a stand-in session (a mock) answers any
    # attribute, and its answer is not a scope.
    state = getattr(session, "__dict__", None) or {}
    cached = state.get("_script_import_scope")
    if isinstance(cached, ScriptImportScope):
        return cached
    declared = getattr(session, "declared_plugin_config", None)
    cli_cfg = declared("cli") if callable(declared) else None
    if not isinstance(cli_cfg, Mapping) or not cli_cfg.get(SCRIPT_IMPORTS_KEY):
        return None
    workspace = getattr(session, "workspace_path", None)
    scope, where = resolve_script_import_scope(
        cli_cfg, workspace if isinstance(workspace, str) else None,
    )
    if scope is not None:
        logger.info("script imports (processors, prefetch) resolve in: %s",
                    where)
        _remember(session, "_script_import_scope", scope)
    elif not state.get("_script_import_scope_warned"):
        logger.warning("script imports (processors, prefetch) resolve in: %s",
                       where)
        _remember(session, "_script_import_scope_warned", True)
    return scope


def _remember(session: Any, name: str, value: Any) -> None:
    """Set an attribute on the session, tolerating one that refuses it."""
    try:
        setattr(session, name, value)
    except AttributeError:
        pass

