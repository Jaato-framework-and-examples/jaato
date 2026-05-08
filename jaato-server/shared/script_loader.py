"""Shared loader for user-authored Python scripts resolved under ``.jaato/``.

This module centralizes the resolve-and-load pattern used by any framework
feature that lets users drop Python files into their workspace to be
executed at runtime. It provides:

- ``resolve_script_path`` — resolves a script reference via
  absolute → workspace ``.jaato/<path>`` → ``~/.jaato/<path>``.
- ``load_script_symbol`` — imports a script file as a synthetic module
  and returns a named top-level callable. Re-executes the script on
  every call so handler edits are picked up without a daemon restart.

Consumers today:

- ``shared/plugins/permission/evaluator.py`` — loads ``evaluate`` functions
  for runtime permission decisions.
- Reactor handlers under ``<config_root>/.jaato/scripts/reactors/``
  loaded via the premium reactor framework.

Error handling is intentionally permissive: resolution misses and load
failures return ``None`` after logging a warning, never raise. Callers
decide what "missing or broken script" means for their feature (e.g.,
fall back to policy, skip the rule).

**sys.modules cache invalidation for transitively-imported helpers
(server 0.6.60+).**

When a handler script under ``<config_root>/.jaato/scripts/`` imports
a sibling helper module (typical pattern: ``sys.path.insert(0,
str(Path(__file__).parent))`` followed by
``from _session_chain import …``), the helper module lands in
``sys.modules``. ``load_script_symbol`` re-executes the HANDLER on
every call (it doesn't put the synthetic handler module in
``sys.modules``), but the from-import resolves the helper against
``sys.modules`` — daemon-lifetime cached. Edits to helper symbols are
invisible until daemon restart, which produces the diagnostic shape
``cannot import name 'X' from '<helper>'`` after a helper edit added
``X``.

This module tracks helper modules transitively imported during each
``load_script_symbol`` call (filtered to those whose ``__file__``
sits under a ``.jaato/scripts/`` ancestor of the loaded handler). On
every subsequent call, it stats each tracked helper's source file;
when ``mtime`` has advanced, it ``importlib.reload``s the helper
before re-executing the handler. Stdlib + 3rd-party + jaato-server's
own modules are not tracked — they are loaded once at daemon startup
and aren't expected to change at runtime (their reload semantics are
a different problem class addressed by daemon restart).
"""

import importlib.util
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


# Per-process helper-tracking state.  Maps absolute ``__file__`` path of
# a helper module → recorded mtime at the last load_script_symbol call
# that observed it.  Threading lock guards compound check-then-write
# sequences across concurrent script loads (multiple sessions / runners
# sharing the daemon).  Single-thread reads are safe without the lock
# (CPython dict-ops are atomic for individual operations) but the
# stat-and-reload sequence wants a consistent snapshot.
_tracked_helpers: Dict[str, float] = {}
_tracked_helpers_lock = threading.Lock()


def _scripts_root_for(file_path: Path) -> Optional[Path]:
    """Find the ``.jaato/scripts/`` ancestor of ``file_path``.

    Helpers transitively imported by a handler script are tracked iff
    their ``__file__`` is under the same ``.jaato/scripts/`` root —
    keeps the tracker scoped to user-authored helpers and never to
    stdlib / 3rd-party / framework modules.

    Returns:
        The closest ``.jaato/scripts`` ancestor as an absolute Path, or
        ``None`` if the handler isn't under any ``.jaato/scripts/``
        directory (e.g. permission evaluators loaded from elsewhere).
    """
    p = file_path.resolve()
    for ancestor in p.parents:
        if ancestor.name == "scripts" and ancestor.parent.name == ".jaato":
            return ancestor
    return None


def _is_under(child: Path, parent: Path) -> bool:
    """True iff ``child`` is contained under ``parent`` (path-prefix check)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _refresh_stale_helpers() -> None:
    """Reload tracked helper modules whose source file mtime has changed.

    Called at the top of each ``load_script_symbol`` invocation. Walks
    the tracked-helpers map, stats each helper's source, and
    ``importlib.reload``s any module whose source has been modified
    since the last observation. Modules that no longer have an entry
    in ``sys.modules`` (evicted by some external mechanism) or whose
    source file has been deleted are removed from the tracker.

    Reload failures (syntax errors in the edited helper, etc.) are
    logged at warning level and tracker state is preserved with the
    OLD mtime so the next call retries — a corrupt-edit shouldn't stop
    the framework from re-attempting the reload after the operator
    fixes the syntax.

    **Lock scope.** The tracker lock is held for the entire
    stat-and-reload loop. Helpers with heavy reload-time side-effects
    (long-running import-time setup, network calls, large file reads
    at module top level) could bottleneck concurrent
    ``load_script_symbol`` calls.  Helper authors should keep
    import-time work cheap; reactor handlers expect the per-load
    overhead to stay in the microseconds-to-milliseconds range.
    """
    with _tracked_helpers_lock:
        # Snapshot keys to avoid mutating-while-iterating; per-key
        # operations below mutate the shared dict.
        for file_path, recorded_mtime in list(_tracked_helpers.items()):
            module = _module_with_file(file_path)
            if module is None:
                _tracked_helpers.pop(file_path, None)
                continue
            try:
                current_mtime = os.path.getmtime(file_path)
            except OSError:
                _tracked_helpers.pop(file_path, None)
                continue
            if current_mtime <= recorded_mtime:
                continue
            try:
                importlib.reload(module)
            except Exception as exc:
                logger.warning(
                    "Failed to reload helper %s after mtime change: %s "
                    "(operator fix the helper and trigger another load)",
                    file_path, exc,
                )
                continue
            _tracked_helpers[file_path] = current_mtime
            logger.debug(
                "Reloaded helper %s (mtime %s -> %s)",
                file_path, recorded_mtime, current_mtime,
            )


def _module_with_file(file_path: str) -> Optional[object]:
    """Find a module in sys.modules whose ``__file__`` matches ``file_path``.

    Compares against the resolved form of each module's ``__file__`` so
    the lookup is consistent with the tracker's resolved-path keys
    (``_track_new_helpers`` normalizes via ``Path.resolve()`` on store).

    Linear scan over sys.modules — tolerable because the call site
    fires at script-load frequency (cascades, permission evaluations),
    not at hot-path frequency. Avoids relying on a separate
    file-path → name index that could go stale.

    Server 0.6.69+: snapshots ``sys.modules.values()`` via ``list(...)``
    before iterating.  ``importlib.reload(...)`` mutates ``sys.modules``,
    and a concurrent ``_refresh_stale_helpers`` call (e.g. from another
    session's reactor) could trigger a reload while THIS function is
    mid-scan — raising ``RuntimeError: dictionary changed size during
    iteration``.  The tracker lock above protects ``_tracked_helpers``
    but not ``sys.modules`` (which has no public lock).  Snapshotting
    here is cheap and race-safe.
    """
    for mod in list(sys.modules.values()):
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        try:
            if str(Path(mod_file).resolve()) == file_path:
                return mod
        except OSError:
            # Module's source file deleted between import and now.
            continue
    return None


def _track_new_helpers(
    new_module_names: set, scripts_root: Path,
) -> None:
    """Record any newly-imported modules under ``scripts_root`` for invalidation.

    Called after a successful ``exec_module`` of a handler script.
    The new module names (those that appeared in ``sys.modules`` during
    the load and weren't there before) include any helpers the handler
    transitively imported. Filter to those whose source lives under
    ``scripts_root`` (skip stdlib, 3rd-party, framework modules), then
    record each (path, mtime) pair so the next load can detect edits.

    **Partial-import corner case.** Only fires after a SUCCESSFUL
    ``exec_module``.  If the handler's import sequence partially
    completes — e.g. ``import _helper_a; import _helper_b`` raises in
    B's top-level code — A IS in ``sys.modules`` but doesn't get
    tracked here because the surrounding ``load_script_symbol`` call
    returns ``None`` before reaching this function.  Future edits to
    A won't trigger reload until the next time a handler successfully
    completes loading and re-imports A.  Not a regression vs the
    pre-0.6.60 behavior; tracker state is correct-but-incomplete for
    that helper until the next clean handler load.
    """
    with _tracked_helpers_lock:
        for name in new_module_names:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                continue
            try:
                resolved = str(Path(mod_file).resolve())
            except OSError:
                continue
            if not _is_under(Path(resolved), scripts_root):
                continue
            try:
                _tracked_helpers[resolved] = os.path.getmtime(resolved)
            except OSError:
                continue


def resolve_script_path(
    path: str,
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a script path through the standard ``.jaato/`` tier.

    Resolution order:

    1. If ``path`` is absolute, use it directly (must exist).
    2. **Workspace tier** —
       * if ``config_root`` is set: ``<config_root>/<path>``;
       * else if ``workspace_path`` is set: ``<workspace>/.jaato/<path>``.
    3. **User tier** — ``~/.jaato/<path>`` (always tried).

    Args:
        path: Script reference — absolute, or relative to ``.jaato/``.
        workspace_path: Workspace directory for relative resolution.
            When ``None`` and ``config_root`` is also ``None``, only
            the home-level ``~/.jaato/`` tier is tried for relative paths.
        config_root: Optional override for the workspace tier.  When set,
            ``<config_root>/<path>`` replaces ``<workspace>/.jaato/<path>``.
            See :func:`shared.config_resolver.resolve_config_search_path`.

    Returns:
        Resolved ``Path`` pointing at an existing file, or ``None`` when
        no tier matches. Does not log on miss — the caller owns the
        "not found" message so it can include feature-specific context.
    """
    p = Path(path)
    if p.is_absolute():
        return p if p.is_file() else None

    if config_root:
        cr_path = Path(config_root).expanduser().resolve() / path
        if cr_path.is_file():
            return cr_path
    elif workspace_path:
        ws_path = Path(workspace_path) / ".jaato" / path
        if ws_path.is_file():
            return ws_path

    home_path = Path.home() / ".jaato" / path
    if home_path.is_file():
        return home_path

    return None


def load_script_symbol(
    file_path: Path,
    symbol: str,
    module_prefix: str = "_jaato_script",
) -> Optional[Callable]:
    """Load ``symbol`` as a top-level callable from a Python script.

    The script is imported as a synthetic module named
    ``{module_prefix}_{file_stem}``. The module is **not** inserted into
    ``sys.modules``, so two scripts with the same filename in different
    directories do not collide — the name is only used in log output.

    Args:
        file_path: Absolute path to the script file (as returned by
            ``resolve_script_path``).
        symbol: Name of the top-level attribute to return, e.g. ``"evaluate"``
            for permission evaluators or ``"execute"`` for reactor actions.
        module_prefix: Prefix used when building the synthetic module
            name for log messages. Pass a feature-specific prefix so
            load errors are attributable.

    Returns:
        The loaded callable, or ``None`` when the file cannot be loaded,
        the attribute is missing, or the attribute is not callable.
        All failure modes are logged as warnings.
    """
    # Server 0.6.60+: refresh stale helpers (mtime check) before every
    # load. See module docstring for the bug class this closes.
    _refresh_stale_helpers()

    # Snapshot sys.modules keys before the load so we can detect any
    # helpers transitively imported by the handler. The snapshot is
    # cheap (set of strings).
    pre_load_modules = set(sys.modules.keys())

    module_name = f"{module_prefix}_{file_path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            logger.warning("Cannot load script module from %s", file_path)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning("Failed to load script from %s: %s", file_path, exc)
        return None

    # Server 0.6.60+: track any helpers transitively imported during the
    # load, scoped to the handler's .jaato/scripts/ root. Stdlib + 3rd
    # party + framework modules are filtered out by path-prefix.
    scripts_root = _scripts_root_for(file_path)
    if scripts_root is not None:
        new_module_names = set(sys.modules.keys()) - pre_load_modules
        if new_module_names:
            _track_new_helpers(new_module_names, scripts_root)

    fn = getattr(module, symbol, None)
    if fn is None:
        logger.warning(
            "Script %s has no '%s' function", file_path, symbol,
        )
        return None
    if not callable(fn):
        logger.warning(
            "Script %s: '%s' is not callable", file_path, symbol,
        )
        return None
    return fn
