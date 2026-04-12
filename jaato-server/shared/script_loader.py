"""Shared loader for user-authored Python scripts resolved under ``.jaato/``.

This module centralizes the resolve-and-load pattern used by any framework
feature that lets users drop Python files into their workspace to be
executed at runtime. It provides:

- ``resolve_script_path`` — resolves a script reference via
  absolute → workspace ``.jaato/<path>`` → ``~/.jaato/<path>``.
- ``load_script_symbol`` — imports a script file as a synthetic module
  and returns a named top-level callable.

Consumers today:

- ``shared/plugins/permission/evaluator.py`` — loads ``evaluate`` functions
  for runtime permission decisions.

Error handling is intentionally permissive: resolution misses and load
failures return ``None`` after logging a warning, never raise. Callers
decide what "missing or broken script" means for their feature (e.g.,
fall back to policy, skip the rule).
"""

import importlib.util
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def resolve_script_path(
    path: str,
    workspace_path: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a script path through the standard ``.jaato/`` tier.

    Resolution order:

    1. If ``path`` is absolute, use it directly (must exist).
    2. If ``workspace_path`` is given, try ``<workspace>/.jaato/<path>``.
    3. Fall back to ``~/.jaato/<path>``.

    Args:
        path: Script reference — absolute, or relative to ``.jaato/``.
        workspace_path: Workspace directory for relative resolution.
            When ``None``, only the home-level ``~/.jaato/`` tier is tried
            for relative paths.

    Returns:
        Resolved ``Path`` pointing at an existing file, or ``None`` when
        no tier matches. Does not log on miss — the caller owns the
        "not found" message so it can include feature-specific context.
    """
    p = Path(path)
    if p.is_absolute():
        return p if p.is_file() else None

    if workspace_path:
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
