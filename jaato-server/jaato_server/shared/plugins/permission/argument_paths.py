"""Path-typed ``whitelist.arguments`` / ``blacklist.arguments`` rules (#1578).

An argument rule is ``{"<tool>": {"<arg>": ["<value>", ...]}}``.  For most
arguments the value is matched as a string PREFIX (``command`` starting with
``git``).  For a PATH argument a prefix is not a containment test:
``{"writeNewFile": {"path": ["out/"]}}`` used to allow
``out/../../anywhere``, and a rule ``out`` allowed the sibling ``out2``.

So an argument whose NAME is in :data:`PATH_ARGUMENT_NAMES` is matched by
containment instead: both the value and the rule are made absolute against
the policy's base directory, normalised (``..`` collapsed) and compared by
path component, then compared again after resolving symlinks.

The two directions are deliberately asymmetric:

- **whitelist** (:func:`path_rule_allows`): ALLOW only when the value is
  inside the rule both lexically AND after symlink resolution.  Anything
  that cannot be decided (no base for a relative/absolute mix, an
  unresolvable path) is not an allow.
- **blacklist** (:func:`path_rule_denies`): DENY when the value is inside
  the rule lexically OR after symlink resolution.  The caller keeps the old
  prefix test beside it, so normalisation can only widen a deny, never
  narrow one.

Stdlib only.  The name list is a security decision and is kept here rather
than borrowed from the display heuristic in ``shared/ui_utils.py``, so a
change to how arguments are ellipsized cannot change what a rule allows.
Classifying a non-path argument as a path is the safe direction for both
lists (a whitelist narrows, a blacklist widens).
"""

import os
from typing import Optional

#: Argument names matched by path containment rather than string prefix.
PATH_ARGUMENT_NAMES = frozenset({
    "path", "file_path", "filepath", "file", "filename",
    "directory", "dir", "folder", "cwd", "working_directory",
    "target", "target_path", "source", "source_path",
    "dest", "destination", "dest_path",
    "base_path", "root_path",
    "old_path", "new_path", "old_file_path", "new_file_path",
    "notebook_path",
})


def is_path_argument(arg_name: str) -> bool:
    """True when rules on ``arg_name`` are path rules (case-insensitive)."""
    return isinstance(arg_name, str) and arg_name.lower() in PATH_ARGUMENT_NAMES


def _absolute(path: str, base: Optional[str]) -> Optional[str]:
    """``path`` made absolute against ``base`` and normalised, or ``None``.

    Without a base a relative path stays relative (normalised); the caller
    then compares like with like and refuses a relative/absolute mix.
    """
    if not isinstance(path, str):
        return None
    expanded = os.path.expanduser(path) if path.startswith("~") else path
    if not os.path.isabs(expanded) and base:
        expanded = os.path.join(base, expanded)
    return os.path.normpath(expanded or ".")


def _within(value: str, root: str) -> bool:
    """Component-wise containment of two already-normalised paths."""
    if os.path.isabs(value) != os.path.isabs(root):
        return False
    if root == ".":
        # A relative rule naming the base itself: anything that does not
        # climb out of it.
        return not os.path.isabs(value) and value != ".." and not value.startswith(".." + os.sep)
    if value == root:
        return True
    prefix = root if root.endswith(os.sep) else root + os.sep
    return value.startswith(prefix)


def _real(path: str) -> Optional[str]:
    """``os.path.realpath`` for an absolute path, ``None`` when it fails."""
    if not os.path.isabs(path):
        return None
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return None


def _comparisons(value: str, rule: str, base: Optional[str]):
    """``(lexical, resolved)`` containment verdicts; either may be ``None``."""
    v, r = _absolute(value, base), _absolute(rule, base)
    if v is None or r is None:
        return None, None
    lexical = _within(v, r)
    rv, rr = _real(v), _real(r)
    resolved = _within(rv, rr) if rv is not None and rr is not None else None
    return lexical, resolved


def path_rule_allows(value: str, rule: str, base: Optional[str]) -> bool:
    """Whitelist: ``value`` is inside ``rule`` lexically and after symlinks.

    When neither side can be resolved (no base, both relative) the lexical
    verdict alone decides; a normalised relative path that climbs (``..``)
    is never inside a relative rule.
    """
    lexical, resolved = _comparisons(value, rule, base)
    if not lexical:
        return False
    # ``resolved`` is None only for a base-less relative pair (an absolute
    # path always resolves): there is nothing on disk to consult, and the
    # lexical verdict decides.
    return True if resolved is None else resolved


def path_rule_denies(value: str, rule: str, base: Optional[str]) -> bool:
    """Blacklist: ``value`` is inside ``rule`` lexically or after symlinks."""
    lexical, resolved = _comparisons(value, rule, base)
    return bool(lexical) or bool(resolved)
