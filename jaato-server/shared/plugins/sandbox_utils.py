"""Shared sandbox utilities for path validation.

This module provides common path validation logic used by multiple plugins
(file_edit, cli, etc.) to enforce workspace sandboxing with special handling
for the .jaato configuration directory.

Key feature: .jaato Access Restriction
=======================================
The .jaato directory is **denied by default** for model tool calls. The model
cannot access any files under .jaato unless the user explicitly grants access
via the ``sandbox add`` command (which registers the path in the plugin
registry's authorized paths).

Even when explicitly authorized, containment rules still apply:
- .jaato can be a symlink pointing outside the workspace (allowed)
- Once inside .jaato, paths cannot escape that boundary
- Nested symlinks inside .jaato are NOT allowed

Example:
    Workspace: /home/user/project/
    .jaato symlink: /home/user/project/.jaato -> /home/user/.jaato (external)

    DENIED BY DEFAULT (requires ``sandbox add``):
        .jaato/config.json     -> /home/user/.jaato/config.json
        .jaato/vision/img.png  -> /home/user/.jaato/vision/img.png

    ALWAYS BLOCKED (even with ``sandbox add``):
        .jaato/../secret.txt   -> /home/user/secret.txt (escapes boundary)
        .jaato/plugins -> /opt  (nested symlink, not allowed)

Key feature: /tmp Access
========================
The /tmp directory is allowed by default for sandboxed tools to support
temporary file operations. This can be disabled via the allow_tmp parameter.

Key feature: POSIX pseudo-devices
=================================
``/dev/null`` and its siblings (``/dev/zero``, ``/dev/stdout``,
``/dev/fd/<n>``, ...) are always allowed -- they hold no workspace data and
their semantics are fixed by the kernel, so they are not what the sandbox
exists to contain.  Refusing them made ``2>/dev/null`` -- one of the most
common idioms a model emits -- look like a broken environment rather than a
boundary (jaato issue #784).  The allowance is an explicit list, not a
``/dev/`` prefix match: block devices and ``/dev/mem`` stay subject to the
ordinary workspace rules.  See :func:`is_pseudo_device_path`.
"""

import os
import re
import tempfile
from functools import lru_cache
from typing import Optional, Tuple

from shared.path_utils import msys2_to_windows_path, normalize_for_comparison


# The special configuration directory that gets contained symlink escape
JAATO_CONFIG_DIR = ".jaato"

# System temp directories that are allowed by default
SYSTEM_TEMP_PATHS = ["/tmp", tempfile.gettempdir()]

# Standard POSIX pseudo-devices, allowed regardless of workspace_root.
#
# These are not filesystem locations in the sandbox's sense: they carry no
# workspace data, reading or writing them cannot reach another tenant's
# files, and their semantics are fixed by the kernel.  ``2>/dev/null`` is
# one of the most common idioms a model emits, and refusing it made the
# sandbox look like a broken environment rather than a boundary -- the
# refusal is reported as ``<cmd>: /dev/null: No such file or directory``
# (see ``CLIToolPlugin._make_not_found_result``), from which the only
# available inference is "this machine has no /dev/null" (jaato issue
# #784).
#
# The list is deliberately exhaustive rather than a ``/dev/`` prefix
# match: block devices (``/dev/sda``), the tty multiplexer's peers
# (``/dev/pts/*``) and memory devices (``/dev/mem``) are real access, and
# stay subject to the ordinary workspace rules.
PSEUDO_DEVICE_PATHS = frozenset({
    "/dev/null",
    "/dev/zero",
    "/dev/full",
    "/dev/random",
    "/dev/urandom",
    "/dev/tty",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
})

# ``/dev/fd/<n>`` names a descriptor the process already holds -- what
# bash process substitution (``<(...)``, ``>(...)``) expands to.  Allowing
# it grants nothing the process cannot already reach.
_PSEUDO_DEVICE_FD_RE = re.compile(r"^/dev/fd/\d+$")


def is_pseudo_device_path(path: str) -> bool:
    """Return ``True`` for a standard POSIX pseudo-device path.

    Matches the literal path, deliberately *before* symlink resolution:
    ``/dev/stdin`` and friends resolve through ``/proc/self/fd/<n>`` to
    whatever the descriptor currently points at, which may legitimately
    live outside the workspace.  It is the well-known name that is being
    allowed, not the target it happens to have right now.

    Args:
        path: Path to check.  Normalised with ``os.path.normpath`` so
            ``/dev/./null`` and ``/dev/foo/../null`` are recognised, but
            not made absolute -- a relative ``dev/null`` inside the
            workspace is an ordinary file and is not matched here.

    Returns:
        ``True`` if *path* names one of the allowed pseudo-devices.
    """
    if not path or not path.startswith("/"):
        return False
    normalised = os.path.normpath(path)
    return (
        normalised in PSEUDO_DEVICE_PATHS
        or bool(_PSEUDO_DEVICE_FD_RE.match(normalised))
    )


def is_jaato_path(path: str, workspace_root: str) -> bool:
    """Check if a path references the workspace's .jaato config directory.

    This checks if the path goes through ``<workspace>/.jaato/``, even if
    it later escapes via ``..`` traversal.  Detects traversal attacks like:
    - ``.jaato/../secret.txt``
    - ``/workspace/.jaato/../../etc/passwd``

    The check is **workspace-relative**: a ``.jaato`` component appearing
    in the workspace path itself (e.g. WS-provisioned workspaces under
    ``~/.jaato/workspaces/sessions/<id>/``) is NOT treated as accessing
    the workspace's ``.jaato`` config.

    Args:
        path: Path to check (may contain .. traversal).
        workspace_root: The workspace root directory.

    Returns:
        True if path references the workspace's ``.jaato`` config dir.
    """
    # Check the workspace-relative .jaato path
    jaato_dir = os.path.join(workspace_root, JAATO_CONFIG_DIR)
    # Use normalized comparison to handle mixed separators (e.g., MSYS2 on Windows
    # where os.path.join uses backslashes but paths may have forward slashes)
    norm_path = normalize_for_comparison(path)
    norm_jaato_dir = normalize_for_comparison(jaato_dir)
    norm_jaato_prefix = norm_jaato_dir + '/'

    # Direct check for absolute paths
    if norm_path == norm_jaato_dir or norm_path.startswith(norm_jaato_prefix):
        return True

    # Check for traversal attempts: compute the workspace-relative form
    # (without abspath normalization, so .. components are preserved)
    # and look for .jaato as a component there.  This catches
    # ``.jaato/../secret`` and ``/workspace/.jaato/../../etc/passwd``
    # without false-positiving on workspace paths that happen to contain
    # ``.jaato`` as part of their location prefix (WS-provisioned
    # workspaces under ``~/.jaato/workspaces/sessions/<id>/``).
    norm_workspace = normalize_for_comparison(workspace_root)
    if norm_path.startswith(norm_workspace + '/'):
        relative = norm_path[len(norm_workspace) + 1:]
    elif not os.path.isabs(path):
        relative = norm_path
    else:
        # Absolute path outside the workspace — it's not accessing the
        # workspace's .jaato config dir.  Other checks (sandbox bounds)
        # will handle whether it's allowed at all.
        return False

    rel_parts = relative.split('/')
    if JAATO_CONFIG_DIR in rel_parts:
        return True

    return False


def get_jaato_boundary(workspace_root: str) -> Optional[str]:
    """Get the resolved .jaato directory boundary.

    If .jaato is a symlink, returns the resolved target directory.
    If .jaato doesn't exist, returns None.

    Args:
        workspace_root: The workspace root directory.

    Returns:
        Resolved canonical path to .jaato, or None if it doesn't exist.
    """
    jaato_dir = os.path.join(workspace_root, JAATO_CONFIG_DIR)
    if not os.path.exists(jaato_dir):
        return None
    # Resolve symlinks to get the actual directory
    return os.path.realpath(jaato_dir)


def detect_jaato_symlink(workspace_root: str) -> Tuple[bool, Optional[str]]:
    """Detect if .jaato is a symlink and return info for logging.

    Args:
        workspace_root: The workspace root directory.

    Returns:
        Tuple of (is_symlink, target_path).
        If .jaato doesn't exist or isn't a symlink, returns (False, None).
    """
    jaato_dir = os.path.join(workspace_root, JAATO_CONFIG_DIR)
    if not os.path.islink(jaato_dir):
        return False, None
    target = os.path.realpath(jaato_dir)
    return True, target


def has_nested_symlink(path: str, jaato_boundary: str, workspace_root: str) -> bool:
    """Check if there are any symlinks inside .jaato (not allowed).

    This walks the path from .jaato down to the target and checks each
    component. If any intermediate directory is a symlink, it's blocked.

    Note: The top-level .jaato symlink is allowed; this only checks for
    symlinks INSIDE .jaato.

    Args:
        path: Absolute path to check.
        jaato_boundary: The resolved .jaato root boundary.
        workspace_root: The workspace root directory.

    Returns:
        True if there's a nested symlink (path should be BLOCKED).
        False if path is safe (no nested symlinks).
    """
    jaato_dir = os.path.join(workspace_root, JAATO_CONFIG_DIR)

    # Get the path relative to .jaato in the workspace
    try:
        rel_from_jaato = os.path.relpath(path, jaato_dir)
    except ValueError:
        # Different drives on Windows
        return True

    # If the path goes up (..) from .jaato, it's trying to escape
    if rel_from_jaato.startswith('..'):
        return True

    # Walk from jaato_boundary through each component
    # Split on both separators for MSYS2/Windows compatibility
    parts = rel_from_jaato.replace('\\', '/').split('/')
    current = jaato_boundary

    for part in parts:
        if not part or part == '.':
            continue

        current = os.path.join(current, part)

        # Check if this component is a symlink
        # We use lexists + islink to handle broken symlinks too
        if os.path.islink(current):
            # Found a nested symlink - not allowed
            return True

    return False


def is_path_within_jaato_boundary(
    path: str,
    workspace_root: str,
    jaato_boundary: str
) -> bool:
    """Check if a resolved path is within the .jaato containment boundary.

    This performs two checks:
    1. The final resolved path must be within jaato_boundary
    2. There must be no nested symlinks inside .jaato

    Args:
        path: Absolute path to check (may include traversal like ../).
        workspace_root: The workspace root directory.
        jaato_boundary: The resolved .jaato directory (from get_jaato_boundary).

    Returns:
        True if path is safely within .jaato boundary, False otherwise.
    """
    # Resolve to canonical path (follows ALL symlinks, normalizes ..)
    real_path = os.path.realpath(path)

    # Check if resolved path is within .jaato boundary
    # Use normalized comparison to handle mixed separators (MSYS2/Windows)
    norm_real = normalize_for_comparison(real_path)
    norm_boundary = normalize_for_comparison(jaato_boundary).rstrip('/') + '/'
    if not (norm_real == normalize_for_comparison(jaato_boundary) or norm_real.startswith(norm_boundary)):
        # Path escapes .jaato boundary (e.g., .jaato/../secret.txt)
        return False

    # Check for nested symlinks inside .jaato (not allowed)
    if has_nested_symlink(path, jaato_boundary, workspace_root):
        return False

    return True


@lru_cache(maxsize=8)
def _resolved_temp_roots(roots: Tuple[str, ...]) -> Tuple[str, ...]:
    """Canonicalise the temp roots once per distinct ``SYSTEM_TEMP_PATHS``.

    Taking the tuple as an argument rather than reading the module global
    keeps the cache correct when tests substitute the roots, and keeps two
    extra ``realpath`` calls off every path check.

    Args:
        roots: The temp-directory roots to canonicalise.

    Returns:
        The roots, resolved and normalised for comparison.
    """
    return tuple(
        normalize_for_comparison(os.path.realpath(root)) for root in roots
    )


def is_under_temp_path(path: str) -> bool:
    """Check whether a path *resolves* to somewhere under a system temp dir.

    Both sides are canonicalised before comparison, and that is load-bearing
    rather than tidy.  Comparing the path as written admits a symlink for
    where the link **lives** instead of where it **points**: a link at
    ``/tmp/x`` aimed at ``~/.ssh/id_rsa`` reads as "under /tmp", and the temp
    allowance short-circuits the workspace check below it, so the content of
    a file outside both /tmp and the workspace comes back allowed (jaato
    issue #669).  An allow rule has to resolve for the same reason a deny
    rule does.

    Resolving the roots too is what keeps this correct on macOS, where
    ``/tmp`` is itself a symlink to ``/private/tmp``: a resolved path
    compared against an unresolved root would reject every real temp path
    there.

    Args:
        path: Path to check (absolute; a relative path resolves against CWD).

    Returns:
        True if the path's resolved target is at or under a temp directory.
    """
    real = normalize_for_comparison(os.path.realpath(path))
    for temp_root in _resolved_temp_roots(tuple(SYSTEM_TEMP_PATHS)):
        if real == temp_root or real.startswith(temp_root + '/'):
            return True
    return False


def is_proc_attr_path(path: str) -> bool:
    """Return ``True`` for a procfs *security attribute* path.

    ``/proc/**/attr/**`` and ``/proc/**/task/**/attr/**`` carry process
    security attributes.  Writing to ``/proc/self/attr/current`` with the
    literal string ``changeprofile unconfined`` transitions the current
    task out of any AppArmor profile that permits the
    ``change_profile -> unconfined`` capability — including the
    ``jaato-ws-*`` per-session profiles, which need that capability for
    the framework's ``apparmor_confine.__exit__`` to restore unconfined
    when a tool finishes.  Without an application-layer denylist, an agent
    driving a path-taking in-process tool (file_edit, readFile,
    glob_files) at ``/proc/self/attr/current`` can escape confinement
    entirely.  Verified empirically against apparmor.py template v10
    (2026-05-01).

    Callers must check both the literal and the ``realpath``-resolved
    form: a symlink into ``/proc`` reaches the same file.

    See ``server/apparmor.py:296`` for the kernel-level analysis and
    ``project_backlog_apparmor_child_subprofile`` for the subprocess-side
    fix that complements this in-process gate.

    Args:
        path: An absolute path (literal or resolved).

    Returns:
        ``True`` when the path names a process security attribute.
    """
    if path.startswith(("/proc/self/attr/", "/proc/thread-self/attr/")):
        return True
    if not path.startswith("/proc/"):
        return False
    # parts: ['', 'proc', '<pid>', 'attr', ...] OR
    #        ['', 'proc', '<pid>', 'task', '<tid>', 'attr', ...]
    parts = path.split("/")
    if len(parts) >= 4 and parts[3] == "attr":
        return True
    return len(parts) >= 6 and parts[3] == "task" and parts[5] == "attr"


def check_path_with_jaato_containment(
    path: str,
    workspace_root: str,
    plugin_registry=None,
    allow_tmp: bool = True,
    mode: str = "read"
) -> bool:
    """Check if a path is allowed, with special .jaato containment handling.

    This is the main entry point for path validation that respects:
    1. Denied paths (checked first, takes precedence over all other rules)
    2. Standard POSIX pseudo-devices (``/dev/null``, ``/dev/stdout``,
       ``/dev/fd/<n>``, ...) -- always allowed; see
       :func:`is_pseudo_device_path`
    3. .jaato restriction: denied by default, requires explicit authorization
       via ``sandbox add`` (registered in plugin registry). Even when authorized,
       containment checks still apply (no traversal escapes, no nested symlinks).
       This takes precedence over /tmp allowance.
    4. System temp directories (/tmp) when allow_tmp=True
    5. Standard workspace sandboxing (paths must be within workspace)
    6. Plugin registry authorization (for external paths, respects access mode)

    Args:
        path: Path to check (absolute or will be made absolute).
        workspace_root: The workspace root directory.
        plugin_registry: Optional PluginRegistry for external path authorization
                        and denial checking.
        allow_tmp: Whether to allow /tmp/** access (default: True).
        mode: Access mode being requested - "read" or "write" (default: "read").
             This is used when checking authorized external paths to respect
             their access level (e.g., a "readonly" path blocks write access).

    Returns:
        True if path is allowed, False otherwise.
    """
    # Convert MSYS2 drive paths (/c/...) to Windows (C:/...) for Python
    path = msys2_to_windows_path(path)

    # Make path absolute
    abs_path = os.path.abspath(path)
    real_abs_path = os.path.realpath(abs_path)

    # Hard denylist — applies even when workspace_root is unset.
    if is_proc_attr_path(abs_path) or is_proc_attr_path(real_abs_path):
        return False

    if not workspace_root:
        # No further sandboxing configured (after the global denylist above).
        return True

    # Check if path is explicitly denied (takes precedence over all other rules)
    if plugin_registry and hasattr(plugin_registry, 'is_path_denied'):
        if plugin_registry.is_path_denied(abs_path):
            return False

    # Standard POSIX pseudo-devices are allowed once an explicit deny rule
    # has had its say.  Checked on the pre-resolution path: /dev/stdin and
    # friends are symlinks through /proc/self/fd/<n>, so their realpath is
    # whatever the descriptor points at and would fail the workspace test
    # for reasons that have nothing to do with the sandbox's purpose.
    if is_pseudo_device_path(path):
        return True

    # IMPORTANT: Check if path references .jaato BEFORE /tmp or workspace checks.
    # .jaato is denied by default and this takes precedence over /tmp allowance.
    # This also catches traversal attacks like .jaato/../secret.txt
    if is_jaato_path(path, workspace_root):
        # .jaato is DENIED BY DEFAULT. The model cannot access .jaato unless
        # the user explicitly authorizes it via "sandbox add".
        jaato_boundary = get_jaato_boundary(workspace_root)
        if jaato_boundary is None:
            # .jaato doesn't exist - path can't exist either
            return False

        # Make path absolute for the boundary check
        if not os.path.isabs(path):
            abs_path = os.path.join(workspace_root, path)
        else:
            abs_path = path

        # Security: containment must pass even if authorized
        if not is_path_within_jaato_boundary(abs_path, workspace_root, jaato_boundary):
            return False

        # Only allow if explicitly authorized via plugin registry
        # (populated by "sandbox add" command)
        if plugin_registry and hasattr(plugin_registry, 'is_path_authorized'):
            real_path = os.path.realpath(abs_path)
            if plugin_registry.is_path_authorized(real_path, mode=mode):
                return True

        # Not authorized - denied by default
        return False

    # Check whether the path RESOLVES to somewhere under /tmp (allowed by
    # default).  This branch short-circuits the workspace check below it, so
    # it must be decided on the resolved target — see is_under_temp_path.
    if allow_tmp and is_under_temp_path(abs_path):
        return True

    # Standard workspace check - resolve symlinks
    # Use normalized comparison to handle mixed separators (MSYS2/Windows)
    real_path = os.path.realpath(abs_path)
    norm_real = normalize_for_comparison(real_path)
    norm_workspace = normalize_for_comparison(workspace_root).rstrip('/') + '/'
    if norm_real == normalize_for_comparison(workspace_root) or norm_real.startswith(norm_workspace):
        return True

    # Check if authorized via plugin registry (for external paths)
    # Pass the mode so that "readonly" paths block write access
    if plugin_registry and hasattr(plugin_registry, 'is_path_authorized'):
        if plugin_registry.is_path_authorized(real_path, mode=mode):
            return True

    return False
