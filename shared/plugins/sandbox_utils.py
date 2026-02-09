"""Shared sandbox utilities for path validation.

This module provides common path validation logic used by multiple plugins
(file_edit, cli, etc.) to enforce workspace sandboxing with special handling
for the .jaato configuration directory.

Key feature: .jaato Contained Symlink Escape
============================================
The .jaato directory is allowed to be a symlink pointing outside the workspace,
but once inside .jaato, paths cannot escape that boundary. Nested symlinks
inside .jaato are NOT allowed.

Example:
    Workspace: /home/user/project/
    .jaato symlink: /home/user/project/.jaato -> /home/user/.jaato (external)

    ALLOWED:
        .jaato/config.json     -> /home/user/.jaato/config.json
        .jaato/vision/img.png  -> /home/user/.jaato/vision/img.png

    BLOCKED:
        .jaato/../secret.txt   -> /home/user/secret.txt (escapes boundary)
        .jaato/plugins -> /opt  (nested symlink, not allowed)

Key feature: /tmp Access
========================
The /tmp directory is allowed by default for sandboxed tools to support
temporary file operations. This can be disabled via the allow_tmp parameter.
"""

import os
import tempfile
from typing import Optional, Tuple

from shared.path_utils import msys2_to_windows_path, normalize_for_comparison


# The special configuration directory that gets contained symlink escape
JAATO_CONFIG_DIR = ".jaato"

# System temp directories that are allowed by default
SYSTEM_TEMP_PATHS = ["/tmp", tempfile.gettempdir()]


def is_jaato_path(path: str, workspace_root: str) -> bool:
    """Check if a path references the .jaato directory.

    This checks if the path goes through .jaato, even if it later escapes
    via .. traversal. This is important because:
    - .jaato/../secret.txt should be treated as a .jaato path attempt
    - /workspace/.jaato/../../etc/passwd should be treated as .jaato path attempt

    Args:
        path: Path to check (may contain .. traversal).
        workspace_root: The workspace root directory.

    Returns:
        True if path references .jaato (directly or via traversal).
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

    # Also check if the path CONTAINS .jaato as a component
    # This catches cases like /workspace/.jaato/../secret where abspath
    # would normalize away the .jaato reference
    path_parts = path.replace('\\', '/').split('/')
    if JAATO_CONFIG_DIR in path_parts:
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


def is_under_temp_path(path: str) -> bool:
    """Check if a path is under a system temp directory.

    Args:
        path: Path to check (should be absolute or normalized).

    Returns:
        True if path is under /tmp or system temp directory.
    """
    normalized = normalize_for_comparison(os.path.normpath(path))
    for temp_path in SYSTEM_TEMP_PATHS:
        temp_normalized = normalize_for_comparison(os.path.normpath(temp_path))
        if normalized == temp_normalized or normalized.startswith(temp_normalized + '/'):
            return True
    return False


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
    2. Standard workspace sandboxing (paths must be within workspace)
    3. Special .jaato handling (symlink allowed, but contained)
    4. Plugin registry authorization (for external paths, respects access mode)
    5. System temp directories (/tmp) when allow_tmp=True

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
    if not workspace_root:
        # No sandboxing configured
        return True

    # Convert MSYS2 drive paths (/c/...) to Windows (C:/...) for Python
    path = msys2_to_windows_path(path)

    # Make path absolute
    abs_path = os.path.abspath(path)

    # Check if path is explicitly denied (takes precedence over all other rules)
    if plugin_registry and hasattr(plugin_registry, 'is_path_denied'):
        if plugin_registry.is_path_denied(abs_path):
            return False

    # Check if path is under /tmp (allowed by default)
    if allow_tmp and is_under_temp_path(abs_path):
        return True

    # IMPORTANT: Check if path references .jaato BEFORE normalizing
    # This catches traversal attacks like .jaato/../secret.txt
    if is_jaato_path(path, workspace_root):
        # Special .jaato handling
        jaato_boundary = get_jaato_boundary(workspace_root)
        if jaato_boundary is None:
            # .jaato doesn't exist - path can't exist either
            return False

        # Make path absolute for the boundary check
        if not os.path.isabs(path):
            abs_path = os.path.join(workspace_root, path)
        else:
            abs_path = path

        return is_path_within_jaato_boundary(abs_path, workspace_root, jaato_boundary)

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
