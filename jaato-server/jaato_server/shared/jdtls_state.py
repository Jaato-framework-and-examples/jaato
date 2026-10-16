"""The jdtls state sibling: when it is made, and who removes it (#1710).

Eclipse forbids jdtls's ``-data`` directory inside an imported project, so
the framework offers ``${jdtlsStateRoot}``: a SIBLING of the workspace,
``<workspace.parent>/.<workspace.basename>-jdtls-state``
(:func:`jaato_server.shared.plugins.subagent.config._compute_jdtls_state_root`).
A session writes it BESIDE its workspace rather than in it, in a directory
the application owns (the workspace's parent).

Until #1710 the daemon created it for every session that had a workspace,
whatever the session could run, and nothing removed it.  The rules now:

- **made only when a language server is configured to use it**: the LSP
  configuration the session would load (``plugin_configs.lsp.languageServers``,
  else ``config_path`` / ``<workspace>/.lsp.json`` / ``~/.lsp.json``, the
  search the lsp plugin and its AppArmor composer use) names
  ``jdtlsStateRoot``.  A session that cannot start such a server gets no
  sibling;
- **made again at the server's start** when it is missing (a ``.lsp.json``
  written mid-session, or one removed since), by the lsp plugin, best effort;
- **removed with the workspace**: ``workspace.delete`` removes it beside the
  workspace directory.  An application that deletes a workspace itself calls
  :func:`remove_jdtls_state_root` or removes :func:`jdtls_state_root_of`.

:data:`JDTLS_STATE_RULES` is the text ``explain paths`` renders, so the
documentation is read from here rather than restated.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

#: The ``${...}`` template variable that names the sibling.
JDTLS_STATE_VAR = "jdtlsStateRoot"

#: Lines ``jaato-scaffold explain paths`` renders under the sibling.
JDTLS_STATE_RULES = (
    "made only when the session's LSP configuration (plugin_configs.lsp."
    "languageServers, else .lsp.json) names ${jdtlsStateRoot}",
    "made again by the lsp plugin when such a server starts and it is missing",
    "removed by workspace.delete; an application deleting a workspace "
    "itself removes it too (jaato_server.shared.jdtls_state."
    "remove_jdtls_state_root)",
)


def jdtls_state_root_of(workspace_path: Optional[str]) -> str:
    """The sibling's absolute path for *workspace_path* (``""`` for none)."""
    from jaato_server.shared.plugins.subagent.config import (
        _compute_jdtls_state_root,
    )
    return _compute_jdtls_state_root(workspace_path or "")


def lsp_config_uses_jdtls_state(
    workspace_path: Optional[str],
    lsp_plugin_config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Whether the LSP configuration this session would load names the sibling.

    Reads what the lsp plugin's AppArmor composer reads
    (``LSPToolPlugin._load_lsp_config_static``), so the daemon provisions
    exactly when the composer grants.  Any failure to read answers ``False``:
    a session whose configuration cannot be read cannot start a server from
    it either, and the lsp plugin makes the directory at start if one does.
    """
    try:
        from jaato_server.shared.plugins.lsp.plugin import LSPToolPlugin
        config = LSPToolPlugin._load_lsp_config_static(
            workspace_path, dict(lsp_plugin_config or {}))
    except Exception:  # noqa: BLE001 - a predicate, never a failure
        logger.debug("jdtls state: could not read the LSP configuration",
                     exc_info=True)
        return False
    if not isinstance(config, dict):
        return False
    servers = config.get("languageServers")
    if not servers:
        return False
    try:
        text = json.dumps(servers, default=str)
    except (TypeError, ValueError):
        return False
    return "${" + JDTLS_STATE_VAR + "}" in text


def remove_jdtls_state_root(workspace_path: Optional[str]) -> bool:
    """Remove *workspace_path*'s sibling, if it is a real directory.

    A symlink at that name is left alone (never followed: the parent is an
    application's directory, and the name is predictable).  Returns whether
    something was removed.  Errors are logged, never raised: the workspace
    itself is already gone when this runs.
    """
    root = jdtls_state_root_of(workspace_path)
    if not root:
        return False
    try:
        st = os.lstat(root)
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("jdtls state: cannot inspect %s: %s", root, exc)
        return False
    if not stat.S_ISDIR(st.st_mode):
        logger.warning(
            "jdtls state: %s is not a directory (a link or a file); left in "
            "place", root)
        return False
    try:
        shutil.rmtree(root)
    except OSError as exc:
        logger.warning("jdtls state: could not remove %s: %s", root, exc)
        return False
    logger.info("jdtls state: removed %s", root)
    return True


def ensure_jdtls_state_root(state_root: str) -> Optional[str]:
    """``mkdir -p`` *state_root*; ``None`` on success, else the error text."""
    if not state_root:
        return None
    try:
        os.makedirs(state_root, exist_ok=True)
    except OSError as exc:
        return str(exc)
    return None


def profile_lsp_config(profile: Any) -> Dict[str, Any]:
    """``plugin_configs.lsp`` of a resolved profile (``{}`` when absent)."""
    configs = getattr(profile, "plugin_configs", None) or {}
    block = configs.get("lsp") if isinstance(configs, dict) else None
    return dict(block) if isinstance(block, dict) else {}
